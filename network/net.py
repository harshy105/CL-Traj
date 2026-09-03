import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch import Tensor
from typing import Dict, Optional, Tuple
from copy import deepcopy
from pytorch_lightning import LightningModule

from losses.multivariate_loss import MultiVariateLoss
from losses.mixture_multivariate_loss import MixtureMultivariateLoss
from metrics.pred_metrics import PredMetric
from metrics.planning_metrics import PlanningMetric
from network.modules.map_encoder import MapEncoder
from network.modules.agent_encoder import AgentEncoder
from network.modules.decoder import Decoder
from config.train_config import DataStructureConfig, NetConfig, TrainingConfig
from utilities.transformation import (
    angle_between_2d_vectors,
    simulate_single_trajectory_per_agent_,
    recurr_to_vcs_to_local_tensor,
    vcs_to_local_tensor,
    wrap_angle_rad,
)
from utilities.utils import change_device_of_dict, combine_batch_elements_dim, unbatch_and_pad


class Net(LightningModule):
    def __init__(self, data_config: DataStructureConfig, net_config: NetConfig, train_config: TrainingConfig):
        super(Net, self).__init__()
        self.train_config = train_config
        self.use_target_net = train_config.use_target_net
        self.use_scene_net = train_config.use_scene_net
        self.diff_simulator = train_config.diff_simulator
        self.head_regularization_scale = train_config.head_regularization_scale
        self.cls_loss_scale = train_config.cls_loss_scale
        self.reg_loss_scale = train_config.reg_loss_scale
        self.target_loss_scale = train_config.target_loss_scale
        self.scene_loss_scale = train_config.scene_loss_scale
        self.target_cl_loss_scale = train_config.target_cl_loss_scale
        self.scene_cl_loss_scale = train_config.scene_cl_loss_scale
        self.num_train_recurr_steps = train_config.num_train_recurr_steps
        self.num_train_sim_steps = data_config.num_future_steps // train_config.num_train_recurr_steps
        self.num_eval_recurr_steps = train_config.num_eval_recurr_steps
        self.num_eval_sim_steps = data_config.num_future_steps // train_config.num_eval_recurr_steps
        
        self.current_frame = data_config.current_frame
        self.sample_frequency = data_config.sample_frequency
        self.num_future_steps = data_config.num_future_steps
        
        self.output_dim = net_config.output_dim
        self.target_num_modes = net_config.target_num_modes
        self.scene_num_modes = net_config.scene_num_modes
        self.max_num_modes = max(net_config.target_num_modes * self.use_target_net, 
                                 net_config.scene_num_modes * self.use_scene_net)
        self.num_layers = net_config.num_layers
        assert net_config.max_num_input_frames > self.current_frame, "using all the input frames in ol sample"

        self.map_encoder = MapEncoder(hidden_dim=net_config.hidden_dim)
        self.agent_encoder = AgentEncoder(
            hidden_dim=net_config.hidden_dim,
            rel_dis_norm_factor=data_config.rel_dis_norm_factor,
            vel_norm_factor=data_config.vel_norm_factor,
            max_num_input_frames=net_config.max_num_input_frames,
        )
        if self.use_target_net:
            self.target_decoder = Decoder(
                num_modes=net_config.target_num_modes,
                hidden_dim=net_config.hidden_dim,
                num_heads=net_config.num_heads,
                dropout=train_config.dropout,
                head_dim=net_config.head_dim,
                use_goal=data_config.target_goal,
                max_num_input_frames=net_config.max_num_input_frames,
                num_dec_recurr_steps=train_config.num_dec_recurr_steps,
                num_layers=net_config.num_layers,
                num_dec_future_steps=net_config.num_dec_future_steps,
                output_dim=net_config.output_dim,
                rel_dis_norm_factor=data_config.rel_dis_norm_factor,
                dis_threshold=train_config.dis_threshold,
                modes_noise_factor=train_config.modes_noise_factor,
            )
        if self.use_scene_net:
            self.scene_decoder = Decoder(
                num_modes=net_config.scene_num_modes,
                hidden_dim=net_config.hidden_dim,
                num_heads=net_config.num_heads,
                dropout=train_config.dropout,
                head_dim=net_config.head_dim,
                use_goal=data_config.surr_goal,
                max_num_input_frames=net_config.max_num_input_frames,
                num_dec_recurr_steps=train_config.num_dec_recurr_steps,
                num_layers=net_config.num_layers,
                num_dec_future_steps=net_config.num_dec_future_steps,
                output_dim=net_config.output_dim,
                rel_dis_norm_factor=data_config.rel_dis_norm_factor,
                dis_threshold=train_config.dis_threshold,
                modes_noise_factor=train_config.modes_noise_factor,
            )
        self.reg_loss = MultiVariateLoss()
        self.cls_loss = MixtureMultivariateLoss()
        self.time_of_interest = self.current_frame + torch.tensor([2, 6, 12])
        self.planning_metrics_holder = PlanningMetric(len(self.time_of_interest))
        self.pred_metrics_holder = PredMetric()

    def forward(self, current_frame: int, data: Dict) -> Dict:
        map_enc = self.map_encoder(data)
        agent_enc = self.agent_encoder(current_frame, data)
        scene_enc = {**map_enc, **agent_enc}
        if self.use_target_net:
            target_agent_mask = data["agents_target_mask"].bool() # (A,)
            pred_target = self.target_decoder(current_frame, target_agent_mask, data, scene_enc)
        if self.use_scene_net:
            surr_agents_sim_mask = data["agents_surr_sim_mask"].bool() # (A,)
            pred_surr = self.scene_decoder(current_frame, surr_agents_sim_mask, data, scene_enc)
            
        if self.use_target_net and not self.use_scene_net:
            pred = pred_target
        elif not self.use_target_net and self.use_scene_net:
            pred = pred_surr
        elif self.use_target_net and self.use_scene_net:
            pred = {}
            for k in pred_target.keys():
                assert pred_target[k].shape[1] > pred_surr[k].shape[1], "Target agent must have more modes than surr agents"
                assert pred_surr[k].shape[1] == 1, "Surr agents only have one mode"
                pred[k] = torch.zeros_like(pred_target[k])
                pred[k][target_agent_mask] = pred_target[k][target_agent_mask]
                pred[k][surr_agents_sim_mask] = pred_surr[k][surr_agents_sim_mask] # broacasting along modes
        else:
            raise ValueError("Neither of the Networks are used in forward pass")
        return pred

    def training_step(self, data: Dict, batch_idx) -> torch.FloatTensor:
        data = combine_batch_elements_dim(data)
        gt = data["agents_gt_position"]  # (A, T, 2)
        gt_reg_mask = data["agents_log_reg_mask"][:, self.current_frame + 1 :].bool()  # (A, T)
        if self.use_target_net and not self.use_scene_net:
            sim_mask = data["agents_target_mask"].bool() # (A,)
        elif not self.use_target_net and self.use_scene_net:
            sim_mask = data["agents_surr_sim_mask"].bool() # (A,)
        elif self.use_target_net and self.use_scene_net:
           sim_mask = torch.bitwise_or(data["agents_target_mask"].bool(), data["agents_surr_sim_mask"].bool()) # (A,)
        else:
            raise ValueError("Neither of the Networks are used in training")

        num_agents = data["agents_position"].shape[0]
        DEVICE = data["agents_position"].device
        ol_pos_mu_recurr = torch.zeros(
            num_agents, self.max_num_modes, self.num_layers, self.num_train_recurr_steps,
            self.num_future_steps, self.output_dim, device=DEVICE,
        )
        ol_pos_scale_recurr = torch.zeros(
            num_agents, self.max_num_modes, self.num_layers, self.num_train_recurr_steps,
            self.num_future_steps, self.output_dim, self.output_dim, device=DEVICE,
        )
        ol_best_pos_mu_local = torch.zeros(
            num_agents, self.num_layers, self.num_train_recurr_steps, 
            self.num_future_steps, self.output_dim, device=DEVICE
        )
        ol_best_pos_scale_local = torch.zeros(
            num_agents, self.num_layers, self.num_train_recurr_steps, 
            self.num_future_steps, self.output_dim, self.output_dim, device=DEVICE
        )

        current_frame = self.current_frame
        data_sim = deepcopy(data)
        for itr in range(self.num_train_recurr_steps):
            pred = self(current_frame, data_sim)
            if itr == 0:
                ol_pos_mu_recurr[:, :, :, itr, :, :] = pred["traj_pos_mu"]  # fill ol output for all the modes
                ol_pos_scale_recurr[:, :, :, itr, :, :, :] = pred["traj_pos_scale"]
                pi = pred["pi"]  # (A, M)
                l2_norm = (torch.norm(pred["traj_pos_mu"][:, :, -1, :, : self.output_dim] - gt[:, None, :, : self.output_dim],
                        p=2, dim=-1)* gt_reg_mask.unsqueeze(1)).sum(dim=-1)
                best_mode = l2_norm.argmin(dim=-1)
            else:
                ol_pos_mu_recurr[torch.arange(num_agents), best_mode, :, itr, :, :] = \
                    pred["traj_pos_mu"][torch.arange(num_agents), best_mode]  # fill the ol output for the best mode
                ol_pos_scale_recurr[torch.arange(num_agents), best_mode, :, itr, :, :, :] = \
                    pred["traj_pos_scale"][torch.arange(num_agents), best_mode]

            # perform coordinate transformation
            ol_best_pos_mu_recurr_itr = ol_pos_mu_recurr[torch.arange(num_agents), best_mode, :, itr, :, :]  # (A, L, T, 2)
            ol_best_pos_mu_recurr_itr = ol_best_pos_mu_recurr_itr.reshape(num_agents*self.num_layers, self.num_future_steps, self.output_dim) # (A*L, T, 2)
            ol_best_pos_scale_recurr_itr = ol_pos_scale_recurr[torch.arange(num_agents), best_mode, :, itr, :, :, :]  # (A, L, T, 2, 2)
            ol_best_pos_scale_recurr_itr = ol_best_pos_scale_recurr_itr.reshape(num_agents*self.num_layers, self.num_future_steps, 
                                                                                self.output_dim, self.output_dim) # (A*L, T, 2, 2)
            (ol_best_pos_mu_vcs_itr, ol_best_head_mu_vcs_itr, ol_best_vel_mu_vcs_itr, ol_best_pos_mu_local_itr,
            ol_best_pos_scale_local_itr) = recurr_to_vcs_to_local_tensor(
                recurr_vcs_pos = data_sim["agents_position"][:, current_frame, :].repeat_interleave(self.num_layers, dim=0),
                recurr_vcs_head_rad = data_sim["agents_heading_rad"][:, current_frame].repeat_interleave(self.num_layers, dim=0),
                local_vcs_pos = data_sim["agents_position"][:, self.current_frame, :].repeat_interleave(self.num_layers, dim=0),
                local_vcs_head_rad = data_sim["agents_heading_rad"][:, self.current_frame].repeat_interleave(self.num_layers, dim=0),
                traj_recurr_pos = ol_best_pos_mu_recurr_itr,
                scale_recurr_pos=ol_best_pos_scale_recurr_itr,
                sample_frequency = self.sample_frequency,
            ) # (A*L, T, 2), (A*L, T), (A*L, T, 2), (A*L, T, 2), (A*L, T, 2, 2)
            ol_best_pos_mu_local[:, :, itr, :, :] = ol_best_pos_mu_local_itr.reshape(num_agents, self.num_layers, 
                                                                self.num_future_steps, self.output_dim)
            ol_best_pos_scale_local[:, :, itr, :, :, :] = ol_best_pos_scale_local_itr.reshape(num_agents, self.num_layers, 
                                                                self.num_future_steps, self.output_dim, self.output_dim)
            
            # simulate the trajectory and update training data
            ol_best_last_pos_mu_vcs_itr = \
                ol_best_pos_mu_vcs_itr.reshape(num_agents, self.num_layers, self.num_future_steps, self.output_dim)[:, -1, :, :] # (A, T, 2)
            ol_best_last_head_mu_vcs_itr = \
                ol_best_head_mu_vcs_itr.reshape(num_agents, self.num_layers, self.num_future_steps)[:, -1, :] # (A, T)
            ol_best_last_vel_mu_vcs_itr = \
                ol_best_vel_mu_vcs_itr.reshape(num_agents, self.num_layers, self.num_future_steps, self.output_dim)[:, -1, :, :] # (A, T, 2)
            simulate_single_trajectory_per_agent_(
                data = data_sim,
                sim_mask = sim_mask,
                current_frame = current_frame,
                num_sim_steps = self.num_train_sim_steps,
                trajectories_vcs_pos = ol_best_last_pos_mu_vcs_itr if self.diff_simulator else ol_best_last_pos_mu_vcs_itr.detach(),
                trajectories_vcs_head_rad = ol_best_last_head_mu_vcs_itr if self.diff_simulator else ol_best_last_head_mu_vcs_itr.detach(),
                trajectories_vcs_vel = ol_best_last_vel_mu_vcs_itr if self.diff_simulator else ol_best_last_vel_mu_vcs_itr.detach(),
                diff_simulator=self.diff_simulator,
            )

            # update the current frame
            current_frame += self.num_train_sim_steps

        ol_last_pos_mu_local_itr_0 = ol_pos_mu_recurr[:, :, -1, 0, :, :] # (A, M, T, 2)
        ol_last_pos_scale_local_itr_0 = ol_pos_scale_recurr[:, :, -1, 0, :, :, :] # (A, M, T, 2, 2)
        
        loss = 0
        if self.use_target_net:
            target_agent_mask = data["agents_target_mask"].bool() # (A,)
            target_loss = self.compute_loss(
                ol_last_pos_mu_local_itr_0=ol_last_pos_mu_local_itr_0[target_agent_mask],
                ol_last_pos_scale_local_itr_0=ol_last_pos_scale_local_itr_0[target_agent_mask],
                ol_best_pos_mu_local=ol_best_pos_mu_local[target_agent_mask],
                ol_best_pos_scale_local=ol_best_pos_scale_local[target_agent_mask],
                pi=pi[target_agent_mask],
                gt_local=gt[target_agent_mask],
                gt_reg_mask=gt_reg_mask[target_agent_mask],
                cls_mask=target_agent_mask[target_agent_mask],
                use_cls_loss=True,
                cl_loss_scale=self.target_cl_loss_scale,
                data_split="train",
            )
            loss += self.target_loss_scale * target_loss
            
        if self.use_scene_net:
            surr_agents_sim_mask = data["agents_surr_sim_mask"].bool()
            surr_loss = self.compute_loss(
                ol_last_pos_mu_local_itr_0=ol_last_pos_mu_local_itr_0[surr_agents_sim_mask],
                ol_last_pos_scale_local_itr_0=ol_last_pos_scale_local_itr_0[surr_agents_sim_mask],
                ol_best_pos_mu_local=ol_best_pos_mu_local[surr_agents_sim_mask],
                ol_best_pos_scale_local=ol_best_pos_scale_local[surr_agents_sim_mask],
                pi=pi[surr_agents_sim_mask],
                gt_local=gt[surr_agents_sim_mask],
                gt_reg_mask=gt_reg_mask[surr_agents_sim_mask],
                cls_mask=None,
                use_cls_loss=False,
                cl_loss_scale=self.scene_cl_loss_scale,
                data_split="train",
                agent_type="scene"
            )
            loss += self.scene_loss_scale * surr_loss
            
        return loss

    def validation_step(self, data: Dict, batch_idx) -> torch.FloatTensor:
        data = combine_batch_elements_dim(data)
        gt = data["agents_gt_position"]  # (A, T, 2)
        gt_reg_mask = data["agents_log_reg_mask"][:, self.current_frame + 1 :].bool()  # (A, T)
        if self.use_target_net:
            target_agent_mask = data["agents_target_mask"].bool() # (A,)
            (
                pi, ol_last_pos_mu_local_itr_0, 
                ol_last_pos_scale_local_itr_0, 
                ol_best_pos_mu_local, ol_best_pos_scale_local, 
                cl_last_pos_mu_vcs_detached,
                cl_last_head_mu_vcs_detached,
                cl_last_pos_mu_local_detached, 
            ) = self.validation_rollout(
                num_modes=self.target_num_modes,
                sim_mask=target_agent_mask,
                gt_reg_mask=gt_reg_mask,
                gt=gt,
                data=data,
            )
            target_loss = self.compute_loss(
                ol_last_pos_mu_local_itr_0=ol_last_pos_mu_local_itr_0[target_agent_mask],
                ol_last_pos_scale_local_itr_0=ol_last_pos_scale_local_itr_0[target_agent_mask],
                ol_best_pos_mu_local=ol_best_pos_mu_local[target_agent_mask],
                ol_best_pos_scale_local=ol_best_pos_scale_local[target_agent_mask],
                pi=pi[target_agent_mask],
                gt_local=gt[target_agent_mask],
                gt_reg_mask=gt_reg_mask[target_agent_mask],
                cls_mask=target_agent_mask[target_agent_mask],
                use_cls_loss=True,
                cl_loss_scale=self.target_cl_loss_scale,
                data_split="val",
            )
            # compute the open loop metrics in GT's local coordinate frame
            ol_traj_target = ol_last_pos_mu_local_itr_0[target_agent_mask] # (A_target, M, T, 2)
            gt_target = gt[target_agent_mask]
            gt_target_reg_mask = gt_reg_mask[target_agent_mask]
            pi_target = pi[target_agent_mask]

            self.pred_metrics_holder.update(ol_traj_target, gt_target, gt_target_reg_mask, pi_target)
            min_ade_1, min_fde_1, mr_1 = self.pred_metrics_holder.compute(n=1)
            min_ade_5, min_fde_5, mr_5 = self.pred_metrics_holder.compute(n=5)
            self.pred_metrics_holder.reset()
            self.log(f"val_minADE_1", min_ade_1, sync_dist=True, batch_size=1, prog_bar=True)
            self.log(f"val_minADE", min_ade_5, sync_dist=True, batch_size=1, prog_bar=True)
            self.log(f"val_minFDE_1", min_fde_1, sync_dist=True, batch_size=1, prog_bar=True)
            self.log(f"val_minFDE", min_fde_5, sync_dist=True, batch_size=1, prog_bar=True)
            
            # compute the close loop prediction metrics in GT's local coordinate frame
            cl_traj_target = cl_last_pos_mu_local_detached[target_agent_mask]  # (A_target, M, T, 2)
            self.pred_metrics_holder.update(cl_traj_target, gt_target, gt_target_reg_mask, pi_target)
            min_ade_1, min_fde_1, mr_1 = self.pred_metrics_holder.compute(n=1)
            min_ade_5, min_fde_5, mr_5 = self.pred_metrics_holder.compute(n=5)
            self.pred_metrics_holder.reset()
            self.log(f"val_minADE_1_cl", min_ade_1, sync_dist=True, batch_size=1, prog_bar=True)
            self.log(f"val_minADE_cl", min_ade_5, sync_dist=True, batch_size=1, prog_bar=True)
            self.log(f"val_minFDE_1_cl", min_fde_1, sync_dist=True, batch_size=1, prog_bar=True)
            self.log(f"val_minFDE_cl", min_fde_5, sync_dist=True, batch_size=1, prog_bar=True)

            # compute the close loop planning metrics for target agent in a vcs
            unbatched_data = unbatch_and_pad(data)
            B = unbatched_data["agents_position"].shape[0]
            assert pi_target.shape[0] == B

            cl_last_pos_head_mu_vcs_detached_target = torch.concat(
                [cl_last_pos_mu_vcs_detached, cl_last_head_mu_vcs_detached.unsqueeze(-1)], 
                dim=-1)[target_agent_mask]
            assert cl_last_pos_head_mu_vcs_detached_target.shape[0] == B
            
            self.planning_metrics_holder.update(
                ego_trajs=cl_last_pos_head_mu_vcs_detached_target[:, :, self.time_of_interest - self.current_frame - 1, :],
                ego_mode_probs=pi_target,
                ego_gt_trajs=torch.concat(
                    [unbatched_data["agents_position"][:, 0, self.time_of_interest, :2],
                    unbatched_data["agents_heading_rad"][:, 0, self.time_of_interest, None]],
                    dim=-1),
                ego_wh_trajs=unbatched_data["agents_size"][:, 0, self.time_of_interest, :2],
                ego_gt_trajs_mask=unbatched_data["agents_log_reg_mask"][:, 0, self.time_of_interest],
                others_gt_trajs=torch.concat(
                    [unbatched_data["agents_position"][:, 1:, self.time_of_interest, :2],
                    unbatched_data["agents_heading_rad"][:, 1:, self.time_of_interest, None]],
                    dim=-1),
                others_wh_trajs=unbatched_data["agents_size"][:, 1:, self.time_of_interest, :2],
                others_gt_trajs_mask=unbatched_data["agents_log_reg_mask"][:, 1:, self.time_of_interest],
            )
            min_plan_metrics_1 = self.planning_metrics_holder.compute(n=1)
            self.planning_metrics_holder.reset()
            self.log(f"val_min_col_1sec_1_cl", min_plan_metrics_1["mean_box_col_percent"][0], sync_dist=True, batch_size=1, prog_bar=True)
            self.log(f"val_min_col_3sec_1_cl", min_plan_metrics_1["mean_box_col_percent"][1], sync_dist=True, batch_size=1, prog_bar=True)
            self.log(f"val_min_col_6sec_1_cl", min_plan_metrics_1["mean_box_col_percent"][2], sync_dist=True, batch_size=1, prog_bar=True)
            self.log(f"val_min_l2_1sec_1_cl", min_plan_metrics_1["min_L2"][0], sync_dist=True, batch_size=1, prog_bar=True)
            self.log(f"val_min_l2_3sec_1_cl", min_plan_metrics_1["min_L2"][1], sync_dist=True, batch_size=1, prog_bar=True)
            self.log(f"val_min_l2_6sec_1_cl", min_plan_metrics_1["min_L2"][2], sync_dist=True, batch_size=1, prog_bar=True)
        
        if self.use_scene_net:
            surr_agents_sim_mask = data["agents_surr_sim_mask"].bool() # (A,)
            (
                pi, ol_last_pos_mu_local_itr_0, 
                ol_last_pos_scale_local_itr_0, 
                ol_best_pos_mu_local, ol_best_pos_scale_local, 
                _, _, cl_last_pos_mu_local_detached
            ) = self.validation_rollout(
                num_modes=self.scene_num_modes,
                sim_mask=surr_agents_sim_mask,
                gt_reg_mask=gt_reg_mask,
                gt=gt,
                data=data,
            )
            if surr_agents_sim_mask.any():
                surr_loss = self.compute_loss(
                    ol_last_pos_mu_local_itr_0=ol_last_pos_mu_local_itr_0[surr_agents_sim_mask],
                    ol_last_pos_scale_local_itr_0=ol_last_pos_scale_local_itr_0[surr_agents_sim_mask],
                    ol_best_pos_mu_local=ol_best_pos_mu_local[surr_agents_sim_mask],
                    ol_best_pos_scale_local=ol_best_pos_scale_local[surr_agents_sim_mask],
                    pi=pi[surr_agents_sim_mask],
                    gt_local=gt[surr_agents_sim_mask],
                    gt_reg_mask=gt_reg_mask[surr_agents_sim_mask],
                    cls_mask=None,
                    use_cls_loss=False,
                    cl_loss_scale=self.scene_cl_loss_scale,
                    data_split="val",
                    agent_type="scene"
                )
                # compute the open loop metrics in GT's local coordinate frame
                ol_traj_surr = ol_last_pos_mu_local_itr_0[surr_agents_sim_mask] # (A_surr, M, T, 2)
                gt_surr = gt[surr_agents_sim_mask]
                gt_surr_reg_mask = gt_reg_mask[surr_agents_sim_mask]
                pi_surr = pi[surr_agents_sim_mask]

                self.pred_metrics_holder.update(ol_traj_surr, gt_surr, gt_surr_reg_mask, pi_surr)
                min_ade_1, min_fde_1, mr_1 = self.pred_metrics_holder.compute(n=1)
                self.pred_metrics_holder.reset()
                self.log(f"scene_val_minADE_1", min_ade_1, sync_dist=True, batch_size=1, prog_bar=True)
                self.log(f"scene_val_minFDE_1", min_fde_1, sync_dist=True, batch_size=1, prog_bar=True)
                
                # compute the close loop prediction metrics in GT's local coordinate frame
                cl_traj_surr = cl_last_pos_mu_local_detached[surr_agents_sim_mask]  # (A_surr, M, T, 2)
                self.pred_metrics_holder.update(cl_traj_surr, gt_surr, gt_surr_reg_mask, pi_surr)
                min_ade_1, min_fde_1, mr_1 = self.pred_metrics_holder.compute(n=1)
                self.pred_metrics_holder.reset()
                self.log(f"scene_val_minADE_1_cl", min_ade_1, sync_dist=True, batch_size=1, prog_bar=True)
                self.log(f"scene_val_minFDE_1_cl", min_fde_1, sync_dist=True, batch_size=1, prog_bar=True)
    
    def validation_rollout(self, num_modes: int, sim_mask: Tensor, gt_reg_mask: Tensor, 
                           gt: Tensor, data: Dict) -> Tuple[Tensor, ...]:
        num_agents = data["agents_position"].shape[0]
        DEVICE = data["agents_position"].device
        ol_pos_mu_recurr = torch.zeros(
            num_agents, num_modes, self.num_layers,  self.num_eval_recurr_steps,
            self.num_future_steps, self.output_dim, device=DEVICE,
        )
        ol_pos_scale_recurr = torch.zeros(
            num_agents, num_modes, self.num_layers, self.num_eval_recurr_steps,
            self.num_future_steps, self.output_dim, self.output_dim, device=DEVICE,
        )
        ol_pos_mu_local = torch.zeros(
            num_agents, num_modes, self.num_layers, self.num_eval_recurr_steps,
            self.num_future_steps, self.output_dim, device=DEVICE,
        )
        ol_pos_scale_local = torch.zeros(
            num_agents, num_modes, self.num_layers, self.num_eval_recurr_steps,
            self.num_future_steps, self.output_dim, self.output_dim, device=DEVICE,
        )

        pred = self(self.current_frame, data)
        ol_pos_mu_recurr[:, :, :, 0, :, :] = pred["traj_pos_mu"][:, : num_modes, :, :, :]  # remove the non predicted modes
        ol_pos_scale_recurr[:, :, :, 0, :, :, :] = pred["traj_pos_scale"][:, : num_modes, :, :, :, :] # remove the non predicted modes
        pi = pred["pi"][:, : num_modes]  # (A, M)
        data_sim_modes_list = []
        for mode in range(num_modes):
            current_frame = self.current_frame
            data_sim_modes_list.append(deepcopy(data))
            for itr in range(self.num_eval_recurr_steps):
                # perfom the coordinate transformation
                ol_mode_pos_mu_recurr_itr = ol_pos_mu_recurr[:, mode, :, itr, :, :]  # (A, L, T, 2)
                ol_mode_pos_mu_recurr_itr = ol_mode_pos_mu_recurr_itr.reshape(num_agents*self.num_layers, self.num_future_steps, self.output_dim) # (A*L, T, 2)
                ol_mode_pos_scale_recurr_itr = ol_pos_scale_recurr[:, mode, :, itr, :, :, :]  # (A, L, T, 2, 2)
                ol_mode_pos_scale_recurr_itr = ol_mode_pos_scale_recurr_itr.reshape(num_agents*self.num_layers, self.num_future_steps, self.output_dim, self.output_dim) # (A*L, T, 2, 2)
                (ol_mode_pos_mu_vcs_itr, ol_mode_head_mu_vcs_itr, ol_mode_vel_mu_vcs_itr, ol_mode_pos_mu_local_itr,
                ol_mode_pos_scale_local_itr) = recurr_to_vcs_to_local_tensor(
                    recurr_vcs_pos=data_sim_modes_list[-1]["agents_position"][:, current_frame, :].repeat_interleave(self.num_layers, dim=0),
                    recurr_vcs_head_rad=data_sim_modes_list[-1]["agents_heading_rad"][:, current_frame].repeat_interleave(self.num_layers, dim=0),
                    local_vcs_pos=data_sim_modes_list[-1]["agents_position"][:, self.current_frame, :].repeat_interleave(self.num_layers, dim=0),
                    local_vcs_head_rad=data_sim_modes_list[-1]["agents_heading_rad"][:, self.current_frame].repeat_interleave(self.num_layers, dim=0),
                    traj_recurr_pos=ol_mode_pos_mu_recurr_itr,
                    scale_recurr_pos=ol_mode_pos_scale_recurr_itr,
                    sample_frequency=self.sample_frequency,
                ) # (A*L, T, 2), (A*L, T), (A*L, T, 2), (A*L, T, 2), (A*L, T, 2, 2)
                ol_pos_mu_local[:, mode, :, itr, :, :] = ol_mode_pos_mu_local_itr.reshape(num_agents, 
                                                                    self.num_layers, self.num_future_steps, self.output_dim)
                ol_pos_scale_local[:, mode, :, itr, :, :, :] = ol_mode_pos_scale_local_itr.reshape(num_agents, 
                                                                    self.num_layers, self.num_future_steps, self.output_dim, self.output_dim)
                
                # simulate the trajectory and update training data
                ol_mode_last_pos_mu_vcs_itr_deatched = \
                    ol_mode_pos_mu_vcs_itr.reshape(num_agents, self.num_layers, self.num_future_steps, self.output_dim)[:, -1, :, :].detach() # (A, T, 2)
                ol_mode_last_head_mu_vcs_itr_deatched = \
                    ol_mode_head_mu_vcs_itr.reshape(num_agents, self.num_layers, self.num_future_steps)[:, -1, :].detach() # (A, T)
                ol_mode_last_vel_mu_vcs_itr_deatched = \
                    ol_mode_vel_mu_vcs_itr.reshape(num_agents, self.num_layers, self.num_future_steps, self.output_dim)[:, -1, :, :].detach() # (A, T, 2)
                simulate_single_trajectory_per_agent_(
                    data=data_sim_modes_list[-1],
                    sim_mask=sim_mask,
                    current_frame=current_frame,
                    num_sim_steps=self.num_eval_sim_steps,
                    trajectories_vcs_pos=ol_mode_last_pos_mu_vcs_itr_deatched,
                    trajectories_vcs_head_rad=ol_mode_last_head_mu_vcs_itr_deatched,
                    trajectories_vcs_vel=ol_mode_last_vel_mu_vcs_itr_deatched,
                )
                # update the current frame
                current_frame += self.num_eval_sim_steps

                if itr < self.num_eval_recurr_steps - 1:
                    pred = self(current_frame, data_sim_modes_list[-1])
                    ol_pos_mu_recurr[:, mode, :, itr + 1, :, :] = pred["traj_pos_mu"][:, mode]
                    ol_pos_scale_recurr[:, mode, :, itr + 1, :, :, :] = pred["traj_pos_scale"][:, mode]

        l2_norm = (torch.norm(ol_pos_mu_recurr[:, :, -1, 0, :, : self.output_dim] - gt[..., : self.output_dim].unsqueeze(1),
                            p=2, dim=-1)* gt_reg_mask.unsqueeze(1)).sum(dim=-1)
        best_mode = l2_norm.argmin(dim=-1)

        ol_best_pos_mu_local = ol_pos_mu_local[torch.arange(num_agents), best_mode]  # (A, L, N, T, 2)
        ol_best_pos_scale_local = ol_pos_scale_local[torch.arange(num_agents), best_mode]  # (A, L, N, T, 2)

        ol_last_pos_mu_local_itr_0 = ol_pos_mu_recurr[:, :, -1, 0, :, :] # (A, M, T, 2)
        ol_last_pos_scale_local_itr_0 = ol_pos_scale_recurr[:, :, -1, 0, :, :, :] # (A, M, T, 2, 2)
        
        cl_last_pos_mu_vcs_detached = torch.zeros(num_agents, num_modes, self.num_future_steps, self.output_dim, device=DEVICE) 
        cl_last_head_mu_vcs_detached = torch.zeros(num_agents, num_modes, self.num_future_steps, device=DEVICE) 
        cl_last_pos_mu_local_detached = torch.zeros(num_agents, num_modes, self.num_future_steps, self.output_dim, device=DEVICE) 
        for mode, data_sim in enumerate(data_sim_modes_list):
            cl_last_pos_mu_vcs_detached[:, mode, :, :] = data_sim['agents_position'][:, self.current_frame + 1 :, :]
            cl_last_head_mu_vcs_detached[:, mode, :] = data_sim['agents_heading_rad'][:, self.current_frame + 1 :]
            cl_last_pos_mu_local_detached[:, mode, :, :] = \
                vcs_to_local_tensor(
                        local_vcs_pos=data_sim['agents_position'][:, self.current_frame, :],
                        local_vcs_head_rad=data_sim['agents_heading_rad'][:, self.current_frame],
                        traj_vcs_pos=cl_last_pos_mu_vcs_detached[:, mode, :, :],
                    )[0]
                
        return (pi, ol_last_pos_mu_local_itr_0, ol_last_pos_scale_local_itr_0, ol_best_pos_mu_local, 
                ol_best_pos_scale_local, cl_last_pos_mu_vcs_detached, cl_last_head_mu_vcs_detached, 
                cl_last_pos_mu_local_detached)

    def compute_loss(self, ol_last_pos_mu_local_itr_0: Tensor, ol_last_pos_scale_local_itr_0: Tensor,
        ol_best_pos_mu_local: Tensor, ol_best_pos_scale_local: Tensor, pi: Tensor, gt_local: Tensor, 
        gt_reg_mask: torch.BoolTensor, cls_mask: torch.BoolTensor, use_cls_loss: bool, cl_loss_scale: float, 
        data_split: str, agent_type: Optional[str] = None,
    ) -> torch.FloatTensor:
        """
        ol_last_pos_mu_local_itr_0: (A, M, T, 2)
        ol_last_pos_scale_local_itr_0: (A, M, T, 2, 2)
        ol_best_pos_mu_local: (A, L, N, T, 2)
        ol_best_pos_scale_local: (A, L, N, T, 2, 2)
        pi: (A, M)
        gt_local: (A, T, 2)
        gt_reg_mask: (A, T)
        cls_mask: (A,)
        """
        agent_split = agent_type + "_" + data_split if agent_type is not None else data_split
        loss = 0
        # compute the classification loss
        if use_cls_loss:
            cls_loss = self.cls_loss(
                pred=ol_last_pos_mu_local_itr_0[:, :, -1 :, :].detach(), # (A, M, 1, 2)
                scale=ol_last_pos_scale_local_itr_0[:, :, -1 :, :].detach(), # (A, M, 1, 2, 2)
                target=gt_local[:, -1 :, : self.output_dim], # (A, 1, 2)
                prob=pi, # (A, M)
                reg_mask=gt_reg_mask[:, -1 :] # (A, 1)
            ) * cls_mask # (A,)
            cls_loss = cls_loss.sum() / cls_mask.sum().clamp_(min=1)
            loss += self.cls_loss_scale * cls_loss
            self.log(f"{agent_split}_cls_loss", cls_loss, sync_dist=True, batch_size=1)

        # compute the reg losses for all the recurrent iterations
        if data_split == "train":
            num_recurr_steps = self.num_train_recurr_steps
            num_sim_steps = self.num_train_sim_steps  
        elif data_split == "val":
            num_recurr_steps = self.num_eval_recurr_steps
            num_sim_steps = self.num_eval_sim_steps
        else:
            raise ValueError("No correct data split is provided")
        current_frame = self.current_frame
        for itr in range(num_recurr_steps):
            # get iterative gt and reg_mask for best mode in local coordinate system
            gt_local_itr = torch.concat([gt_local[:, itr * num_sim_steps :, :],
                    torch.zeros_like(gt_local[:, : itr * num_sim_steps, :])], dim=-2) # (A, T, 2)
            gt_reg_mask_itr = torch.concat([gt_reg_mask[:, itr * num_sim_steps :],
                    torch.zeros_like(gt_reg_mask[:, : itr * num_sim_steps])], dim=-1) # (A, T)

            # get iterative traj transformed to local coordinate system from recurr coordinate system
            ol_best_pos_mu_local_itr = ol_best_pos_mu_local[:, :, itr, :, :]  # (A, L, T, 2)
            ol_best_pos_scale_local_itr = ol_best_pos_scale_local[:, :, itr, :, :, :]  # (A, L, T, 2, 2)

            # compute the regression and heading loss based
            reg_loss_itr = self.reg_loss(ol_best_pos_mu_local_itr, ol_best_pos_scale_local_itr,
                            gt_local_itr.unsqueeze(1).repeat(1, self.num_layers, 1, 1)).sum(dim=1) * gt_reg_mask_itr
            reg_loss_itr = reg_loss_itr.sum(dim=0) / gt_reg_mask_itr.sum(dim=0).clamp_(min=1)
            reg_loss_itr = reg_loss_itr.mean()
            loss += (cl_loss_scale**itr) * self.reg_loss_scale * reg_loss_itr

            head_delta_best_local_itr = angle_between_2d_vectors(
                ctr_vector=ol_best_pos_mu_local_itr[:, :, 1:-1, :] - ol_best_pos_mu_local_itr[:, :, :-2, :],
                nbr_vector=ol_best_pos_mu_local_itr[:, :, 2:, :] - ol_best_pos_mu_local_itr[:, :, 1:-1, :],
            )
            head_delta_mask_local_itr = (gt_reg_mask_itr[:, :-2] & gt_reg_mask_itr[:, 1:-1] & gt_reg_mask_itr[:, 2:])
            head_delta2_loss_itr = abs(head_delta_best_local_itr[:, :, :-1] - head_delta_best_local_itr[:, :, 1:]).sum(dim=1)
            head_delta2_mask_local_itr = head_delta_mask_local_itr[:, :-1] & head_delta_mask_local_itr[:, 1:]
            head_delta2_loss_itr = head_delta2_loss_itr.sum(dim=0) / head_delta2_mask_local_itr.sum(dim=0).clamp_(min=1)
            head_delta2_loss_itr = head_delta2_loss_itr.mean()
            loss += (cl_loss_scale**itr) * self.head_regularization_scale * head_delta2_loss_itr

            if itr == 0:
                self.log(f"{agent_split}_reg_loss_propose", reg_loss_itr, sync_dist=True, batch_size=1)

            # update the current frame
            current_frame += num_sim_steps

        self.log(f"{agent_split}_loss", loss, sync_dist=True, batch_size=1)
        
        return loss

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention,
            nn.LSTM, nn.LSTMCell, nn.GRU, nn.GRUCell,
        )
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = "%s.%s" % (module_name, param_name) if module_name else param_name
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                "params": [param_dict[param_name] for param_name in sorted(list(decay))],
                "weight_decay": self.train_config.weight_decay,
            },
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.train_config.learning_rate, weight_decay=self.train_config.weight_decay
        )
        if self.train_config.use_swa:
            return {
                "optimizer": optimizer,
            }
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=self.train_config.lr_factor_on_plateau,
                patience=self.train_config.patience,
                min_lr=self.train_config.min_lr,
            )
            if self.train_config.use_target_net and not self.train_config.use_scene_net:
                monitor_metric = "val_loss"
            elif not self.train_config.use_target_net and self.train_config.use_scene_net:
                monitor_metric = "scene_val_loss"
            elif self.train_config.use_target_net and self.train_config.use_scene_net:
                monitor_metric = "val_loss"
            else:
                raise ValueError("None of networks are being trained")
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": monitor_metric}}


if __name__ == "__main__":
    from network.data_generator import TrajectoryGridDataModule
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning import Trainer

    train_config = TrainingConfig(batch_size=1)
    data_config = DataStructureConfig()
    net_config = NetConfig()
    data_modules = TrajectoryGridDataModule(data_config=data_config, training_config=train_config)
    model = Net(data_config, net_config, train_config)
    logger = TensorBoardLogger("tb_logs", name="lane_net")
    trainer = Trainer(limit_train_batches=0.5, max_epochs=2, profiler="pytorch", logger=logger, devices=[0])
    trainer.fit(model, data_modules)
