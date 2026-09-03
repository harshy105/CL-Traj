import datetime, json, os, torch, gc
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch import Tensor
from typing import Tuple, Optional, Dict, Any, List
from copy import deepcopy

from network.net import Net
from network.data_generator import TrajectoryGridMapDatasetLMDB
from metrics.planning_metrics import PlanningMetric
from metrics.metrics_io import build_metrics_record, MetricsWriter

# from datasets.nuscenes.data_preprocessing.sample_creation import SampleCreator
from utilities.transformation import (
    vcs_to_gcs_tensor,
    create_patch,
    recurr_to_vcs_to_local_tensor,
    simulate_single_trajectory_per_agent_,
    vcs_to_local_tensor
)
from metrics.pred_metrics import PredMetric
from metrics.offroad_metrics import OffroadMetric
from metrics.comfort_metrics import ComfortMetric
from utilities.utils import append_one_for_single_batch, combine_batch_elements_dim, change_device_of_dict, unbatch_and_pad
from config.config import TARGET_PATH_DS, SAVE_PATH
from config.train_config import TrainingConfig, DataStructureConfig, NetConfig


class DeepScenarioEvaluation:
    def __init__(
        self, data_split: str, save_dir: str, model_name: str, ckpt_name: str, 
        num_eval_recurr_steps: Optional[int] = None, batch_size: Optional[int] = None,
        train_condition: Optional[str] = None, eval_reactive: Optional[bool] = None,
    ) -> None:
        self.save_dir = save_dir
        self.model_name = model_name
        self.ckpt_name = ckpt_name
        self.data_split = data_split
        self.train_condition = train_condition
        self.eval_reactive = eval_reactive
        config_name = [f for f in os.listdir(save_dir + model_name + "/") if (".npz" in f) and (model_name in f)][0]

        train_config, data_config, net_config = self._load_config(save_dir, model_name, config_name)
        self.batch_size = train_config.batch_size if batch_size is None else batch_size
        self.use_target_net = train_config.use_target_net
        self.use_scene_net = train_config.use_scene_net
        self.data_config = data_config
        self.net_config = net_config
        self.current_frame = data_config.current_frame
        self.output_dim = net_config.output_dim
        self.num_layers = net_config.num_layers
        self.num_future_steps = data_config.num_future_steps
        self.num_eval_recurr_steps = train_config.num_eval_recurr_steps if num_eval_recurr_steps is None else num_eval_recurr_steps
        self.num_sim_steps = self.num_future_steps // self.num_eval_recurr_steps
        self.max_num_modes = max(net_config.target_num_modes * train_config.use_target_net, 
                                 net_config.scene_num_modes * train_config.use_scene_net)
        self.sample_frequency = data_config.sample_frequency

        self.data_loader = TrajectoryGridMapDatasetLMDB(
            data_config, TARGET_PATH_DS, data_split, evaluation_mode=True, return_sample=True
        )
        self.sample_of_interest = np.linspace(0, len(self.data_loader)-1, 20).astype(np.int32)
        self.model = self._load_ckpt(save_dir, model_name, ckpt_name, data_config, net_config, train_config)

        # planning eval
        self.time_of_interest = self.current_frame + np.arange(1, 13)
        self.planning_metrics_holder = PlanningMetric(len(self.time_of_interest))
        self.target_pred_metrics_holder = PredMetric()
        self.target_offroad_metrics_holder = OffroadMetric()
        self.target_comfort_metrics_holder = ComfortMetric(dt=1.0 / self.sample_frequency)
        self.scene_pred_metrics_holder = PredMetric()
        self.metrics_writer = MetricsWriter(os.path.join(save_dir, model_name, f"{ckpt_name}_eval_metrics.jsonl"))

    @staticmethod
    def _load_config(
        save_dir: str, model_name: str, config_name: str
    ) -> Tuple[TrainingConfig, DataStructureConfig, NetConfig]:
        return np.load(save_dir + model_name + "/" + config_name, allow_pickle=True)["arr_0"]

    @staticmethod
    def _load_ckpt(
        save_dir: str,
        model_name: str,
        ckpt_name: str,
        data_config: DataStructureConfig,
        net_config: NetConfig,
        train_config: TrainingConfig,
    ) -> Net:
        model = Net.load_from_checkpoint(
            save_dir + model_name + "/" + ckpt_name,
            strict=True,
            data_config=data_config,
            net_config=net_config,
            train_config=train_config,
        )
        model.eval()
        model.freeze()
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(DEVICE)
        return model

    def qual_evaluate(self) -> None:
        data_list = []
        idxs = []
        num_samples = len(self.sample_of_interest)
        for i in range(num_samples):
            idx = self.sample_of_interest[i]
            idxs.append(idx)
            sample = self.data_loader[idx]
            data = self.data_loader.process_sample(sample)
            data = append_one_for_single_batch(data)
            data_list.append(data)
        
            if (i + 1) % self.batch_size == 0 or (i + 1) == num_samples:
                batch_data = {}
                for k in data_list[0].keys():
                    batch_data[k] = torch.cat([d[k] for d in data_list], dim=0)
                
                self._evaluate_batch(batch_data, idxs, visualize_type="vcs")
                data_list = []
                idxs = []

    def quant_evaluate(self) -> None:
        data_list = []
        print(self.model_name)
        print("Eval Reactive:", self.eval_reactive)
        print("T_sim:", self.num_future_steps/(self.num_eval_recurr_steps*self.sample_frequency))
        for i, sample in enumerate(self.data_loader):
            data = self.data_loader.process_sample(sample)
            data = append_one_for_single_batch(data)
            data_list.append(data)
        
            if (i + 1) % self.batch_size == 0 or (i + 1) == len(self.data_loader):
                # print(i)
                batch_data = {}
                for k in data_list[0].keys():
                    batch_data[k] = torch.cat([d[k] for d in data_list], dim=0)
                
                self._evaluate_batch(batch_data)
                data_list = []

        record = build_metrics_record(
            checkpoint=self.ckpt_name,
            model_name=self.model_name,
            data_split=self.data_split,
            tsim=self.num_sim_steps / self.sample_frequency,
            train_condition=self.train_condition,
            eval_reactive=self.eval_reactive,
            planning_metrics_holder=self.planning_metrics_holder if self.use_target_net else None,
            target_pred_metrics_holder=self.target_pred_metrics_holder if self.use_target_net else None,
            target_offroad_metrics_holder=self.target_offroad_metrics_holder if self.use_target_net else None,
            target_comfort_metrics_holder=self.target_comfort_metrics_holder if self.use_target_net else None,
            scene_pred_metrics_holder=self.scene_pred_metrics_holder if self.use_scene_net else None,
        )
        self.metrics_writer.write(record)

    def _evaluate_batch(self, data: Dict, indices: Optional[List[int]] = None, visualize_type: Optional[str] = None) -> None:
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        data = change_device_of_dict(data, DEVICE)
        data = combine_batch_elements_dim(data)
        if self.use_target_net and not self.use_scene_net:
            sim_mask = data["agents_target_mask"].bool() # (A,)
            assert not self.eval_reactive
        elif not self.use_target_net and self.use_scene_net:
            sim_mask = ~(data["agents_target_mask"].bool()) # (A,)
        elif self.use_target_net and self.use_scene_net:
            if self.eval_reactive:
                sim_mask = torch.ones_like(data["agents_target_mask"], dtype=torch.bool) # (A,)
            else:
                sim_mask = data["agents_target_mask"].bool() # (A,)
                
        else:
            raise ValueError("Neither of the Networks are used in training")


        num_agents = data["agents_position"].shape[0]
        ol_pos_mu_recurr = torch.zeros(
            num_agents, self.max_num_modes, self.num_layers, self.num_eval_recurr_steps,
            self.num_future_steps, self.output_dim, device=DEVICE,
        )

        pred = self.model(self.current_frame, data)
        ol_pos_mu_recurr[:, :, :, 0, :, :] = pred["traj_pos_mu"]  # fill ol output for all the modes
        pi = F.softmax(pred["pi"], dim=-1) # (A, M)
        data_sim_modes_list = []
        for mode in range(self.max_num_modes):
            current_frame = self.current_frame
            data_sim_modes_list.append(deepcopy(data))
            for itr in range(self.num_eval_recurr_steps):
                # perfom the coordinate transformation
                ol_mode_pos_mu_recurr_itr = ol_pos_mu_recurr[:, mode, :, itr, :, :]  # (A, L, T, 2)
                ol_mode_pos_mu_recurr_itr = ol_mode_pos_mu_recurr_itr.reshape(num_agents*self.num_layers, self.num_future_steps, self.output_dim) # (A*L, T, 2)
                (ol_mode_pos_mu_vcs_itr, ol_mode_head_mu_vcs_itr, ol_mode_vel_mu_vcs_itr, 
                _, _) = recurr_to_vcs_to_local_tensor(
                    recurr_vcs_pos=data_sim_modes_list[-1]["agents_position"][:, current_frame, :].repeat_interleave(self.num_layers, dim=0),
                    recurr_vcs_head_rad=data_sim_modes_list[-1]["agents_heading_rad"][:, current_frame].repeat_interleave(self.num_layers, dim=0),
                    local_vcs_pos=data_sim_modes_list[-1]["agents_position"][:, self.current_frame, :].repeat_interleave(self.num_layers, dim=0),
                    local_vcs_head_rad=data_sim_modes_list[-1]["agents_heading_rad"][:, self.current_frame].repeat_interleave(self.num_layers, dim=0),
                    traj_recurr_pos=ol_mode_pos_mu_recurr_itr,
                    sample_frequency=self.sample_frequency,
                ) # (A*L, T, 2), (A*L, T), (A*L, T, 2)
                
                # simulate the trajectory and update training data
                ol_mode_last_pos_mu_vcs_itr_deatched = ol_mode_pos_mu_vcs_itr.reshape(num_agents, self.num_layers, self.num_future_steps, self.output_dim)[:, -1, :, :].detach() # (A, T, 2)
                ol_mode_last_head_mu_vcs_itr_deatched = ol_mode_head_mu_vcs_itr.reshape(num_agents, self.num_layers, self.num_future_steps)[:, -1, :].detach() # (A, T)
                ol_mode_last_vel_mu_vcs_itr_deatched = ol_mode_vel_mu_vcs_itr.reshape(num_agents, self.num_layers, self.num_future_steps, self.output_dim)[:, -1, :, :].detach() # (A, T, 2)
                simulate_single_trajectory_per_agent_(
                    data=data_sim_modes_list[-1],
                    sim_mask=sim_mask,
                    current_frame=current_frame,
                    num_sim_steps=self.num_sim_steps,
                    trajectories_vcs_pos=ol_mode_last_pos_mu_vcs_itr_deatched,
                    trajectories_vcs_head_rad=ol_mode_last_head_mu_vcs_itr_deatched,
                    trajectories_vcs_vel=ol_mode_last_vel_mu_vcs_itr_deatched,
                )
                # update the current frame
                current_frame += self.num_sim_steps

                if itr < self.num_eval_recurr_steps - 1:
                    pred = self.model(current_frame, data_sim_modes_list[-1])
                    ol_pos_mu_recurr[:, mode, :, itr + 1, :, :] = pred["traj_pos_mu"][:, mode]

        # get the closed loop trajectories
        cl_last_pos_mu_vcs_detached = torch.zeros(num_agents, self.max_num_modes, self.num_future_steps, self.output_dim, device=DEVICE) 
        cl_last_head_mu_vcs_detached = torch.zeros(num_agents, self.max_num_modes, self.num_future_steps, device=DEVICE) 
        for mode, data_sim in enumerate(data_sim_modes_list):
            cl_last_pos_mu_vcs_detached[:, mode, :, :] = data_sim['agents_position'][:, self.current_frame+1:, :]
            cl_last_head_mu_vcs_detached[:, mode, :] = data_sim['agents_heading_rad'][:, self.current_frame+1:]

        if visualize_type is None:
            gt = data["agents_gt_position"]  # (A, T, 2)
            gt_reg_mask = data["agents_log_reg_mask"][:, self.current_frame + 1 :].bool()  # (A, T)
            if self.use_target_net:
                target_agent_mask = data["agents_target_mask"].bool()
                gt_target = gt[target_agent_mask] # (A_target, T, 2)
                gt_target_reg_mask = gt_reg_mask[target_agent_mask] # (A_target, T)
                pi_target = pi[target_agent_mask] # (A_target,)
                
                cl_last_pos_mu_local_detached = vcs_to_local_tensor(
                    local_vcs_pos=data_sim['agents_position'][:, self.current_frame, :],
                    local_vcs_head_rad=data_sim['agents_heading_rad'][:, self.current_frame],
                    traj_vcs_pos=cl_last_pos_mu_vcs_detached.reshape(num_agents, self.max_num_modes*self.num_future_steps, self.output_dim)
                )[0].reshape(num_agents, self.max_num_modes, self.num_future_steps, self.output_dim)
                
                cl_traj_target = cl_last_pos_mu_local_detached[target_agent_mask]  # (A_target, M, T, 2)
                self.target_pred_metrics_holder.update(cl_traj_target, gt_target, gt_target_reg_mask, pi_target)
                self.target_offroad_metrics_holder.update(cl_traj_target, gt_target_reg_mask, pi_target,
                                                    data['lanes_start_locs'], data['lanes_end_locs'],
                                                    data['agents_batch'][target_agent_mask], 
                                                    data['lanes_batch'])
                self.target_comfort_metrics_holder.update(cl_traj_target, gt_target_reg_mask, pi_target)
                
                # compute plan metrics in a vcs_head0 coordinate system
                # get the GT information
                unbatched_data = unbatch_and_pad(data)
                B = unbatched_data["agents_position"].shape[0]
                assert pi_target.shape[0] == B
                others_gt_trajs = torch.concat(
                        [unbatched_data["agents_position"][:, 1:, self.time_of_interest, :2],
                        unbatched_data["agents_heading_rad"][:, 1:, self.time_of_interest, None]],
                        dim=-1)

                # get the ego closed-loop trajectory with all the modes
                cl_last_pos_head_mu_vcs_detached_target = torch.concat(
                    [cl_last_pos_mu_vcs_detached, cl_last_head_mu_vcs_detached.unsqueeze(-1)], 
                    dim=-1)[target_agent_mask]
                assert cl_last_pos_head_mu_vcs_detached_target.shape[0] == B
                
                # Get the surrounding-agent closed-loop trajectories, one per ego mode.
                # The surrounding-agent network head is still genuinely single-mode --
                # scene_num_modes == 1, asserted below -- the divergence across m comes purely
                # from reacting to a different ego trajectory per mode, not from the
                # surrounding-agent model itself being multi-modal.
                if self.eval_reactive:
                    assert sim_mask.all() 
                    assert self.net_config.scene_num_modes == 1
                    others_trajs_per_mode = []
                    for mode_data in data_sim_modes_list:
                        unbatched_mode = unbatch_and_pad(mode_data)
                        B_ = unbatched_mode["agents_position"].shape[0]
                        assert B == B_
                        others_trajs_per_mode.append(
                            torch.concat(
                                [unbatched_mode["agents_position"][:, 1:, self.time_of_interest, :2],
                                unbatched_mode["agents_heading_rad"][:, 1:, self.time_of_interest, None]],
                                dim=-1
                            )
                        )  # each: (B, n_agents, T, 3)
                    others_trajs = torch.stack(others_trajs_per_mode, dim=2)  # (B, n_agents, M, T, 3)
                else:
                    others_trajs = others_gt_trajs.unsqueeze(2).expand(-1, -1, self.max_num_modes, -1, -1)
                
                self.planning_metrics_holder.update(
                    ego_trajs=cl_last_pos_head_mu_vcs_detached_target[:, :, self.time_of_interest - self.current_frame - 1, :],
                    ego_mode_probs=pi_target,
                    ego_gt_trajs=torch.concat(
                        [unbatched_data["agents_position"][:, 0, self.time_of_interest, :2],
                        unbatched_data["agents_heading_rad"][:, 0, self.time_of_interest, None]],
                        dim=-1),
                    ego_wh_trajs=unbatched_data["agents_size"][:, 0, self.time_of_interest, :2],
                    ego_gt_trajs_mask=unbatched_data["agents_log_reg_mask"][:, 0, self.time_of_interest],
                    others_trajs=others_trajs,
                    others_gt_trajs=others_gt_trajs,
                    others_wh_trajs=unbatched_data["agents_size"][:, 1:, self.time_of_interest, :2],
                    others_gt_trajs_mask=unbatched_data["agents_log_reg_mask"][:, 1:, self.time_of_interest],
                )
                
            if self.use_scene_net:
                surr_agents_mask = ~(data["agents_target_mask"].bool())
                if surr_agents_mask.any():
                    gt_surr = gt[surr_agents_mask] # (A_surr, T, 2)
                    gt_surr_reg_mask = gt_reg_mask[surr_agents_mask] # (A_surr, T)
                    pi_surr = pi[surr_agents_mask] # (A_surr,)
                    
                    cl_last_pos_mu_local_detached = vcs_to_local_tensor(
                        local_vcs_pos=data_sim['agents_position'][:, self.current_frame, :],
                        local_vcs_head_rad=data_sim['agents_heading_rad'][:, self.current_frame],
                        traj_vcs_pos=cl_last_pos_mu_vcs_detached.reshape(num_agents, self.max_num_modes*self.num_future_steps, self.output_dim)
                    )[0].reshape(num_agents, self.max_num_modes, self.num_future_steps, self.output_dim)
                    
                    cl_traj_surr = cl_last_pos_mu_local_detached[surr_agents_mask]  # (A_surr, M, T, 2)
                    self.scene_pred_metrics_holder.update(cl_traj_surr, gt_surr, gt_surr_reg_mask, pi_surr)
            
        elif visualize_type == "vcs":
            unbatched_data = unbatch_and_pad(data)
            
            batch_indices = data["agents_batch"]
            num_elements = data["agents_position"].shape[0]
            num_batches = batch_indices.max().item() + 1
            elements_per_batch = torch.bincount(batch_indices)
            max_num_elements = elements_per_batch.max().item()

            arange_total = torch.arange(num_elements, device=DEVICE)
            batch_starts = torch.cat(
                (torch.tensor([0], dtype=torch.long, device=DEVICE), elements_per_batch.cumsum(0)[:-1])
            )
            element_indices = arange_total - torch.gather(batch_starts, 0, batch_indices.long())
            
            pi_unbatched = torch.full((num_batches, max_num_elements, self.max_num_modes), 0.0, device=pi.device)
            pi_unbatched[batch_indices, element_indices] = pi

            traj_vcs_pos_sim_unbatched = torch.full((num_batches, max_num_elements, self.max_num_modes, self.num_future_steps, self.output_dim), 0.0, device=cl_last_pos_mu_vcs_detached.device)
            traj_vcs_pos_sim_unbatched[batch_indices, element_indices] = cl_last_pos_mu_vcs_detached

            if self.use_target_net and not self.use_scene_net:
                sim_mask_unbatched = unbatched_data["agents_target_mask"].bool()
                assert not self.eval_reactive
            elif not self.use_target_net and self.use_scene_net:
                sim_mask_unbatched = ~(unbatched_data["agents_target_mask"].bool())
            elif self.use_target_net and self.use_scene_net:
                if self.eval_reactive:
                    sim_mask_unbatched = torch.ones_like(unbatched_data["agents_target_mask"].bool())
                else:
                    sim_mask_unbatched = unbatched_data["agents_target_mask"].bool()
            else:
                raise ValueError("Neither of the Networks are used in training")

            unbatched_data = change_device_of_dict(unbatched_data, torch.device("cpu"))
            pi_unbatched = pi_unbatched.cpu()
            traj_vcs_pos_sim_unbatched = traj_vcs_pos_sim_unbatched.cpu()
            sim_mask_unbatched = sim_mask_unbatched.cpu()
            
            for sample_idx in range(unbatched_data['agents_position'].shape[0]):
                data_sample = {}
                for k, v in unbatched_data.items():
                    data_sample[k] = v[sample_idx]
                
                sim_mask_sample = sim_mask_unbatched[sample_idx]
                prediction_sample = traj_vcs_pos_sim_unbatched[sample_idx]
                pi_sample = pi_unbatched[sample_idx]

                traj_vcs_pos_sample = prediction_sample[sim_mask_sample]  # (A_sim, M, T, 2)
                pi_sample = pi_sample[sim_mask_sample]  # (A_sim, M)

                if self.eval_reactive:
                    assert sim_mask_unbatched.all()
                    # Under reactive eval, mode m's surrounding-agent trajectories are a real
                    # reaction to ego's mode-m plan specifically , so plot each mode separately.
                    for m in range(self.max_num_modes):
                        self.visualization_vcs(
                            token=f"{np.random.rand()}_mode{m}",
                            data_vcs=data_sample,
                            traj_vcs=traj_vcs_pos_sample[:, m : m + 1, :, :],
                            pi=pi_sample[:, m : m + 1],
                        )
                else:
                    # Log-replay: surrounding agents' single trajectory is the same background
                    # regardless of which ego mode is shown, so all ego modes can still be
                    # meaningfully overlaid in one combined plot, as before.
                    self.visualization_vcs(
                        token=str(np.random.rand()), data_vcs=data_sample, traj_vcs=traj_vcs_pos_sample, pi=pi_sample
                    )

    def visualization_vcs(
        self,
        token: str,
        data_vcs: Dict,
        traj_vcs: Tensor,
        pi: Tensor,
    ):
        target_pos = data_vcs["agents_position"][0, self.current_frame]
        target_heading_rad = data_vcs["agents_heading_rad"][0, self.current_frame]

        dynamic_patch_polygon, _ = create_patch(
            self.data_config.dynamic_region_size,
            angle_rad=target_heading_rad,
            agent_pos=target_pos,
            offset=self.data_config.dynamic_translation,
        )
        static_patch_polygon, _ = create_patch(
            self.data_config.static_region_size,
            angle_rad=target_heading_rad,
            agent_pos=target_pos,
            offset=self.data_config.static_translation,
        )

        # visualize training sample
        self.data_loader.visualize_training_sample(data_vcs, dynamic_patch_polygon, static_patch_polygon, vis_bb=False)

        # visualize multimodal predictions
        num_modes_to_plot = traj_vcs.shape[1]
        for agent in range(traj_vcs.shape[0]):
            for m in range(num_modes_to_plot):
                prediction = traj_vcs[agent, m]
                probability = pi[agent, m].item()
                ckpt_name = self.ckpt_name
                if self.use_target_net and agent == 0:
                    plt.plot(prediction[:, 0], prediction[:, 1], zorder=7,
                        linewidth=1.4, c="green", label="Target Prediction", alpha=1.0,
                    )
                    plt.scatter(prediction[-1, 0], prediction[-1, 1], zorder=7, c="green", alpha=1.0, marker="o", s=10)
                    plt.text(prediction[-1, 0], prediction[-1, 1], str(np.round(probability, 3)), 
                            size=9, c="black", zorder=7, label="Mode Probability",
                    )
                else:
                    plt.plot(prediction[:, 0], prediction[:, 1], zorder=7,
                        linewidth=1.4, c="royalblue", label="Scene Prediction", alpha=1.0,
                    )
                    plt.scatter(prediction[-1, 0], prediction[-1, 1], zorder=7, c="royalblue", alpha=1.0, marker="o", s=10)

        # removing multiple legends
        handles, labels = plt.gca().get_legend_handles_labels()
        alpha_dict = {}
        for handle, label in zip(handles, labels):
            assert handle._label == label
            alpha = handle.get_alpha()
            alpha = 0.0 if alpha is None else alpha
            alpha_dict[handle] = alpha
        sorted_handles = sorted(
            alpha_dict.keys(), key=lambda x: alpha_dict[x], reverse=False
        )  # handels with sorted alphas
        by_label = {handle._label: handle for handle in sorted_handles}  # unique label with highest alphas
        plt.legend(by_label.values(), by_label.keys(), fontsize="10", loc="upper left")
        # save the plot
        img_folder = os.path.join(self.save_dir, self.model_name, f"qual_eval_{ckpt_name}_{self.num_sim_steps/self.sample_frequency}secs")
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)
        plt.savefig(os.path.join(img_folder, f"{token}.png"), dpi=200)
        # plt.show()
        plt.close("all")
        gc.collect()


if __name__ == "__main__":
    train_condition = "reactive" # "non-reactive", reactive refers to when scene network is used
    eval_reactive = True # True is the scene network is used
    ckpt_names = [
        "******.ckpt"
    ]

    save_dir = SAVE_PATH
    batch_size = TrainingConfig().batch_size
    data_split = "val"
    for num_eval_recurr_steps in [1, 2, 3, 4, 6, 12]:
        for ckpt_name in ckpt_names:
            model_name = ckpt_name.split("-epoch")[0]
            track_eval = DeepScenarioEvaluation(
                data_split, save_dir, model_name, ckpt_name, num_eval_recurr_steps=num_eval_recurr_steps,
                batch_size=batch_size, train_condition=train_condition, eval_reactive=eval_reactive,
            )
            track_eval.quant_evaluate()
            track_eval.qual_evaluate()
            del track_eval