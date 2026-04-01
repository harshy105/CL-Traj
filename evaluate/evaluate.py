import datetime, json, os, torch, gc
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch import Tensor
from typing import Tuple, Optional, Dict, Any
from copy import deepcopy
from datasets.nuscenes.nuscenes_devkit.eval.prediction.compute_metrics import compute_metrics
from datasets.nuscenes.nuscenes_devkit.eval.prediction.config import load_prediction_config
from datasets.nuscenes.nuscenes_devkit.prediction import PredictHelper
from datasets.nuscenes.nuscenes_devkit import NuScenes

from network.net import Net
from network.data_generator import TrajectoryGridMapDatasetLMDB
from metrics.planning_metrics import PlanningMetric

# from datasets.nuscenes.data_preprocessing.sample_creation import SampleCreator
from utilities.transformation import (
    vcs_to_gcs_tensor,
    create_patch,
    recurr_to_vcs_to_local_tensor,
    simulate_single_trajectory_per_agent_,
    vcs_to_local_tensor
)
from metrics.pred_metrics import PredMetric
from utilities.utils import append_one_for_single_batch, combine_batch_elements_dim, change_device_of_dict
from datasets.nuscenes.nuscenes_devkit.eval.prediction.splits import get_prediction_challenge_split
from config.config import NUSCENES_PATH, TARGET_PATH, SAVE_PATH
from config.train_config import TrainingConfig, DataStructureConfig, NetConfig
from config.nuscenes_config import NuScenesPreprocessConfig
from config.train_config import SampleOfInterest


class NuScenesEvaluation:
    def __init__(
        self, data_split: str, save_dir: str, model_name: str, ckpt_name: str, num_eval_recurr_steps: Optional[int] = None
    ) -> None:
        self.save_dir = save_dir
        self.model_name = model_name
        self.pred_results_gpu = []
        self.pred_results = []
        self.target_metric_results = []
        self.ckpt_name = ckpt_name
        self.data_split = data_split
        config_name = [f for f in os.listdir(save_dir + model_name + "/") if (".npz" in f) and (model_name in f)][0]

        train_config, data_config, net_config = self._load_config(save_dir, model_name, config_name)
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

        data_split_map = {"val": "v1.0-trainval", "train_val": "v1.0-trainval", "mini_val": "v1.0-mini"}
        data_set = data_split_map[data_split]
        self.sample_of_interest = SampleOfInterest().sample_list

        nuscenes = NuScenes(version=data_set, dataroot=NUSCENES_PATH, verbose=True)
        # self.create_sample_from_token = SampleCreator(NuScenesPreprocessConfig(), nuscenes, data_split, data_set).create_sample_from_token
        self.data_loader = TrajectoryGridMapDatasetLMDB(
            data_config, TARGET_PATH, data_split, evaluation_mode=True, return_sample=True
        )
        self.model = self._load_ckpt(save_dir, model_name, ckpt_name, data_config, net_config, train_config)

        # evaluation parameter
        self.helper = PredictHelper(nuscenes)
        self.nuscenes_config = load_prediction_config(self.helper, "predict_2020_icra.json")
        self.test_set = get_prediction_challenge_split(data_split, dataroot=NUSCENES_PATH)

        # planning eval
        self.time_of_interest = self.current_frame + np.arange(1, 13)
        self.planning_metrics_holder = PlanningMetric(len(self.time_of_interest))
        self.planning_metrics_results_list = []
        self.pred_metrics_holder = PredMetric()
        self.scene_pred_metrics_results_list = []

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
        for idx in self.sample_of_interest:
            token = self.test_set[idx]
            self._evaluate_sample(idx, token, visualize_type="vcs")

    def quant_evaluate(self) -> None:
        for idx, token in enumerate(self.test_set):
            self._evaluate_sample(idx, token)
            # print(idx)

        if self.use_target_net:
            min_plan_metrics_1 = self.planning_metrics_holder.compute(n=1)
            min_plan_metrics_5 = self.planning_metrics_holder.compute(n=5)
            planning_metrics = {
                "min_col_1": min_plan_metrics_1["box_col_percent"].cpu().numpy().tolist(),
                "min_l2_1": min_plan_metrics_1["L2"].cpu().numpy().tolist(),
                "min_col_5": min_plan_metrics_5["box_col_percent"].cpu().numpy().tolist(),
                "min_l2_5": min_plan_metrics_5["L2"].cpu().numpy().tolist(),
            }
            self.planning_metrics_results_list.append(planning_metrics)
            print(planning_metrics)
            self._pred_gpu_to_cpu()
            target_pred_metrics = compute_metrics(self.pred_results, self.helper, self.nuscenes_config)
            self.target_metric_results.append(target_pred_metrics)
            print(target_pred_metrics)
        if self.use_scene_net:
            min_ade_1, min_fde_1 = self.pred_metrics_holder.compute(n=1)
            scene_pred_metrics = {
                "min_ade_1": min_ade_1.item(),
                "min_fde_1": min_fde_1.item()
            }
            self.scene_pred_metrics_results_list.append(scene_pred_metrics)
            print(scene_pred_metrics)            
        
        self._save_predictions()

    def _evaluate_sample(self, index: int, token: str, visualize_type: Optional[str] = None) -> None:
        sample = self.data_loader[index]
        data = self.data_loader.process_sample(sample)
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        data = change_device_of_dict(data, DEVICE)
        data = append_one_for_single_batch(data)
        data = combine_batch_elements_dim(data)
        if self.use_target_net and not self.use_scene_net:
            sim_mask = data["agents_target_mask"].bool() # (A,)
        elif not self.use_target_net and self.use_scene_net:
            sim_mask = ~(data["agents_target_mask"].bool()) # (A,)
        elif self.use_target_net and self.use_scene_net:
           sim_mask = torch.ones_like(data["agents_target_mask"], dtype=torch.bool) # (A,)
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
        for mode, data_sim in enumerate(data_sim_modes_list):
            cl_last_pos_mu_vcs_detached[:, mode, :, :] = data_sim['agents_position'][:, self.current_frame+1:, :]
        # sample out the masked agent

        if visualize_type is None:
            if self.use_target_net:
                target_agent_mask = data["agents_target_mask"].bool()
                traj_vcs_pos_target = cl_last_pos_mu_vcs_detached[target_agent_mask]
                pi_target = pi[target_agent_mask]
                # compute pred metrics in global coordinate system
                target_gcs_pos = torch.tensor(sample.target_trajectory[0, sum(sample.input_mask) - 1, :2], 
                                            device=DEVICE, dtype=traj_vcs_pos_target.dtype)  # (2,)
                target_gcs_head_deg = torch.tensor(sample.target_trajectory[0, sum(sample.input_mask) - 1, 4], 
                                            device=DEVICE, dtype=traj_vcs_pos_target.dtype)  # Float
                traj_gcs_pos_target, _ = vcs_to_gcs_tensor(target_gcs_pos, target_gcs_head_deg, traj_vcs_pos_target)
                self.pred_results_gpu.append({"token": token, "prediction": traj_gcs_pos_target, "probabilities": pi_target})
                # compute plan metrics in a vcs_head0 coordinate system
                combined_sim_trajs = torch.zeros((1, self.max_num_modes, len(self.time_of_interest), 3), device=DEVICE)
                for mode, data_sim_mode in enumerate(data_sim_modes_list):
                    combined_sim_trajs[:, mode] = torch.concat(
                        [data_sim_mode["agents_position"][:1, self.time_of_interest, :2],
                        data_sim_mode["agents_heading_rad"][:1, self.time_of_interest, None]],
                        dim=-1)
                self.planning_metrics_holder.update(
                    ego_trajs=combined_sim_trajs,
                    ego_mode_probs=pi_target,
                    ego_gt_trajs=torch.concat(
                        [data["agents_position"][:1, self.time_of_interest, :2],
                        data["agents_heading_rad"][:1, self.time_of_interest, None]],
                        dim=-1),
                    ego_wh_trajs=data["agents_size"][:1, self.time_of_interest, :2],
                    ego_gt_trajs_mask=data["agents_log_reg_mask"][:1, self.time_of_interest],
                    others_gt_trajs=torch.concat(
                        [data["agents_position"][None, 1:, self.time_of_interest, :2],
                        data["agents_heading_rad"][None, 1:, self.time_of_interest, None]],
                        dim=-1),
                    others_wh_trajs=data["agents_size"][None, 1:, self.time_of_interest, :2],
                    others_gt_trajs_mask=data["agents_log_reg_mask"][None, 1:, self.time_of_interest],
                )
                
            if self.use_scene_net:
                surr_agents_mask = ~(data["agents_target_mask"].bool())
                if surr_agents_mask.any():
                    gt = data["agents_gt_position"]  # (A, T, 2)
                    gt_reg_mask = data["agents_log_reg_mask"][:, self.current_frame + 1 :].bool()  # (A, T)
                    gt_surr = gt[surr_agents_mask] # (A_surr, T, 2)
                    gt_surr_reg_mask = gt_reg_mask[surr_agents_mask] # (A_surr, T)
                    pi_surr = pi[surr_agents_mask] # (A_surr,)
                    
                    cl_last_pos_mu_local_detached = vcs_to_local_tensor(
                        local_vcs_pos=data_sim['agents_position'][:, self.current_frame, :],
                        local_vcs_head_rad=data_sim['agents_heading_rad'][:, self.current_frame],
                        traj_vcs_pos=cl_last_pos_mu_vcs_detached.reshape(num_agents, self.max_num_modes*self.num_future_steps, self.output_dim)
                    )[0].reshape(num_agents, self.max_num_modes, self.num_future_steps, self.output_dim)
                    
                    cl_traj_surr = cl_last_pos_mu_local_detached[surr_agents_mask]  # (A_surr, M, T, 2)
                    self.pred_metrics_holder.update(cl_traj_surr, gt_surr, gt_surr_reg_mask, pi_surr)
            
        elif visualize_type == "vcs":
            data = change_device_of_dict(data, torch.device("cpu"))
            traj_vcs_pos_sim = cl_last_pos_mu_vcs_detached[sim_mask]
            pi_sim = pi[sim_mask]
            self.visualization_vcs(
                token=token, data_vcs=data, traj_vcs=traj_vcs_pos_sim.cpu(), pi=pi_sim.cpu()
            )

    def _save_predictions(self):
        name = (
            self.ckpt_name + "_" + str(self.data_split) + "_" + str(self.num_sim_steps / self.sample_frequency) + "secs"
        )
        path = self.save_dir + self.model_name + "/"
        if self.use_target_net:
            json.dump(self.target_metric_results, open(os.path.join(path, name + "_pred_metrics.json"), "w"))
            json.dump(self.pred_results, open(os.path.join(path, name + "_pred_trajectories.json"), "w"))
            json.dump(self.planning_metrics_results_list, open(os.path.join(path, name + "_plan_metrics.json"), "w"))
        if self.use_scene_net:
            json.dump(self.scene_pred_metrics_results_list, open(os.path.join(path, name + "_scene_pred_metrics.json"), "w"))

    def _pred_gpu_to_cpu(self) -> None:
        while len(self.pred_results_gpu) > 0:
            pred = self.pred_results_gpu.pop(0)
            trajectories = pred["prediction"].cpu().numpy().astype("float64")[0]
            probabilities = pred["probabilities"].cpu().numpy().astype("float64")[0]
            token = pred["token"]

            instance_token, sample_token = token.split("_")
            # add results to the result dictionary
            new_result = {"instance": instance_token, "sample": sample_token, "prediction": [], "probabilities": []}

            for k, track in enumerate(trajectories):
                new_trajectory = []

                for trajectory_point in track:
                    trajectory_point = trajectory_point.tolist()
                    new_trajectory.append(trajectory_point[:2])

                likelihood = probabilities[k]
                new_result["prediction"].append(new_trajectory)
                new_result["probabilities"].append(likelihood)

            self.pred_results.append(new_result)

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
        for agent in range(traj_vcs.shape[0]):
            for m in range(self.max_num_modes):
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
    ckpt_names = [
        "******.ckpt"
    ]

    save_dir = SAVE_PATH
    data_split = "val"
    for num_eval_recurr_steps in [12, 6, 2, 4]:
        for ckpt_name in ckpt_names:
            model_name = ckpt_name.split("-epoch")[0]
            track_eval = NuScenesEvaluation(
                data_split, save_dir, model_name, ckpt_name, num_eval_recurr_steps=num_eval_recurr_steps
            )
            track_eval.quant_evaluate()
            track_eval.qual_evaluate()
            del track_eval
