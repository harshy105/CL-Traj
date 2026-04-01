import lmdb, os, zstandard, pickle, torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, Tuple, Dict
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from pytorch_lightning import LightningDataModule
from torch_geometric.utils import degree, dense_to_sparse, to_dense_adj
from dataclasses import asdict
from shapely.geometry import Polygon
from datetime import datetime
from matplotlib import patches

from utilities.utils import bipartite_dense_to_sparse
from config.train_config import DataStructureConfig, TrainingConfig, TrainingSample
from config.nuscenes_config import CityNodesFeatures, CityCenterlinesGraph, NuScenesPreprocessConfig
from utilities.transformation import (
    wrap_angle_deg,
    wrap_angle_rad,
    gcs_to_vcs_tensor,
    create_patch,
    sample_subgraph_in_patch,
    vcs_to_local_tensor,
    local_to_vcs_tensor,
    get_bb_corners,
)
from utilities.utils import append_one_for_single_batch, combine_batch_elements_dim


class TrajectoryGridMapDataset(Dataset):
    def __init__(
        self,
        data_config: DataStructureConfig,
        evaluation_mode: bool = False,
    ) -> None:

        self.data_config = data_config
        self.evaluation_mode = evaluation_mode

        self.agent_type_ids = data_config.agent_type_ids
        self.current_frame = data_config.current_frame
        self.num_future_steps = data_config.num_future_steps
        self.static_region_size = data_config.static_region_size
        self.static_translation = data_config.static_translation
        self.dynamic_region_size = data_config.dynamic_region_size
        self.dynamic_translation = data_config.dynamic_translation
        self.target_goal = data_config.target_goal
        self.surr_goal = data_config.surr_goal
        self.x_flip_prob = data_config.x_flip_prob
        self.surr_agent_to_sim_prob = data_config.surr_agent_to_sim_prob
        self.max_goal_offset = data_config.max_goal_offset
        self.max_num_agents = data_config.max_num_agents
        self.max_num_lane_nodes = data_config.max_num_lane_nodes
        self.lane_resolution = NuScenesPreprocessConfig().resolution

    def process_sample(
        self,
        sample: TrainingSample,
    ) -> Dict:
        # get the trajectories in Global Coordinate System
        mask_all_road_users, all_road_users = self.concatenate_road_users(
            sample
        )  # (A, T=15, F=10); F = (# pos_x, pos_y, v_x, v_y, yaw_deg, acc_x, acc_y, box_width, box_length, agent_type)

        
        # Transform trajectories and map to target vehicle at T=0 Coordinate System
        target_pos = torch.tensor(sample.target_trajectory[0, sum(sample.input_mask) - 1, :2])
        target_heading_deg = torch.tensor(sample.target_trajectory[0, sum(sample.input_mask) - 1, 4])
        self.coordinate_transform_to_target_(
                target_pos, target_heading_deg, all_road_users, sample.vectorized_map.nodes_features
            )
        
        # Crop the input
        (dynamic_patch_polygon, static_patch_polygon, patch_road_users, mask_patch_road_users, patch_vectorized_map) = (
            self.crop_input_target(all_road_users, sample.vectorized_map)
        )
        input_mask_patch_road_users = mask_patch_road_users.clone()
        input_mask_patch_road_users[:, self.current_frame+1:] = False
        
        # get the mask
        target_agent_mask = torch.zeros(mask_patch_road_users.shape[0]).to(torch.bool)
        target_agent_mask[0] = True
        surr_agents_sim_mask = mask_patch_road_users[:, self.current_frame].clone()
        surr_agents_sim_mask[0] = False
        mask_goal_patch_road_users = self.get_goal_mask(mask_patch_road_users)
        
        # data augumentation
        if not self.evaluation_mode:
            surr_agent_to_sim = torch.rand_like(surr_agents_sim_mask[1:], dtype=torch.float) < self.surr_agent_to_sim_prob
            surr_agents_sim_mask[1:] = surr_agents_sim_mask[1:] * surr_agent_to_sim            
            mask_goal_patch_road_users = self.offset_goal_mask(mask_goal_patch_road_users, self.max_goal_offset)
            mask_goal_patch_road_users[1:] = mask_goal_patch_road_users[1:] * surr_agents_sim_mask[1:, None]
            if self.x_flip_prob > np.random.random():
                patch_road_users, patch_vectorized_map = self.flip_input_along_x_axis(
                    patch_road_users, patch_vectorized_map
                )

        # Get GT
        gt_local_pos, gt_local_head_rad = self.get_gt(patch_road_users)
        
        # Create the input dict
        agents_dict = {
            "agents_type": patch_road_users[:, 0, 9],  # TODO use one-hot encoding
            "agents_position": patch_road_users[..., :2],  # input and gt positions of agents target vehicle coordinates
            "agents_vel": patch_road_users[..., 2:4],
            "agents_heading_rad": patch_road_users[..., 4],
            # 'agents_acceleration': patch_road_users[..., 5:7],  # TODO coordinate transform and augumentation
            "agents_size": patch_road_users[..., 7:9],  # box_width, box_length
            "agents_log_reg_mask": mask_patch_road_users,  # reg mask for log replay
            "agents_input_reg_mask": input_mask_patch_road_users,  # reg mask for input
            "agents_goal_mask": mask_goal_patch_road_users,
            "agents_target_mask": target_agent_mask,
            "agents_surr_sim_mask": surr_agents_sim_mask,
            "agents_gt_position": gt_local_pos,
            "agents_gt_head_rad": gt_local_head_rad,
        }

        lanes_nodes_dict = {
            "lanes_nodes_id_global": patch_vectorized_map.nodes_features.nodes_id,
            "lanes_start_locs": patch_vectorized_map.nodes_features.start_locs,
            "lanes_end_locs": patch_vectorized_map.nodes_features.end_locs,
        }

        # pad the input to make the same dimension across all samples
        agents_dict, lanes_nodes_dict = self.pad_input(agents_dict, lanes_nodes_dict)

        data = {}
        data.update(agents_dict)
        data.update(lanes_nodes_dict)

        # # visualization
        # data_vis = append_one_for_single_batch(data.copy())
        # data_vis = combine_batch_elements_dim(data_vis)
        # self.visualize_training_sample(data_vis, dynamic_patch_polygon, static_patch_polygon)
        # plt.show()
        # plt.close('all')
        return data

    @staticmethod
    def coordinate_transform_to_target_(
        target_pos: Tensor,
        target_heading_deg: torch.FloatTensor,
        all_road_users: Tensor,
        map_nodes_features: CityNodesFeatures,
    ) -> None:
        """
        Trasnform dynamic and static contexts from Global Coodinate System (GCS) to Target Agent Coordinate System (VCS)
        """
        # transform vel
        all_road_users[..., 2:4], _ = gcs_to_vcs_tensor(
            torch.zeros_like(target_pos), target_heading_deg, all_road_users[..., 2:4], all_road_users[..., 4]
        )
        # transform the postions and head (deg->rad)
        all_road_users[..., :2], all_road_users[..., 4] = gcs_to_vcs_tensor(
            target_pos, target_heading_deg, all_road_users[..., :2], all_road_users[..., 4]
        )
        nodes_start_locs, _ = gcs_to_vcs_tensor(
            target_pos, target_heading_deg, map_nodes_features.start_locs.unsqueeze(1)
        )
        nodes_end_locs, _ = gcs_to_vcs_tensor(target_pos, target_heading_deg, map_nodes_features.end_locs.unsqueeze(1))
        map_nodes_features.start_locs = nodes_start_locs.squeeze(1)
        map_nodes_features.end_locs = nodes_end_locs.squeeze(1)

    def crop_input_target(
        self, all_road_users: Tensor, vectorized_map: CityCenterlinesGraph
    ) -> Tuple[Polygon, Polygon, Tensor, Tensor, CityCenterlinesGraph]:
        target_pos = all_road_users[0, self.current_frame, :2]
        target_heading_rad = all_road_users[0, self.current_frame, 4]
        # crop dynamic context
        dynamic_patch_polygon, dynamic_patch_container = create_patch(
            self.dynamic_region_size, angle_rad=target_heading_rad,
            agent_pos=target_pos, offset=self.dynamic_translation,
        )
        A, T, F = all_road_users.shape
        patch_road_users = all_road_users.reshape(-1, F)
        patch_road_user_traj = patch_road_users[:, :2]
        patch_traj = dynamic_patch_container(patch_road_user_traj)
        patch_road_users[~patch_traj, : F - 1] = (
            torch.nan
        )  # set all the features to nan for any agent at any time outside the region
        patch_road_users = patch_road_users.reshape(A, T, F)
        non_patch_road_users = torch.isnan(
            patch_road_users[:, self.current_frame, 0]
        )  # eliminate all the agents with T=0 being outside the region
        patch_road_users = patch_road_users[~non_patch_road_users]
        mask_patch_road_users = ~torch.isnan(patch_road_users[:, :, 0])  # compute the mask for dynamic agents
        # set all nan values to zeros
        patch_road_users = torch.nan_to_num(patch_road_users)

        # crop static context
        static_patch_polygon, static_patch_container = create_patch(
            self.static_region_size, angle_rad=target_heading_rad, 
            agent_pos=target_pos, offset=self.static_translation
        )
        patch_vectorized_map = sample_subgraph_in_patch(static_patch_container, vectorized_map)
        return (
            dynamic_patch_polygon,
            static_patch_polygon,
            patch_road_users,
            mask_patch_road_users,
            patch_vectorized_map,
        )
    
    def get_goal_mask(self, mask_log: Tensor) -> Tensor:
        mask_future = mask_log[:, self.current_frame+1:]
        goal_indices = mask_future.cumprod(dim=-1).sum(dim=-1) + self.current_frame
        has_goal = goal_indices > self.current_frame
        if not self.target_goal:
            has_goal[0] = False
        if not self.surr_goal:
            has_goal[1:] = False
            
        mask_goal = torch.zeros_like(mask_log)
        mask_goal[torch.arange(mask_log.shape[0], device=mask_log.device)[has_goal], 
                goal_indices[has_goal]] = True
        return mask_goal
    
    def offset_goal_mask(self, mask_goal: Tensor, max_goal_offset: int) -> Tensor:
        goal_indices = mask_goal.int().argmax(dim=-1)
        has_goal = goal_indices > self.current_frame
        
        goal_offsets = torch.randint(0, max_goal_offset, size=goal_indices.shape, 
                                     device=goal_indices.device)
        goal_indices_offsetted = torch.clamp(goal_indices - goal_offsets, min=self.current_frame+1)
        
        mask_goal_offsetted = torch.zeros_like(mask_goal)
        mask_goal_offsetted[torch.arange(mask_goal.shape[0], device=mask_goal.device)[has_goal], 
                    goal_indices_offsetted[has_goal]] = True
        return mask_goal_offsetted
    
    @staticmethod
    def flip_input_along_x_axis(
        all_road_users: Tensor, vectorized_map: CityCenterlinesGraph
    ) -> Tuple[Tensor, CityCenterlinesGraph]:
        all_road_users[:, :, 1] = -all_road_users[:, :, 1]  # flip the y pos along x axis
        all_road_users[:, :, 3] = -all_road_users[:, :, 3]  # flip the y vel along x axis
        all_road_users[:, :, 4] = -all_road_users[:, :, 4]  # flip the theta along the x axis
        vectorized_map.nodes_features.start_locs[:, 1] = -vectorized_map.nodes_features.start_locs[:, 1]
        vectorized_map.nodes_features.end_locs[:, 1] = -vectorized_map.nodes_features.end_locs[:, 1]
        return all_road_users, vectorized_map

    def get_gt(self, all_road_users: Tensor) -> Tuple[Tensor, Tensor]:
        gt_vcs_pos = all_road_users[:, self.current_frame + 1 :, :2]
        gt_vcs_head_rad = all_road_users[:, self.current_frame + 1 :, 4]
        local_vcs_pos = all_road_users[:, self.current_frame, :2]
        local_vcs_head_rad = all_road_users[:, self.current_frame, 4]
        gt_local_pos, gt_local_head_rad, _ = vcs_to_local_tensor(
            local_vcs_pos, local_vcs_head_rad, gt_vcs_pos, gt_vcs_head_rad
        )
        return gt_local_pos, gt_local_head_rad

    def pad_input(self, agents_dict: Dict, lanes_nodes_dict: Dict) -> Tuple[Dict, Dict]:
        pad_num_agents = self.max_num_agents - agents_dict["agents_position"].shape[0]
        for k, v in agents_dict.items():
            pad_feature = torch.inf * torch.ones(pad_num_agents, *v.shape[1:])  # add inf values in pading nodes
            agents_dict[k] = torch.concat([agents_dict[k], pad_feature], dim=0)

        pad_num_lane_nodes = self.max_num_lane_nodes - lanes_nodes_dict["lanes_start_locs"].shape[0]
        for k, v in lanes_nodes_dict.items():
            pad_feature = torch.inf * torch.ones(pad_num_lane_nodes, *v.shape[1:])
            lanes_nodes_dict[k] = torch.concat([lanes_nodes_dict[k], pad_feature], dim=0)

        return agents_dict, lanes_nodes_dict

    def visualize_training_sample(
        self,
        data: Dict,
        dynamic_patch_polygon: Optional[Polygon] = None,
        static_patch_polygon: Optional[Polygon] = None,
        visualize_vel_head: Optional[bool] = True,
        vis_bb: Optional[bool] = True,
        vis_goal: Optional[bool] = True,
    ) -> None:
        # visulaize agent trajectories
        agent_type_ids = self.agent_type_ids
        all_road_users_pos = data["agents_position"]  # (A, T, 2)
        all_road_users_vel = data["agents_vel"]  # (A, T, 2)
        all_road_users_vel_norm = torch.norm(all_road_users_vel, p=2, dim=-1)  # (A, T)
        all_road_users_vel_head = torch.atan2(all_road_users_vel[:, :, 1], all_road_users_vel[:, :, 0])  # (A, T)
        all_road_users_head = data["agents_heading_rad"]  # (A, T)
        all_road_users_type = data["agents_type"]  # (A)
        agents_bounding_box = data["agents_size"]
        all_road_users_reg_mask = data["agents_log_reg_mask"].bool()  # (A, T)
        all_road_users_goal_mask = data["agents_goal_mask"].bool()  # (A, T)
        colors = {"Target vehicle": "red", "Bicycle": "yellow", "Pedestrian": "salmon",
                  "Surrounding vehicle": "mediumorchid"}

        gt_local_pos = data["agents_gt_position"]
        gt_local_head_rad = data["agents_gt_head_rad"]
        gt_vcs_pos, gt_vcs_head_rad, _ = local_to_vcs_tensor(
            all_road_users_pos[:, self.current_frame, :2],
            all_road_users_head[:, self.current_frame],
            gt_local_pos.unsqueeze(1),
            gt_local_head_rad.unsqueeze(1),
        )
        gt_vcs_pos = gt_vcs_pos.squeeze(1)
        gt_vcs_head_rad = gt_vcs_head_rad.squeeze(1)
        assert (all_road_users_pos[:, self.current_frame + 1 :] - gt_vcs_pos).abs().max() < 1e-3
        assert wrap_angle_rad(all_road_users_head[:, self.current_frame + 1 :] - gt_vcs_head_rad).abs().max() < 1e-3
        assert (
            all_road_users_head[:, self.current_frame + 1 :].abs().max() <= torch.pi
            or gt_vcs_head_rad.abs().max() <= torch.pi
        )

        plt.figure(figsize=(10, 10))
        for agent in range(len(all_road_users_pos) - 1, -1, -1):
            past_reg_mask = all_road_users_reg_mask[agent, : self.current_frame + 1]
            past_traj = all_road_users_pos[agent, : self.current_frame + 1][past_reg_mask]
            past_head_rad = all_road_users_head[agent, : self.current_frame + 1][past_reg_mask]
            past_bb = agents_bounding_box[agent, : self.current_frame + 1][past_reg_mask]
            future_reg_mask = all_road_users_reg_mask[agent, self.current_frame + 1 :]
            future_traj = gt_vcs_pos[agent][future_reg_mask]
            future_head_rad = gt_vcs_head_rad[agent][future_reg_mask]
            future_bb = agents_bounding_box[agent, self.current_frame + 1 :][future_reg_mask]
            if vis_goal:
                goal_mask = all_road_users_goal_mask[agent, self.current_frame + 1 :]
                goal_pos = gt_vcs_pos[agent][goal_mask]
                goal_head_rad = gt_vcs_head_rad[agent][goal_mask]
                goal_bb = agents_bounding_box[agent, self.current_frame + 1 :][goal_mask]
            
            if visualize_vel_head:
                past_vel_norm = all_road_users_vel_norm[agent, : self.current_frame + 1][past_reg_mask]
                past_vel_head_rad = all_road_users_vel_head[agent, : self.current_frame + 1][past_reg_mask]
                past_head_rad[past_vel_norm != 0.0] = past_vel_head_rad[past_vel_norm != 0.0]
                future_vel_norm = all_road_users_vel_norm[agent, self.current_frame + 1 :][future_reg_mask]
                future_vel_head_rad = all_road_users_vel_head[agent, self.current_frame + 1 :][future_reg_mask]
                future_head_rad[future_vel_norm != 0.0] = future_vel_head_rad[future_vel_norm != 0.0]
                if vis_goal:
                    goal_vel_norm = all_road_users_vel_norm[agent, self.current_frame + 1 :][goal_mask]
                    goal_vel_head_rad = all_road_users_vel_head[agent, self.current_frame + 1 :][goal_mask]
                    goal_head_rad[goal_vel_norm != 0.0] = goal_vel_head_rad[goal_vel_norm != 0.0]

            agent_type = all_road_users_type[agent].item()
            agent_label = agent_type_ids.inv_map()[str(agent_type)]
            if "target" in agent_label or "ego" in agent_label:
                agent_label = "Target vehicle"
            elif "vehicle" in agent_label:
                agent_label = "Surrounding vehicle"
            elif "bicycle" in agent_label:
                agent_label = "Bicycle"
            elif "pedestrian" in agent_label:
                agent_label = "Pedestrian"
            color = colors[agent_label]
            self.plot_traj_head_bb(future_traj, future_head_rad, future_bb, "grey", 
                                        "Recorded Logs", vis_bb, vis_head=True)
            if agent != 0: # surrounding agents
                self.plot_traj_head_bb(past_traj, past_head_rad, past_bb, color, agent_label, 
                                       vis_bb, vis_head=True)
            else: # target agent
                self.plot_traj_head_bb(past_traj, past_head_rad, past_bb, color, 
                                       agent_label, vis_bb, vis_head=True)
            if vis_goal:
                self.plot_traj_head_bb(goal_pos, goal_head_rad, goal_bb, "teal", 
                                        "Goal Position", vis_bb, vis_head=True)
            
        # visualize city graph
        nodes_end_locs = data["lanes_end_locs"]
        nodes_start_locs = data["lanes_start_locs"]
        # show the lanes node
        nodes_vector = nodes_end_locs - nodes_start_locs
        for i in range(len(nodes_vector)):  # best for resolution=2
            plt.arrow(nodes_start_locs[i, 0], nodes_start_locs[i, 1], nodes_vector[i, 0], 
                nodes_vector[i, 1], width=0.2, length_includes_head=True, zorder=3,
                color="skyblue", label="lanes",
            )

        # plot the boundaries of the patches
        if dynamic_patch_polygon is not None:
            xx, yy = dynamic_patch_polygon.exterior.coords.xy
            plt.plot(xx, yy, linestyle="--", color="white")
        if static_patch_polygon is not None:
            xx, yy = static_patch_polygon.exterior.coords.xy
            plt.plot(xx, yy, linestyle="--", color="white")

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

        # set axis limit
        minx, miny = np.stack((np.array((dynamic_patch_polygon.bounds)), 
                            np.array((static_patch_polygon.bounds)))).min(axis=0)[:2]
        maxx, maxy = np.stack((np.array((dynamic_patch_polygon.bounds)), 
                            np.array((static_patch_polygon.bounds)))).max(axis=0)[2:]

        plt.xlim(minx + 10, maxx - 10)
        plt.ylim(miny, maxy - 10)
        # plt.xlabel('x-coodinate in Target Vehicle Coordinates')
        # plt.ylabel('y-coodinate in Target Vehicle Coordinates')
        plt.axis("equal")
        plt.grid(False)
        plt.axis("off")

    @staticmethod
    def plot_traj_head_bb(
        traj: Tensor, head_rad: Tensor, bb: Tensor, color: str, agent_label: str, vis_bb: bool, 
        vis_head: bool, alpha: Optional[float] = 1.0
    ) -> None:
        plt.plot(traj[:, 0], traj[:, 1], color=color, label=agent_label, zorder=5, alpha=alpha)
        if vis_head:
            plt.quiver(traj[:, 0], traj[:, 1], np.cos(head_rad), np.sin(head_rad), color=color,
                scale=80, label=agent_label, zorder=5, alpha=alpha,
            )
        if vis_bb:
            bb_corners = get_bb_corners(traj.unsqueeze(0), head_rad.unsqueeze(0), bb.unsqueeze(0))
            for i in range(len(traj)):
                agent_bb = patches.Rectangle(xy=bb_corners[0, i, 0], width=bb[i, 1], height=bb[i, 0],
                    angle=torch.rad2deg(head_rad[i]), edgecolor=color, facecolor="none", zorder=5,
                    alpha=alpha)
                plt.gca().add_patch(agent_bb)

    def concatenate_road_users(self, sample: TrainingSample) -> Tensor:
        frames_past = sum(sample.input_mask)
        _, _, F = sample.target_trajectory.shape  # (1, T, F)
        # modify shape to incorporate one agent id in the feature
        all_road_users = torch.zeros((1, self.current_frame + 1 + self.num_future_steps, F + 1), dtype=torch.float32)
        for k, v in self.agent_type_ids.__dict__.items():
            agent_trajectories = torch.tensor(getattr(sample, k), dtype=torch.float32)
            num_agents = agent_trajectories.shape[0]
            agent_id_tensor = v * torch.ones(
                (num_agents, self.current_frame + 1 + self.num_future_steps, 1), dtype=torch.float32
            )
            if num_agents != 0:
                agent_trajectories = agent_trajectories[
                    :, frames_past - (self.current_frame + 1) :
                ]  # removing the non required input frames
                road_user = torch.concat((agent_trajectories, agent_id_tensor), dim=-1)
                all_road_users = torch.concat((all_road_users, road_user), dim=0)
        # remove the zero all road user
        all_road_users = all_road_users[1:]
        # compute the road user mask
        mask_all_road_users = all_road_users[..., 0] != 0
        return mask_all_road_users, all_road_users


class TrajectoryGridMapDatasetLMDB(TrajectoryGridMapDataset):
    def __init__(
        self,
        data_config: DataStructureConfig,
        data_file: str,
        data_split: str,
        evaluation_mode: bool = False,
        use_compressor: bool = True,
        return_sample: bool = False,
    ) -> None:
        super().__init__(data_config, evaluation_mode)

        self.db_path = os.path.join(data_file, data_split)

        self._init_db()  # init db to get length
        self.env = None  # reset env to None, so that dataset can be pickled
        self.decompressor = None  # init decompressor to None, so that dataset can be pickled
        self.use_compressor = use_compressor
        self.return_sample = return_sample

    def _init_db(self):
        self.env = lmdb.open(
            self.db_path, subdir=os.path.isdir(self.db_path), readonly=True, lock=False, readahead=False, meminit=False
        )

        with self.env.begin(write=False) as txn:
            self.length = self.loads(txn.get(b"__len__"))
            self.keys = self.loads(txn.get(b"__keys__"))

    def __getitem__(self, index: int) -> Dict:
        if self.env is None:
            self._init_db()
        if self.decompressor is None and self.use_compressor:
            self.decompressor = zstandard.ZstdDecompressor()

        with self.env.begin(write=False) as txn:
            data = txn.get(self.keys[index])

        if self.use_compressor:
            data = self.decompressor.decompress(data)

        sample = self.loads(data)
        if self.return_sample:
            return sample
        else:
            return self.process_sample(sample)

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + " (" + self.db_path + ")"

    @staticmethod
    def loads(obj):
        return pickle.loads(obj)


class TrajectoryGridDataModule(LightningDataModule):
    def __init__(self, data_config: DataStructureConfig, training_config: TrainingConfig):
        super().__init__()
        self.train = None
        self.val = None
        self.test = None
        self.data_config = data_config
        self.training_config = training_config

        self.dataset = TrajectoryGridMapDatasetLMDB

    def prepare_data(self):
        # called only on 1 GPU
        pass

    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        self.train = self.dataset(
            data_config=self.data_config, data_file=self.data_config.db_folder, data_split="train", evaluation_mode=False,
        )

        self.val = self.dataset(
            data_config=self.data_config, data_file=self.data_config.db_folder, data_split="train_val", evaluation_mode=True,
        )

        self.test = self.dataset(
            data_config=self.data_config, data_file=self.data_config.db_folder, data_split="val", evaluation_mode=True
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            num_workers=self.training_config.num_worker,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.training_config.batch_size,
            shuffle=False,
            num_workers=self.training_config.num_worker,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.training_config.batch_size,
            shuffle=False,
            num_workers=self.training_config.num_worker,
            pin_memory=True,
        )


if __name__ == "__main__":
    import time
    from config.config import TARGET_PATH

    data_config = DataStructureConfig()
    train_config = TrainingConfig()
    data_modules = TrajectoryGridDataModule(data_config=data_config, training_config=train_config)
    data_modules.setup()
    dataloader = data_modules.train_dataloader()

    start = time.time()
    for batch, sample_batched in enumerate(dataloader):
        # print(batch)
        # print('------------ First Sample ----------\n', sample_batched[0])
        # print('------------ Second Sample ----------\n', sample_batched[1])
        # print('------------ Sampled Batch ----------\n', sample_batched)
        # assert (sample_batched['lanes_nodes']['num_nodes'] ==
        #         sample_batched[0]['lanes_nodes']['num_nodes']  +
        #         sample_batched[1]['lanes_nodes']['num_nodes']), 'batch !=2 or some issue with sampling'
        if batch % 50 == 0:
            # break
            end = time.time()
            print("Time to sample: ", np.round(end - start, 2))
            start = time.time()
