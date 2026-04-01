import os, pickle
from typing import Optional, Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from datasets.nuscenes.nuscenes_devkit import NuScenes
from datasets.nuscenes.nuscenes_devkit.eval.prediction.splits import get_prediction_challenge_scene_split

from config.config import NUSCENES_PATH, EXTRACTED_SCENES_DB, EXTRACTED_MAP
from config.train_config import TrainingSample
from config.nuscenes_config import (NuScenesPreprocessConfig, ExtractedScene, 
                                    ObjectSampleNuscenes, NuscenesConfig,
                                    CityCenterlinesGraph)
from datasets.nuscenes.data_preprocessing.scene_extraction import TrajectoryExtraction
from datasets.utilities.lmdb_loader import LMDBLoader
from utilities.transformation import create_patch, sample_subgraph_in_patch


class SampleCreator:
    def __init__(
            self,
            nuscenes_preprocess_config: NuScenesPreprocessConfig,
            nuscenes: NuScenes,
            data_split: str,
            data_set: str,
    ) -> None:
        """
        Helper class to create the expected data structure from [instance and sample token]
        :param nuscenes: nuScenes data instance
        :param data_split: e.g. train_val, val, train
        """
        self.nuscenes = nuscenes
        self.data_split = data_split

        self.data_root = os.path.join(NUSCENES_PATH, data_set)
        self.test_scene_names = get_prediction_challenge_scene_split(data_split, dataroot=self.data_root)

        self.region_size = nuscenes_preprocess_config.region_size
        self.translation = nuscenes_preprocess_config.translation
        self.frames_past = nuscenes_preprocess_config.frames_past
        self.frames_future = nuscenes_preprocess_config.frames_future
        self.scene_extractor = TrajectoryExtraction(nuscenes=nuscenes, nuscenes_cfg=NuscenesConfig)
        self.lmdb_loader = LMDBLoader(data_path=os.path.join(EXTRACTED_SCENES_DB, self.data_split))
        with open(EXTRACTED_MAP, 'rb') as inputfile:
            self.cities_centerlines_graph = pickle.load(inputfile)

    def create_sample_from_token(
            self,
            token: str,
            visualize_sample: bool = False,
    ) -> TrainingSample:
        """
        Creates input structure from token -> expected data structure from data generator
        :param (str) token: instance_token, sample_token
        :param (int) number: token number
        :param (str) data_split: Optional
        :param (bool) visualize_sample: if true visualize training sample
        :return: (TrainingSample)
        """
        instance_token, sample_token = token.split("_")
        extracted_scene, target_vehicle, t0 = self.get_trajectories_from_sample_token(instance_token, sample_token)

        # # #------------------------------------------------------------------------------------------------------# # #
        # Get the target trajectory and the dynamic context

        # relevant indices for current time step t0
        ind_past = [ind for ind in range(t0-(self.frames_past - 1), t0 + 1)]
        ind_predict = [ind for ind in range(t0 + 1, t0 + self.frames_future+1)]
        time_ind_sample = ind_past + ind_predict

        assert len(time_ind_sample) == self.frames_past + self.frames_future

        # create patch
        location = extracted_scene.scene_info.location
        cur_target_pose = target_vehicle[t0]  # ego trajectory
        xy_target = np.array(cur_target_pose.position[:2])
        angle = cur_target_pose.yaw  # angle in driving direction (Left hand coordiante system)
        _, patch_container = create_patch(self.region_size, angle_rad=np.deg2rad(-angle), 
                                            agent_pos=xy_target, offset=self.translation)
        
        # create the frames_past mask
        input_mask = np.zeros((self.frames_past + self.frames_future), dtype=bool)
        input_mask[:self.frames_past] = True
        
        # create data structure -> list of points for every time step
        trajectories = {'ego_vehicle': [], 'target_vehicle': [], 'vehicle': [], 'large_vehicle': [],
                        'pedestrian': [], 'bicycle': [], 'stationary_vehicle': [], 'stationary_object': []}

        for road_user_type in ['pedestrian', 'vehicle', 'bicycle', 'ego_vehicle']:
            if road_user_type != 'ego_vehicle':
                road_users = getattr(extracted_scene, road_user_type + "_trajectories")
            else:
                road_users = {extracted_scene.ego_trajectory[t0].instance_token: extracted_scene.ego_trajectory}
            for num, road_user_instance in enumerate(road_users):
                # get road user track
                road_user = road_users[road_user_instance]

                road_user_trajectory = np.zeros((1, self.frames_past + self.frames_future, 9))
                cur_road_user_attribute, cur_road_user_category = None, None

                for j, t_ind in enumerate(time_ind_sample):
                    # get trajectory for road user
                    if t_ind in road_user:
                        cur_road_user = road_user[t_ind]
                        if patch_container(cur_road_user.position[:2]):
                            road_user_trajectory[0, j, :] = [*cur_road_user.position[:2], *cur_road_user.velocity[:2],
                                                            cur_road_user.yaw, *cur_road_user.acceleration[:2],
                                                            *cur_road_user.box_size[:2]]
                        if j < self.frames_past:  # input attributes
                            cur_road_user_attribute = cur_road_user.attribute
                            cur_road_user_category = cur_road_user.category_name

                # sort out road users without any input trajectory points
                if (sum(road_user_trajectory[0, :self.frames_past, 0] != 0) > 0 and
                        sum(road_user_trajectory[0, :self.frames_past, 1] != 0) > 0):

                    cur_velo = 0.0
                    if t0 in road_user:
                        cur_velo = sum(road_user[t0].velocity[:2]**2)**0.5

                    if road_user_instance == instance_token:  # check if road user is the target vehicle
                        trajectories['target_vehicle'] = road_user_trajectory

                    elif road_user_type == 'pedestrian':
                        trajectories['pedestrian'].append(road_user_trajectory)

                    elif road_user_type == 'ego_vehicle':
                        trajectories['ego_vehicle'] = road_user_trajectory

                    elif road_user_type == 'bicycle':
                        if cur_road_user_attribute == 'cycle.without_rider':
                            trajectories['stationary_vehicle'].append(road_user_trajectory)  # FIXME: to stationary?!
                            # trajectories['bicycle'].append(road_user_trajectory)
                        else:
                            trajectories['bicycle'].append(road_user_trajectory)

                    elif road_user_type == 'vehicle':
                        # if cur_road_user_attribute in ['vehicle.parked', 'cycle.without_rider'] and cur_velo < 0.5:
                        if cur_road_user_attribute in ['vehicle.parked'] and cur_velo < 0.5:
                            trajectories['stationary_vehicle'].append(road_user_trajectory)
                        elif cur_road_user_category in ['vehicle.bus.bendy', 'vehicle.trailer', 'vehicle.bus.rigid',
                                                        'vehicle.truck']:
                            trajectories['large_vehicle'].append(road_user_trajectory)
                        else:
                            trajectories['vehicle'].append(road_user_trajectory)
        # # #------------------------------------------------------------------------------------------------------# # #
        # Get static road objects
        stationary_vehicle = self._concatenate_trajectories(trajectories['stationary_vehicle'])
        static_object_list = []
        for object_type in ['movable_objects', 'static_objects']:
            static_road_objects = getattr(extracted_scene, object_type)
            for static_object_id in static_road_objects:
                static_object = static_road_objects[static_object_id]
                if t0 in static_object:
                    cur_static_obj = static_object[t0]
                    static_object_list.append(np.array([*cur_static_obj.position[:2], cur_static_obj.yaw,
                                                        *cur_static_obj.box_size[:2]]))
        static_object_list = np.array(static_object_list)
        if len(stationary_vehicle) > 0:
            if len(static_object_list) > 0:
                all_stationary_objects = np.row_stack([stationary_vehicle[..., 2, [0, 1, 4, 7, 8]], static_object_list])
            else:
                all_stationary_objects = stationary_vehicle[..., 2, [0, 1, 4, 7, 8]]
        else:
            all_stationary_objects = static_object_list

        # # #------------------------------------------------------------------------------------------------------# # #
        # get the ceterline maps
        city_centerlines_graph = self.cities_centerlines_graph[location]
        city_centerlines_subgraph = sample_subgraph_in_patch(patch_container, city_centerlines_graph)

        train_sample = TrainingSample(
            input_mask=input_mask,
            ego_trajectory=trajectories['ego_vehicle'],
            target_trajectory=trajectories['target_vehicle'],
            vehicle_trajectories=self._concatenate_trajectories(trajectories['vehicle']),
            large_vehicle_trajectories=self._concatenate_trajectories(trajectories['large_vehicle']),
            pedestrian_trajectories=self._concatenate_trajectories(trajectories['pedestrian']),
            bicycle_trajectories=self._concatenate_trajectories(trajectories['bicycle']),
            stationary_vehicles=stationary_vehicle,
            stationary_objects=all_stationary_objects,
            vectorized_map=city_centerlines_subgraph
        )

        # Visualize input
        if visualize_sample:
            self.visualize_training_sample(train_sample, city_centerlines_subgraph)

        return train_sample

    def get_trajectories_from_sample_token(
            self, instance_token: str, sample_token: str) -> Tuple[ExtractedScene, Dict[int, ObjectSampleNuscenes], int]:
        extracted_scene = self.load_processed_scene_lmdb(sample_token)
        target_vehicle, cur_frame = self.get_target_id_and_current_frame(extracted_scene, instance_token, sample_token)
        return extracted_scene, target_vehicle, cur_frame

    @staticmethod
    def get_target_id_and_current_frame(cur_scene: ExtractedScene, instance_token: str,
                                        sample_token: str) -> Optional[Tuple[Dict[int, ObjectSampleNuscenes], int]]:
        if instance_token in cur_scene.vehicle_trajectories:
            target_vehicle = cur_scene.vehicle_trajectories[instance_token]
            for t_id in target_vehicle:
                if target_vehicle[t_id].sample_token == sample_token:
                    return target_vehicle, target_vehicle[t_id].frame
        else:
            # ego vehicle as target case
            for t_id in cur_scene.ego_trajectory:
                if cur_scene.ego_trajectory[t_id].sample_token == sample_token:
                    return cur_scene.ego_trajectory, cur_scene.ego_trajectory[t_id].frame

    def load_processed_scene_lmdb(self, sample_token: str) -> ExtractedScene:
        scene = self.get_scene(sample_token)
        ind = self.test_scene_names.index(scene['name'])
        return self.lmdb_loader.__getitem__(ind)
    
    def get_scene(self, sample_token: str):
        sample = self.nuscenes.get('sample', sample_token)
        scene = self.nuscenes.get('scene', sample['scene_token'])
        return scene
    
    @staticmethod
    def _concatenate_trajectories(trajectories: List[np.ndarray]) -> np.ndarray:
        if len(trajectories) > 0:
            return np.concatenate(trajectories)
        else:
            return np.zeros(0)
        
    def visualize_training_sample(self, train_sample: TrainingSample, 
                                  city_centerlines_subgraph:CityCenterlinesGraph) -> None:

        road_user_types = ["vehicle_trajectories", "stationary_vehicles", "bicycle_trajectories",
                           'large_vehicle_trajectories', 'target_trajectory', 'ego_trajectory',
                           'pedestrian_trajectories']
        colors = ['blue', 'black', 'yellow', 'gray', 'red', 'green', 'orange']
        input_mask = train_sample.input_mask

        plt.figure(figsize=(10,10))
        for num, road_user_type in enumerate(road_user_types):
            trajectories = getattr(train_sample, road_user_type)
            for track_num, track in enumerate(trajectories):
                ind = track[:, 0] != 0.0
                ind_input = np.logical_and(ind, input_mask)
                ind_output = np.logical_and(ind, ~input_mask)
                plt.plot(track[ind_input, 0], track[ind_input, 1], c=colors[num], marker='s', markersize=3,
                            label=road_user_type)
                plt.plot(track[ind_output, 0], track[ind_output, 1], c=colors[num], marker='x', markersize=3,
                            label=road_user_type)
                

        # visualize city graph
        edges_indices = city_centerlines_subgraph.edges_indices
        nodes_start_locs = city_centerlines_subgraph.nodes_features.start_locs
        nodes_end_locs = city_centerlines_subgraph.nodes_features.end_locs
        nodes_avg_locs = (nodes_start_locs + nodes_end_locs)/2
        source_nodes_avg_locs = nodes_avg_locs[edges_indices[0,:]]
        target_nodes_avg_locs = nodes_avg_locs[edges_indices[1,:]]
        graph_edges = target_nodes_avg_locs - source_nodes_avg_locs
        for i in range(len(graph_edges)): # best for resolution=2
            plt.arrow(source_nodes_avg_locs[i,0], source_nodes_avg_locs[i,1], 
                    graph_edges[i,0], graph_edges[i,1], width=0.08, length_includes_head=True)
        
        # removing multiple legends
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        for handle in by_label.values():
            handle.set_alpha(0.8)
        plt.legend(by_label.values(), by_label.keys(), fontsize="10", loc='upper left')   
        
        plt.axis('equal')
        plt.xlabel('x-coodinate in Global Coordinate System')
        plt.ylabel('y-coodinate in Global Coordinate System')
        plt.show()
        plt.close('all')