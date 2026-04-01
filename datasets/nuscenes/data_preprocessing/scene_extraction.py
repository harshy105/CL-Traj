import os
import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt
from datasets.nuscenes.nuscenes_devkit import NuScenes

from config.config import NUSCENES_PATH, EXTRACTED_SCENES_DB
from config.nuscenes_config import NuscenesConfig, SceneInfo, ObjectSampleNuscenes, ExtractedScene
from config.nuscenes_config import RoadObjectSampleNuscenes
from datasets.utilities.lmdb_database_creator import LMDBDatabaseCreator
from utilities.transformation import calculate_yaw_angle
from datasets.nuscenes.nuscenes_devkit.prediction import PredictHelper
from datasets.nuscenes.nuscenes_devkit.eval.common.utils import angle_diff

from datasets.nuscenes.nuscenes_devkit.prediction.input_representation.static_layers import StaticLayerRasterizer
from datasets.nuscenes.nuscenes_devkit.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from datasets.nuscenes.nuscenes_devkit.prediction.input_representation.interface import InputRepresentation
from datasets.nuscenes.nuscenes_devkit.prediction.input_representation.combinators import Rasterizer
from datasets.nuscenes.nuscenes_devkit.eval.prediction.splits import get_prediction_challenge_scene_split


class TrajectoryExtraction:
    def __init__(
            self,
            nuscenes: NuScenes,
            nuscenes_cfg: NuscenesConfig
    ) -> None:
        # NuScene dataset initialization
        self.data_root = NUSCENES_PATH
        self.nuscenes_cfg = nuscenes_cfg
        self.nuscenes = nuscenes

        # define sensor to get sample_data to extract the ego pose
        self.sensor = "LIDAR_TOP"
        self.helper = PredictHelper(self.nuscenes)

    def process_and_save_scenes_lmdb(self, data_split: str):
        scene_list = get_prediction_challenge_scene_split(data_split, dataroot=self.data_root)
        max_size = int(5 * 1e5 * len(scene_list))
        save_path = os.path.join(EXTRACTED_SCENES_DB, data_split)

        lmdb_db_creator = LMDBDatabaseCreator(save_path, max_size=max_size, use_compressor=True)

        for num, scene_name in enumerate(scene_list):
            scene = self._get_scene_by_name(scene_name)
            print(scene['name'])
            processed_scene = self.extract_data_from_scene(scene)
            lmdb_db_creator.write_sample(num, processed_scene)
        lmdb_db_creator.write_meta_data(len(scene_list))

    def extract_data_from_scene(self, cur_scene: dict) -> ExtractedScene:
        # initialize with first time step
        first_sample_token = cur_scene['first_sample_token']

        # Scene information + Log information
        scene_info = self.extract_scene_and_log_information(cur_scene)

        # Ego-Vehicle pose
        # to get information -> sample -> sample data of any sensor -> ego pose token -> ego pose
        ego_vehicle_pose = self.extract_ego_pose(first_sample_token)

        # extract pedestrian trajectories
        pedestrian = self.extract_trajectories(first_sample_token, self.nuscenes_cfg.pedestrian_types)

        # extract bicycle trajectories
        bicycle = self.extract_trajectories(first_sample_token, self.nuscenes_cfg.bicycle_types)

        # extract vehicle trajectories
        vehicle = self.extract_trajectories(first_sample_token, self.nuscenes_cfg.vehicle_types)

        # Objects
        # movable objects
        movable_objects = self.extract_objects(first_sample_token, self.nuscenes_cfg.movable_object_types)

        # static objects
        static_objects = self.extract_objects(first_sample_token, self.nuscenes_cfg.static_object_types)

        return ExtractedScene(
            scene_info=scene_info,
            ego_trajectory=ego_vehicle_pose,
            pedestrian_trajectories=pedestrian,
            vehicle_trajectories=vehicle,
            bicycle_trajectories=bicycle,
            movable_objects=movable_objects,
            static_objects=static_objects)

    def extract_scene_and_log_information(self, cur_scene: dict) -> SceneInfo:
        # Scene information + Log information
        log_info = self.nuscenes.get('log', cur_scene['log_token'])
        return SceneInfo(name=cur_scene['name'], description=cur_scene['description'],
                         location=log_info['location'], num_samples=cur_scene['nbr_samples'])

    def extract_trajectories(self, first_sample_token: str,
                             road_user_types: List[str]) -> Dict[str, Dict[int, ObjectSampleNuscenes]]:
        # list of road user dictionaries
        road_user = {}

        frame = 0
        # generate sample token list for a scene
        sample = self.nuscenes.get('sample', first_sample_token)
        sample_token_list = [first_sample_token]
        while sample['next'] != '':
            sample = self.nuscenes.get('sample', sample['next'])
            sample_token_list.append(sample['token'])

        ########################################################
        # extract the trajectories, instance_token = road_user_id
        # per scene person token database (since road user's trajectories can start from different time steps)

        # Go through the annotations of a scene frame by frame
        # find the same road user through the instance token
        for sample_token in sample_token_list:
            sample = self.nuscenes.get('sample', sample_token)
            cur_timestamp = sample['timestamp']

            for ann in sample['anns']:
                # go through all annotation within a frame
                sample_annotation = self.nuscenes.get('sample_annotation', ann)

                if sample_annotation['category_name'] in road_user_types:
                    if sample_annotation['instance_token'] in road_user:
                        # if road users has been detected before (instance token/ track id)
                        new_road_user_annotation = self.calculate_trajectory_point(
                            sample_annotation, frame, cur_timestamp)
                        road_user[sample_annotation['instance_token']][frame] = new_road_user_annotation

                    else:
                        # new road user -> create a new dict -> frame as key
                        a_new_road_user = dict()
                        a_new_road_user[frame] = self.calculate_trajectory_point(sample_annotation, frame,
                                                                                 cur_timestamp)
                        road_user[sample_annotation['instance_token']] = a_new_road_user
            frame += 1
        return road_user

    def calculate_trajectory_point(self, sample_annotation, frame: int, timestamp: int) -> ObjectSampleNuscenes:
        # Values of interest: Timestamp, x, y, z, v_x, v_y, v_z, yaw (heading),
        # width, length, height, visibility, attribute, category
        instance_token = sample_annotation['instance_token']
        sample_token = sample_annotation['sample_token']

        position = np.array(sample_annotation['translation'])  # x, y, z
        velocity = self.helper.get_velocity_for_agent(instance_token, sample_token)
        acceleration = self.helper.get_acceleration_for_agent(instance_token, sample_token)
        assert velocity is np.nan or (position.shape == velocity.shape and velocity.shape == acceleration.shape)
        # Radians / second.
        heading_change_rate = self.helper.get_heading_change_rate_for_agent(instance_token, sample_token)

        # check for nan
        if np.any(np.isnan(velocity)):
            velocity = np.array([0, 0])
        if np.any(np.isnan(acceleration)):
            acceleration = np.array([0, 0])
        if np.isnan(heading_change_rate):
            heading_change_rate = 0

        yaw = (calculate_yaw_angle(sample_annotation['rotation']) - 90) * -1
        box_size = np.array(sample_annotation['size'])
        category_name = sample_annotation['category_name']
        try:
            attribute = self.nuscenes.get('attribute', sample_annotation['attribute_tokens'][0])['name']
        except IndexError or KeyError:
            attribute = ""

        visibility = int(sample_annotation['visibility_token'])  # 4 = highest visibility
        instance_token = sample_annotation['instance_token']
        sample_token = sample_annotation['sample_token']
        return ObjectSampleNuscenes(frame=frame, position=position, velocity=velocity, acceleration=acceleration,
                                    yaw=yaw, heading_change_rate=heading_change_rate, box_size=box_size,
                                    category_name=category_name, attribute=attribute, visibility=visibility,
                                    instance_token=instance_token, sample_token=sample_token, timestamp=timestamp)

    def extract_objects(self, first_sample_token: str,
                        object_types: List[str]) -> Dict[str, Dict[int, RoadObjectSampleNuscenes]]:
        # list of road objects dictionaries
        road_objects = {}
        frame = 0
        # generate sample token list for a scene
        sample = self.nuscenes.get('sample', first_sample_token)
        sample_token_list = [first_sample_token]
        while sample['next'] != '':
            sample = self.nuscenes.get('sample', sample['next'])
            sample_token_list.append(sample['token'])

        ########################################################
        for sample_token in sample_token_list:
            sample = self.nuscenes.get('sample', sample_token)
            cur_timestamp = sample['timestamp']

            for ann in sample['anns']:
                # go through all annotation within a frame
                sample_annotation = self.nuscenes.get('sample_annotation', ann)
                if sample_annotation['category_name'] in object_types:
                    if sample_annotation['instance_token'] in road_objects:
                        # if road users has been detected before (instance token/ track id)
                        new_road_object_annotation = self._get_object(sample_annotation, frame, cur_timestamp)
                        road_objects[sample_annotation['instance_token']][frame] = new_road_object_annotation

                    else:
                        # new road user -> create a new dict -> frame as key
                        a_new_road_object = dict()
                        a_new_road_object[frame] = self._get_object(sample_annotation, frame, cur_timestamp)
                        road_objects[sample_annotation['instance_token']] = a_new_road_object
            frame += 1
        return road_objects

    @staticmethod
    def _get_object(sample_annotation, frame, timestamp) -> RoadObjectSampleNuscenes:
        # Values of interest: Timestamp, x, y, z, yaw (heading),
        # width, length, height, visibility, attribute, category
        return RoadObjectSampleNuscenes(frame=frame, timestamp=timestamp, position=sample_annotation['translation'],
                                        yaw=(calculate_yaw_angle(sample_annotation['rotation']) - 90) * -1,
                                        box_size=sample_annotation['size'],
                                        category_name=sample_annotation['category_name'],
                                        visibility=int(sample_annotation['visibility_token']),  # 4 = highest visibility
                                        instance_token=sample_annotation['instance_token'],
                                        sample_token=sample_annotation['sample_token'])

    def extract_ego_pose(self, first_sample_token: str) -> Dict[int, ObjectSampleNuscenes]:
        # Ego pose trajectory: timestamp, x, y, yaw,
        # z is always zero
        # yaw: 0 is north
        ego_trajectory = {}

        ego_bbox_size = np.array([1.73, 4.08, 1.56])

        first_sample = self.nuscenes.get('sample', first_sample_token)
        first_sample_data = self.nuscenes.get('sample_data', first_sample['data'][self.sensor])
        first_ego_pose = self.nuscenes.get('ego_pose', first_sample_data['ego_pose_token'])

        # Initialize with first frame
        yaw = (calculate_yaw_angle(first_ego_pose['rotation']) - 90) * -1
        position = np.array(first_ego_pose['translation'])
        timestamp = first_ego_pose['timestamp']
        frame = 0

        # add first ego sample
        ego_trajectory[frame] = ObjectSampleNuscenes(frame=frame, position=position, velocity=np.zeros(3),
                                                     acceleration=np.zeros(3), yaw=yaw, heading_change_rate=0,
                                                     box_size=ego_bbox_size, category_name="ego_vehicle",
                                                     visibility=4, attribute=None,
                                                     instance_token=first_ego_pose['token'],
                                                     sample_token=first_sample_data['sample_token'],
                                                     timestamp=timestamp)
        sample = first_sample
        while sample['next'] != '':
            frame += 1
            sample = self.nuscenes.get('sample', sample['next'])
            sample_data = self.nuscenes.get('sample_data', sample['data'][self.sensor])
            ego_pose = self.nuscenes.get('ego_pose', sample_data['ego_pose_token'])

            # attribute
            yaw = (calculate_yaw_angle(ego_pose['rotation']) - 90) * -1
            position = np.array(ego_pose['translation'])
            timestamp = ego_pose['timestamp']

            time_diff = (timestamp - ego_trajectory[frame-1].timestamp) * 1e-6
            velocity = (position - ego_trajectory[frame-1].position) / time_diff

            # calculate acceleration
            if ego_trajectory[frame-1].velocity[0] != 0:
                acceleration = (velocity - ego_trajectory[frame-1].velocity) / time_diff

                heading_change_rate = angle_diff(
                    np.deg2rad(yaw), np.deg2rad(ego_trajectory[frame-1].yaw), period=2 * np.pi) / time_diff

            else:
                acceleration = np.zeros(2)
                heading_change_rate = 0

            if (velocity[0]**2 + velocity[1]**2)**0.5 < 0.1:
                attribute = 'vehicle.stopped'
            else:
                attribute = 'vehicle.moving'

            ego_trajectory[frame] = ObjectSampleNuscenes(
                frame=frame, position=position, velocity=velocity,
                acceleration=acceleration, yaw=yaw, heading_change_rate=heading_change_rate,
                box_size=ego_bbox_size, category_name="ego_vehicle", visibility=4, attribute=attribute,
                instance_token=ego_pose['token'], sample_token=sample_data['sample_token'], timestamp=timestamp)
        return ego_trajectory

    def rasterize_like_covernet(self, instance_token_img, sample_token_img):
        static_layer_rasterizer = StaticLayerRasterizer(self.helper)
        agent_rasterizer = AgentBoxesWithFadedHistory(self.helper, seconds_of_history=1)
        mtp_input_representation = InputRepresentation(static_layer_rasterizer, agent_rasterizer, Rasterizer())

        img = mtp_input_representation.make_input_representation(instance_token_img, sample_token_img)

        plt.imshow(img)
        plt.show()

    def _get_scene_by_name(self, scene_name: str):
        for scene in self.nuscenes.scene:
            if scene['name'] == scene_name:
                return scene
        print(f"Scene {scene_name} not found.")


if __name__ == "__main__":
    # data split: train, train_val, val (test)
    nusc_cfg = NuscenesConfig()
    
    for split in ['mini_train', 'mini_val', 'train', 'train_val', 'val']:
        if 'mini' in split:
            data_set = 'v1.0-mini'
        else:
            data_set = 'v1.0-trainval'
          
        nuscenes = NuScenes(version=data_set, dataroot=NUSCENES_PATH, verbose=True)
        scene_extractor = TrajectoryExtraction(nuscenes=nuscenes, nuscenes_cfg=nusc_cfg)
        scene_extractor.process_and_save_scenes_lmdb(split)
        
        del nuscenes, scene_extractor
