import os
import random
import warnings
from pathlib import Path
import shutil
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Iterable, Dict, Union
from functools import cached_property
import scipy
import torch
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

from config.config import TARGET_PATH_DS
from config.train_config import TrainingSample
from config.deep_scenario_config import DSPreprocessingConfig, CityCenterlinesGraph

from datasets.deep_scenario.utils.map_parser import DSMapParser
from datasets.deep_scenario.utils.map_utils import HeadingFormat, process_heading, load_json_or_parquet
from datasets.utilities.lmdb_database_creator import LMDBDatabaseCreator


@dataclass
class DSRoadUser:
    """
    Class to store road user information
    @param track_id: int
    @param category: str
    @param timestamps: np.ndarray (n, ) with relative timestamps in seconds
    @param pos: np.ndarray(n, 2) with x and y position in UTM
    @param velocity: np.ndarray(n, 2) with x and y velocity in m/s
    @param heading: np.ndarray(n, ) with heading in degrees
    @param bbox_size: np.ndarray(n, 2) with width and length of bounding box in meters
    """

    track_id: int
    category: str
    timestamps: np.ndarray
    pos: np.ndarray
    velocity: np.ndarray
    heading: np.ndarray
    bbox_size: np.ndarray

    @cached_property
    def acceleration(self) -> np.ndarray:
        if len(self.timestamps) > 2:
            dt = np.diff(self.timestamps)
            first_axay = np.divide(self.velocity[1] - self.velocity[0], dt[0]).reshape(
                (1, 2)
            )  # right derivative for 0th timestamp
            axay = np.divide(
                self.velocity[2:] - self.velocity[:-2], (dt[:-1] + dt[1:])[:, None]
            )  # mean of left and right derivatives
            last_axay = np.divide(
                self.velocity[-1] - self.velocity[-2], dt[-1]
            ).reshape((1, 2))  # left derivative for last timestamp
            axay = np.concatenate((first_axay, axay, last_axay), axis=0)
        elif len(self.timestamps) == 2:
            axay = np.tile(
                np.divide(
                    self.velocity[1] - self.velocity[0],
                    self.timestamps[1] - self.timestamps[0],
                ),
                (2, 1),
            )
        elif len(self.timestamps) == 1:
            axay = np.array([0, 0]).reshape((1, 2))
        else:
            raise ValueError
        return axay

    def as_trajectory(self) -> np.ndarray:
        trajectory = np.zeros((len(self.timestamps), 9))
        trajectory[:, 0:2] = self.pos
        trajectory[:, 2:4] = self.velocity
        trajectory[:, 4] = self.heading
        trajectory[:, 5:7] = self.acceleration
        trajectory[:, 7:9] = self.bbox_size[:, [1, 0]]
        return trajectory

    def to_trajectory(self, timestamps: Union[List[int], np.ndarray]) -> np.ndarray:
        interp_func = scipy.interpolate.interp1d(
            self.timestamps, self.as_trajectory(),
            axis=0, bounds_error=False, fill_value=0
        )
        trajectory = interp_func(timestamps)
        trajectory = np.nan_to_num(trajectory, nan=0.0)
        return trajectory

    def postprocess(self):
        self._check()
        # self._tag_stationary()
        return self

    def _check(self):
        def integrate(param_name: str) -> np.ndarray:
            param_values = getattr(self, param_name)
            dt = np.diff(self.timestamps, axis=0)[:, None]
            integrated_values = np.cumsum(
                param_values[:-1] * dt, axis=0
            )
            integrated_values = np.insert(integrated_values, 0, 0, axis=0)
            return integrated_values

        calc_rel_vel = integrate("acceleration")
        # if np.allclose(calc_rel_vel, self.velocity - self.velocity[0]):
            # warnings.warn(f"Velocity is too different from integrated acceleration")

        calc_rel_pos = integrate("velocity")
        # if np.allclose(calc_rel_pos, self.pos - self.pos[0]):
            # warnings.warn(f"Position is too different from integrated velocity")

        return

    def _tag_stationary(self):
        velocity_threshold = 0.5
        if np.all(np.linalg.norm(self.velocity, axis=1) < velocity_threshold):
            self.category = "stationary_vehicle"
        return


class LMDBHelper:
    def __init__(self, save_path: str, split_name: str):
        self.save_path = save_path
        self.split_name = split_name

        self.lmdb = LMDBDatabaseCreator(
            save_path=save_path + "/" + split_name,
            max_size=int(6e10),
            write_frequency=1000,
            use_compressor=True,
        )
        self.current_sample_index = 0

    def write_sample(self, sample: TrainingSample):
        self.lmdb.write_sample(self.current_sample_index, sample)
        self.current_sample_index += 1

    def write_meta_data(self):
        self.lmdb.write_meta_data(self.current_sample_index)
        print(f"Saved {self.current_sample_index} samples in {self.split_name}")


class DSSampleCreator:
    def __init__(
        self,
        dataset_path: str,
        save_path: str,
        preprocessing_config: DSPreprocessingConfig,
        visualize: bool = False,
        save: bool = True,
    ):
        self.config = preprocessing_config
        self.dataset_path = dataset_path
        self.save_path = save_path
        self.visualize = visualize
        self.save = save

        # init num of sample counter from each scenario
        self.scenario_num_samples_dict = {}
        for scenario in self.config.scenarios:
            self.scenario_num_samples_dict[scenario] = 0

        # this depends on data_generator implementation
        self.heading_format = HeadingFormat.from_list(
            self.config.target_heading_format)

        self.map_parser = DSMapParser(
            output_heading_format=self.heading_format,
        )

        split_names = []
        if self.config.train_scenario:
            split_names.append("train")
        if self.config.train_val_scenario:
            split_names.append("train_val")

        self.val_split_map = {}
        if self.config.val_scenario:
            for scenario in self.config.val_scenario:
                val_split_name = f"val_{scenario.lower().replace(' ', '_')}"
                split_names.append(val_split_name)
                self.val_split_map[scenario] = val_split_name

        self.lmdbs = {
            split_name: LMDBHelper(save_path, split_name)
            for split_name in split_names
        }

    def parse_dataset(self):
        if self.save:
            print(f"Saving processed samples to {self.save_path}")

        recordings = self._get_all_recordings()

        self.recordings_per_scenario_count = {}
        for scenario_name, _ in recordings:
            self.recordings_per_scenario_count[scenario_name] = self.recordings_per_scenario_count.get(scenario_name, 0) + 1

        recordings = self._add_split_to_recs(recordings)

        recordings_by_split = {}
        for rec in recordings:
            split = rec[2]
            if split not in recordings_by_split:
                recordings_by_split[split] = []
            recordings_by_split[split].append(rec)

        processing_order = ["train_val", "train"] + sorted(list(self.val_split_map.values()))
        total_samples_by_split = {split: 0 for split in self.lmdbs.keys()}

        for split in processing_order:
            if split in recordings_by_split:
                split_recordings = recordings_by_split[split]
                print(f"Processing split '{split}' with {len(split_recordings)} recordings...")

                scenarios_in_split = list(set(rec[0] for rec in split_recordings))
                
                total_samples_target = 0
                for s in scenarios_in_split:
                    total_samples_target += self.config.num_samples_per_scenario.get(s, 0)

                recordings_by_scenario = {}
                for rec in split_recordings:
                    scenario_name = rec[0]
                    if scenario_name not in recordings_by_scenario:
                        recordings_by_scenario[scenario_name] = []
                    recordings_by_scenario[scenario_name].append(rec)

                with tqdm(total=total_samples_target, desc=f"Processing {split} (samples)") as pbar:
                    for scenario_name, recs_in_scenario in recordings_by_scenario.items():
                        target_samples_for_scenario = self.config.num_samples_per_scenario.get(scenario_name)

                        random.shuffle(recs_in_scenario)

                        for recording in recs_in_scenario:
                            if target_samples_for_scenario is not None and self.scenario_num_samples_dict.get(scenario_name, 0) >= target_samples_for_scenario:
                                break
                            
                            num_samples = self._process_recording(recording, target_samples_for_scenario)
                            pbar.update(num_samples)
                            total_samples_by_split[split] += num_samples

        print(f'Num samples extracted from each scenario: {self.scenario_num_samples_dict}')

        if self.save:
            for lmdb_helper in self.lmdbs.values():
                lmdb_helper.write_meta_data()

    def _process_recording(self, recording: Tuple[str, str, str], target_samples_for_scenario: int) -> int:
        scenario_name, recording_name, split = recording
        
        # counting number of samples in this recording
        num_samples_in_recording = 0

        # load metadata
        data_path = Path(self.dataset_path).joinpath(scenario_name)
        annotations_dir = os.path.join(data_path, "annotations", recording_name)
        annotations = load_json_or_parquet(
            os.path.join(annotations_dir, "annotations.parquet")
        )
        data_meta = load_json_or_parquet(os.path.join(data_path, "data_meta.json"))

        # load frames
        frames = load_json_or_parquet(
            os.path.join(annotations_dir, "frames.parquet")
        )
        framerate = data_meta["frame_rate"]
        timestamps = frames.frame_id.values / framerate
        annotations["timestamp"] = annotations.frame_id / framerate

        # load and prepare vectorized map
        map_path = Path(self.dataset_path).joinpath(scenario_name)
        city_nodes_features = self.map_parser.get_centerlines_graph_features(map_path)
        city_centerlines_graph = CityCenterlinesGraph(nodes_features = city_nodes_features,
                                                    edges_indices = torch.zeros(2,1))

        # process road user information
        road_users = {}
        all_track_ids = annotations.track_id.unique()
        for track_id in tqdm(all_track_ids, desc="Processing road users", leave=False):
            road_users[track_id] = self._extract_road_user_from_ann(
                annotations, track_id)
            
        # create the frames_past mask to differntiate between input and output trajectories
        input_mask = np.zeros((self.config.frames_past + self.config.frames_future), dtype=bool)
        input_mask[:self.config.frames_past] = True

        # iterate over frames
        ts_iterator = self._get_iterator_over_timestamps(
            timestamps, scenario_name
        )
        stop_processing = False
        for ts_to_use in tqdm(ts_iterator, "Extracting samples", leave=False):
            if stop_processing:
                break
            # extract trajectories for current time step and split by category
            trajectories = self._extract_trajectories_for_ts(
                road_users, ts_to_use
            )
            targets_id = self.identify_target_agents(trajectories)

            for target_id in targets_id:
                if target_samples_for_scenario is not None and self.scenario_num_samples_dict.get(scenario_name, 0) >= target_samples_for_scenario:
                    stop_processing = True
                    break
                # create training sample
                training_sample = TrainingSample(
                    # dynamic context
                    input_mask=input_mask,
                    vehicle_trajectories=np.delete(trajectories["vehicle"].copy(), target_id, axis=0),
                    large_vehicle_trajectories=trajectories["large_vehicle"],
                    pedestrian_trajectories=trajectories["pedestrian"],
                    bicycle_trajectories=trajectories["bicycle"],
                    stationary_vehicles=trajectories["stationary_vehicle"],
                    ego_trajectory=np.zeros((0, len(ts_to_use), 9)),
                    target_trajectory=np.expand_dims(trajectories["vehicle"][target_id], axis=0),
                    # static context
                    stationary_objects=None,
                    vectorized_map=city_centerlines_graph,
                    # scenario_name=scenario_name,
                )

                # save training sample
                if self.save:
                    self.lmdbs[split].write_sample(training_sample)

                if self.visualize:
                    self.visualize_training_sample(scenario_name, training_sample, city_centerlines_graph)
        
                num_samples_in_recording += 1
                self.scenario_num_samples_dict[scenario_name] = self.scenario_num_samples_dict.get(scenario_name, 0) + 1
                
        return num_samples_in_recording
    

    def _get_all_recordings(self) -> List[Tuple[str, str]]:
        print("Collecting scenario and recording names in dataset...")
        scenarios_and_recordings = []
        scenarios = os.listdir(self.dataset_path)
        for scenario_name in scenarios:
            scenario_size = 0
            if scenario_name in self.config.scenarios:
                data_path = Path(self.dataset_path).joinpath(scenario_name).joinpath("annotations")
                recordings = os.listdir(data_path)
                for recording in recordings:
                    recording_size = 0
                    for dirpath, dirnames, filenames in os.walk(data_path.joinpath(f"{recording}")):
                        for f in filenames:
                            fp = os.path.join(dirpath, f)
                            # skip if it is symbolic link
                            if not os.path.islink(fp):
                                recording_size += os.path.getsize(fp)
                    scenario_size += recording_size
                    scenarios_and_recordings.append((scenario_name, recording))
                    if scenario_size > self.config.max_scenario_size:
                        break
        return scenarios_and_recordings

    def _add_split_to_recs(
        self, recordings: List[Tuple[str, str]]
    ) -> List[Tuple[str, str, str]]:
        train_scenario = self.config.train_scenario
        train_val_scenario = self.config.train_val_scenario
        val_scenario = self.config.val_scenario
        num_train_val_recordings = 0
        num_val_recordings = 0
        num_train_recordings = 0
        for i in range(len(recordings)):
            scenario_name = recordings[i][0]
            if scenario_name in train_val_scenario:
                num_train_val_recordings += 1
                recordings[i] += ("train_val",)
            elif scenario_name in val_scenario:
                num_val_recordings += 1
                recordings[i] += (self.val_split_map[scenario_name],)
            elif scenario_name in train_scenario:
                num_train_recordings += 1
                recordings[i] += ("train",)
        
        if len(train_val_scenario) > 0:
            assert num_train_val_recordings > 0, 'train_val scenario is not in scenarios'

        print( f'Number of train recordings: {num_train_recordings}')
        print( f'Number of train_val recordings: {num_train_val_recordings}')
        if num_train_recordings > 0:
            print( f'Ratio of train_val to train recordings: ' + str(round(num_train_val_recordings/num_train_recordings, 2)))

        return recordings

    def _extract_road_user_from_ann(self, annotations, track_id: int) -> DSRoadUser:
        track_data = annotations.loc[annotations['track_id'] == track_id]

        category = track_data.category.iloc[0]
        timestamps = track_data.timestamp.values

        pos = np.stack([
            track_data.translation_x.values,
            track_data.translation_y.values
        ]).T
        velocity = np.stack([
            track_data.velocity_x.values,
            track_data.velocity_y.values
        ]).T
        bbox_size = np.stack([
            track_data.dimension_x.values,
            track_data.dimension_y.values
        ]).T

        # heading in DS data is in radians, zero points along x axis
        # src_heading = track_data.rotation_z.values is close, but not equal
        # below is the method used in devkit
        rotation = np.array([
            track_data.rotation_x.values,
            track_data.rotation_y.values,
            track_data.rotation_z.values
        ]).T
        src_heading = Rotation.from_rotvec(rotation).as_euler('zxy')[:, 0]
        heading = process_heading(
            heading=src_heading,
            curr_heading=HeadingFormat(unit="rad", zero="x"),
            target_heading=self.heading_format,
        )

        road_user = DSRoadUser(
            track_id=track_id,
            category=category,
            timestamps=timestamps,
            pos=pos,
            velocity=velocity,
            heading=heading,
            bbox_size=bbox_size,
        )

        road_user.postprocess()

        return road_user

    def _get_iterator_over_timestamps(
        self, all_timestamps: List[float], scenario_name: str
    ) -> Iterable:
        """
        Iterate over possible current timestamps:
            (0, sliding_window_step, 2 * sliding_window_step, ...).
        Then for each get a list of timestamps, that will be used for that current timestamp:
            (curr_ts, ..., curr_ts + sliding_window_s).
        @param all_timestamps: list of (relative) timestamps in seconds
        @return:
        """
        sliding_window_s = self.config.sliding_window_s
        
        starting_ts = np.arange(
            all_timestamps[0],
            all_timestamps[-1] - self.config.sample_duration_in_s,
            sliding_window_s
        )
        # shuffle to get random samples
        np.random.shuffle(starting_ts)

        ts_iterator = [
            np.linspace(
                start_ts,
                start_ts + self.config.sample_duration_in_s,
                self.config.num_time_steps,
            ) for start_ts in starting_ts
        ]
        return ts_iterator

    def _extract_trajectories_for_ts(
        self,
        road_users: Dict[int, DSRoadUser],
        ts_to_use: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        # categories:
        category_to_type = {
            "car": "vehicle",
            "person": "pedestrian",
            "bicycle": "bicycle",
            "trailer": "large_vehicle",
            "motorcycle": "bicycle",
            "truck": "large_vehicle",
            "bus": "large_vehicle",
            "scooter": "bicycle",
            "train": "large_vehicle",
            "animal": "pedestrian",
            "bicycle_rack": "stationary_vehicle",
            "movable_object": "stationary_vehicle",
            "agricultural_vehicle": "large_vehicle",
            "construction_vehicle": "large_vehicle",
            "pickup": "vehicle",
            "van": "large_vehicle",
            "stationary_vehicle": "stationary_vehicle",
        }

        trajectories = {
            "vehicle": [],
            "large_vehicle": [],
            "pedestrian": [],
            "bicycle": [],
            "stationary_vehicle": [],
        }
        for track_id, road_user in road_users.items():
            trajectory = road_user.to_trajectory(ts_to_use)
            if not np.any(trajectory[:self.config.frames_past]): # skip trajectories with no input time steps
                continue

            trajectory[trajectory==0] = np.nan # set the zero values in trajectories to inf
            road_user_type = category_to_type[road_user.category]
            trajectories[road_user_type].append(trajectory)

        for road_user_type in trajectories.keys():
            if len(trajectories[road_user_type]) > 0:
                trajectories[road_user_type] = np.stack(
                    trajectories[road_user_type], axis=0
                )
            else:
                trajectories[road_user_type] = np.zeros((0, len(ts_to_use), 9))

        return trajectories
    
    def identify_target_agents(self, trajectories: np.ndarray):
        targets_id = []
        for id, track in enumerate(trajectories['vehicle']): # select only from target vehicles
            if (not np.isnan(track[self.config.frames_past : self.config.frames_past+self.config.target_min_future_frames, :2]).any() and # min output frames
                not np.isnan(track[self.config.frames_past-self.config.target_min_past_frames : self.config.frames_past]).any() and # min input frames 
                np.linalg.norm(track[self.config.frames_past-1, :2] - track[self.config.frames_past+self.config.target_min_future_frames-1, :2]) > self.config.target_min_travel): # min distance travel
                targets_id.append(id)
        return targets_id
            
    def visualize_training_sample(self, scenario_name: str, train_sample: TrainingSample, 
                                  city_centerlines_subgraph:CityCenterlinesGraph) -> None:

        road_user_types = ["vehicle_trajectories", "stationary_vehicles", "bicycle_trajectories",
                           'large_vehicle_trajectories', 'target_trajectory', 'ego_trajectory',
                           'pedestrian_trajectories']
        colors = ['blue', 'black', 'green', 'gray', 'red', 'yellow', 'orange']
        input_mask = train_sample.input_mask

        plt.figure(figsize=(10,10))
        for num, road_user_type in enumerate(road_user_types):
            trajectories = getattr(train_sample, road_user_type)
            if trajectories is not None:
                for track_num, track in enumerate(trajectories):
                    ind = track[:, 0] != 0.0
                    ind_input = np.logical_and(ind, input_mask)
                    ind_output = np.logical_and(ind, ~input_mask)
                    plt.plot(track[ind_input, 0], track[ind_input, 1], c=colors[num], marker='s', markersize=3, linewidth=2,
                                label=road_user_type)
                    plt.plot(track[ind_output, 0], track[ind_output, 1], c=colors[num], marker='x', markersize=3, linewidth=2, 
                             linestyle='dashed', label=road_user_type)
        
        # visualize city graph
        nodes_start_locs = city_centerlines_subgraph.nodes_features.start_locs
        nodes_end_locs = city_centerlines_subgraph.nodes_features.end_locs
        nodes_vector = nodes_end_locs - nodes_start_locs
        for i in range(len(nodes_vector)): # best for resolution=2
            plt.arrow(nodes_start_locs[i,0], nodes_start_locs[i,1], 
                    nodes_vector[i,0], nodes_vector[i,1], width=0.2, length_includes_head=True)
        
        # removing multiple legends
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        for handle in by_label.values():
            handle.set_alpha(0.8)
        plt.legend(by_label.values(), by_label.keys(), fontsize="10", loc='upper left')   
        
        plt.axis('equal')
        plt.xlabel('x-coodinate in Global Coordinate System')
        plt.ylabel('y-coodinate in Global Coordinate System')
        plt.title(scenario_name)
        plt.show()
        plt.close('all')


if __name__ == "__main__":
    preprocessing_cfg = DSPreprocessingConfig()
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.interpolate")
    sample_creator = DSSampleCreator(
        dataset_path=preprocessing_cfg.source_data_path,
        save_path=TARGET_PATH_DS,
        preprocessing_config=preprocessing_cfg,
        visualize=False,
        save=True,
    )

    sample_creator.parse_dataset()