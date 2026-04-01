import torch
import numpy as np

from torch import Tensor
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

@dataclass
class NuScenesPreprocessConfig:
    resolution: float = 3.0
    region_size: Tuple = (300, 300)
    translation: Tuple = (0, 0)
    frames_past: int = 5
    frames_future: int = 12

@dataclass
class CityNodesFeatures:
    nodes_id: Tensor
    start_locs: Tensor
    end_locs: Tensor
    incoming_lanes_nodes: Optional[Tensor] = None
    outgoing_lanes_nodes: Optional[Tensor] = None

@dataclass
class CityCenterlinesGraph:
    nodes_features: CityNodesFeatures
    edges_indices: Tensor

@dataclass
class SceneInfo:
    name:        str
    description: str
    location:    str
    num_samples: int

@dataclass
class ObjectSampleNuscenes:
    frame:               int
    position:            np.ndarray
    velocity:            np.ndarray
    acceleration:        np.ndarray
    yaw:                 float
    heading_change_rate: float
    box_size:            np.ndarray
    category_name:       str
    attribute:           Optional[str]
    visibility:          int
    instance_token:      str
    sample_token:        str
    timestamp:           int

@dataclass
class RoadObjectSampleNuscenes:
    frame:          int
    position:       np.ndarray
    yaw:            float
    box_size:       np.ndarray
    category_name:  str
    visibility:     int
    instance_token: str
    sample_token:   str
    timestamp:      int

@dataclass
class ExtractedScene:
    scene_info:                 SceneInfo
    ego_trajectory:             Dict[int, ObjectSampleNuscenes]
    pedestrian_trajectories:    Dict[str, Dict[int, ObjectSampleNuscenes]]
    vehicle_trajectories:       Dict[str, Dict[int, ObjectSampleNuscenes]]
    bicycle_trajectories:       Dict[str, Dict[int, ObjectSampleNuscenes]]
    movable_objects:            Dict[str, Dict[int, RoadObjectSampleNuscenes]]
    static_objects:             Dict[str, Dict[int, RoadObjectSampleNuscenes]]


@dataclass
class NuscenesConfig:
    # dynamic object types
    pedestrian_types: List[str] = (
        'human.pedestrian.adult', 'human.pedestrian.child',
        'human.pedestrian.construction_worker', 'human.pedestrian.police_officer',
        'human.pedestrian.wheelchair', 'human.pedestrian.stroller',
        'human.pedestrian.personal_mobility')
    vehicle_types: List[str] = (
        'vehicle.car', 'vehicle.emergency.ambulance', 'vehicle.emergency.police',
        'vehicle.motorcycle', 'vehicle.bus.bendy', 'vehicle.construction', 'vehicle.trailer',
        'vehicle.bus.rigid', 'vehicle.truck')
    bicycle_types: List[str] = ('vehicle.bicycle')

    # static object types
    static_object_types: List[str] = ('static_object.bicycle_rack', 'static.vegetation')
    movable_object_types: List[str] = (
        'movable_object.barrier', 'movable_object.debris',
        'movable_object.pushable_pullable', 'movable_object.trafficcone')
    # Pedestrian and vehicle attributes are mapped to a scalar
    pedestrian_attributes: List[str] = ('pedestrian.sitting_lying_down', 'pedestrian.standing',
                                        'pedestrian.moving', 'cycle.with_rider')
    vehicle_attributes: List[str] = ('vehicle.moving', 'vehicle.stopped', 'vehicle.parked',
                                     'cycle.with_rider', 'cycle.without_rider')