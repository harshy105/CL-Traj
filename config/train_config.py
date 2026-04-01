import numpy as np

from dataclasses import dataclass
from typing import Tuple, Optional

from config.nuscenes_config import CityCenterlinesGraph
from config.config import TARGET_PATH, TARGET_PATH_DS, DATA_SET_NAME

@dataclass
class TrainingSample:
    # dynamic context
    input_mask:                 np.ndarray  # True for past time frames
    ego_trajectory:             np.ndarray  # pos_x, pos_y, v_x, v_y, yaw_deg, acc_x, acc_y, box_width, box_length # shape(1, T, 9)
    target_trajectory:          np.ndarray
    vehicle_trajectories:       np.ndarray
    large_vehicle_trajectories: np.ndarray
    pedestrian_trajectories:    np.ndarray
    bicycle_trajectories:       np.ndarray

    # static context
    stationary_vehicles:        np.ndarray
    stationary_objects:         np.ndarray
    vectorized_map:             CityCenterlinesGraph

@dataclass
class AgentTypeIds:
    target_trajectory: float = 0.0
    ego_trajectory: float = 1.0
    vehicle_trajectories: float = 1.0
    large_vehicle_trajectories: float = 2.0
    bicycle_trajectories: float = 3.0
    pedestrian_trajectories: float = 4.0

    def __post_init__(self):
        max_id = 0
        for v in self.__dict__.values():
            if max_id < v:
                max_id = v
        for k, v in self.__dict__.items():
            setattr(self, k, v/max_id)

    def inv_map(self):
        max_id = 0
        for v in self.__dict__.values():
            if max_id < v:
                max_id = v
        return({str(float(v*max_id)): k for k, v in self.__dict__.items()})

        
@dataclass
class DataStructureConfig:
    db_folder: str = TARGET_PATH # TARGET_PATH_DS or TARGET_PATH
    agent_type_ids: AgentTypeIds = AgentTypeIds() 
    sample_frequency: int = 2
    current_frame: int = 2
    num_future_steps: int = 12
    static_region_size: Tuple = (150, 100) # (h, w) along (x, y)
    static_translation: Tuple = (-50, 0) # (dx, dy)
    dynamic_region_size: Tuple = (150, 100) # (h, w) along (x, y)
    dynamic_translation: Tuple = (-50, 0) # (dx, dy)
    x_flip_prob: float = 0.5
    surr_agent_to_sim_prob: float = 0.5
    max_goal_offset = 12
    target_goal: bool = False
    surr_goal: bool = True
    max_num_agents: int = int(1e3)
    max_num_lane_nodes: int = int(1e3)
    vel_norm_factor: float = 1.0
    rel_dis_norm_factor: float = 1.0
    
    def __post_init__(self):
        assert DATA_SET_NAME in self.db_folder, "Use the correct database folder for the dataset"
    
@dataclass
class NetConfig:
    hidden_dim: int = 64
    num_layers: int = 4
    num_heads: int = 8
    head_dim: int = 8
    target_num_modes: int = 5
    scene_num_modes: int = 1
    max_num_input_frames: Optional[int] = 3
    num_dec_future_steps: int = 12
    output_dim: int = 2

@dataclass
class TrainingConfig:
    use_target_net: bool = True
    use_scene_net: bool = True
    diff_simulator: bool = False
    dropout: float = 0.3
    modes_noise_factor: float = 0.2
    head_regularization_scale: float = 0.05
    cls_loss_scale: float = 1.0
    reg_loss_scale: float = 0.4
    target_cl_loss_scale: float = 0.1
    scene_cl_loss_scale: float = 0.0
    num_train_recurr_steps: int = 3 
    num_eval_recurr_steps: int = 3 
    num_dec_recurr_steps: int = 2 # <= 6 (for heading computation)
    data_augmentation: bool = False
    weight_decay: float = 5e-5
    use_swa: bool = False
    pretrained_ckpt: Optional[str] = None # "****.ckpt"
    num_worker: int = 8
    max_epochs: int = 60

    def __post_init__(self):
        if self.use_swa:
            if self.pretrained_ckpt is None:
                self.learning_rate = 1e-3
                self.min_lr = 1e-4 
                self.min_lr_patience_epochs = 10 # early stopping (Not tested)
                self.swa_epoch_start = 30 
                self.swa_annealing_epochs = 5 
            else:
                raise ValueError('The SWA variables are not initialized')
        else:
            self.min_lr = 1e-5
            self.min_lr_patience_epochs = 5 # early stopping
            self.lr_factor_on_plateau = 0.1
            if self.pretrained_ckpt is None:
                self.learning_rate = 1e-3
                self.patience = 3
            else:
                self.learning_rate = 1e-4
                self.patience = 4
        
        self.target_loss_scale = 1.0 if self.use_target_net else 0.0
        self.scene_loss_scale = 1.0 if self.use_scene_net else 0.0
        self.dis_threshold = 40.0 if self.use_scene_net else 80.0
        self.batch_size = 16 if self.use_scene_net else 32
        if self.num_train_recurr_steps > 1:
            assert (self.target_cl_loss_scale * self.use_target_net) > 0.0 or \
                (self.scene_cl_loss_scale * self.use_scene_net) > 0.0
        else:
            assert self.target_cl_loss_scale == 0.0 and self.scene_cl_loss_scale == 0.0

@dataclass
class SampleOfInterest:
    sample_list: Tuple[int] = (
        111, 434, 453, 1194, 1196, 1523, 1688, 1948, 2402, 
        2633, 2706, 3735, 4088, 4580, 4582, 4700, 4702, 5264,
        5421, 5479, 5599, 5600, 6365, 6456, 6705, 6717, 7223, 
        7466, 7467, 7827, 7920, 8033, 8309
        )