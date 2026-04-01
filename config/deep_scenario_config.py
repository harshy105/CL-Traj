import torch

from typing import Tuple, Optional
from config.config import DEEP_SCENARIO_PATH
from dataclasses import dataclass, field


@dataclass
class DSPreprocessingConfig:
    resolution: float = 3.0
    source_data_path: str = DEEP_SCENARIO_PATH
    sliding_window_s: float = 6.0
    frame_rate: float = 2.0
    past_secs: float = 2.0 # keeping it 2 secs to not include trajectories which are only non zero in [-3,-2) secs
    future_secs: float = 6.0
    target_heading_format: list = ("deg", "-y")
    target_min_gt_secs: float = 6.0
    target_min_past_secs: float = 2.0
    target_min_travel: float = 4.0
    split_names: list = ("train", "train_val", "val") # keep the names in order, change code if 'test' split is added
    max_scenario_size = 1e+9
    # split_ratios: list = (0.8, 0.2)
    train_scenario_config: dict = field(default_factory=lambda: {
        "Robust Munich": 6000,
        "Glamorous Kronach": 4000,
        "Loving Berlin": 7000,
        "Enthusiastic Ingolstadt": 2000,
        "Positive Wuppertal": 6200,
        "Noble Munich": 7000,
        "Special Munich": 1000,
        "Fortunate Karlsruhe": 300,
        "Visionary San Francisco": 500,
    })
    train_val_scenario_config: dict = field(default_factory=lambda: {
        "Exciting Munich": 1000,
        "Epic Munich": 3000,
        "Trendy Renningen": 500,
    })
    val_scenario_config: dict = field(default_factory=lambda: {
        "Stunning Stuttgart": 1000,
        "Diverse Kronach": 1000,
        "Unparalleled Frankfurt": 1000,
        "Busy Frankfurt": 2000,
        "Euphoric Wuppertal": 1000,
    })

    @property
    def train_scenario(self) -> list:
        return list(self.train_scenario_config.keys())

    @property
    def train_val_scenario(self) -> list:
        return list(self.train_val_scenario_config.keys())

    @property
    def val_scenario(self) -> list:
        return list(self.val_scenario_config.keys())

    @property
    def scenarios(self) -> list:
        return self.train_scenario + self.train_val_scenario + self.val_scenario

    @property
    def num_samples_per_scenario(self) -> dict:
        all_scenarios = {
            **self.train_scenario_config,
            **self.train_val_scenario_config,
            **self.val_scenario_config
        }
        return {k: v for k, v in all_scenarios.items() if v is not None}

    @property
    def frames_past(self) -> int:
        out = self.past_secs * self.frame_rate + 1
        assert out%1 == 0.0, 'num of past frame must be integer'
        return int(out)
    
    @property
    def frames_future(self) -> int:
        out = self.future_secs * self.frame_rate
        assert out%1 == 0.0, 'num of future frame must be integer'
        return int(out)
    
    @property
    def num_time_steps(self) -> int:
        return self.frames_past + self.frames_future

    @property
    def sample_duration_in_s(self) -> float:
        return self.past_secs + self.future_secs
    
    @property
    def target_min_future_frames(self) -> int:
        out = self.target_min_gt_secs * self.frame_rate
        assert out%1 == 0.0, 'num of target future frame must be integer'
        return int(out)
    
    @property
    def target_min_past_frames(self) -> int:
        out = self.target_min_past_secs * self.frame_rate + 1
        assert out%1 == 0.0, 'num of target future frame must be integer'
        return int(out)

    
@dataclass
class CityNodesFeatures:
    nodes_id: torch.Tensor
    start_locs: torch.Tensor
    end_locs: torch.Tensor
    incoming_lanes_nodes: Optional[torch.Tensor] = None
    outgoing_lanes_nodes: Optional[torch.Tensor] = None

@dataclass
class CityCenterlinesGraph:
    nodes_features: CityNodesFeatures
    edges_indices: torch.Tensor