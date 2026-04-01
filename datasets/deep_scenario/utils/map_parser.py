import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import matplotlib.pyplot as plt

from datasets.deep_scenario.utils.map_utils import HeadingFormat, process_heading
from config.deep_scenario_config import DSPreprocessingConfig, CityCenterlinesGraph, CityNodesFeatures

# import DiscretizedMap inside func to avoid unnecessary requirements for caspnet training
# from datasets.deep_scenario.devkit.utils.map import DiscretizedMap


class DSMapParser:
    """
    Class to both parse data as vectorized map (data preprocessing)
     and rasterize the vectorized map (data generation).
    """

    def __init__(
        self,
        output_heading_format: HeadingFormat,
        resolution: float = None,
        region_size: np.ndarray = None,
    ):
        self.resolution = resolution
        self.region_size = region_size
        self.source_heading_format = HeadingFormat(
            unit="deg", zero="y"
        )  # format of DS source data
        self.output_heading_format = output_heading_format  # format of data generator

        # self.lane_types = {
        #     "driving": 0,
        #     "bidirectional": 1,
        #     "median": 2,
        #     "restricted": 3,
        #     "sidewalk": 4,
        #     "biking": 5,
        #     "parking": 6,
        #     "other": 7,
        # }

    def get_centerlines_graph_features(self, map_path: Path) -> CityNodesFeatures:
        from datasets.deep_scenario.devkit.utils.map import DiscretizedMap

        map_file = os.path.join(map_path, "map.xodr")
        discrete_map = DiscretizedMap.load_from_file(map_file, discretization=DSPreprocessingConfig.resolution)

        city_nodes_features = CityNodesFeatures(
            nodes_id = torch.zeros((1), dtype=torch.int32),
            start_locs = torch.zeros((1,2), dtype=torch.float),
            end_locs = torch.zeros((1,2), dtype=torch.float)
        )
        num_nodes = 0
        for road in discrete_map.roads:
            for lane_sec in road.lane_sections:
                for lane in lane_sec.lanes:
                    if lane.lane_type == 'driving':
                        # compute the node features
                        points_locs = torch.tensor(lane.center_vertices[:, :2], dtype=torch.float)[:,:2]
                        start_locs = points_locs[:-1] # (N, 2)
                        end_locs = points_locs[1:] # # (N, 2)
                        nodes_id = torch.arange(num_nodes, num_nodes+len(start_locs), dtype=torch.int32)

                        # append the centerline nodes features to city nodes 
                        city_nodes_features.nodes_id = torch.concat((city_nodes_features.nodes_id, nodes_id), dim=0)
                        city_nodes_features.start_locs = torch.concat((city_nodes_features.start_locs, start_locs), dim=0)
                        city_nodes_features.end_locs = torch.concat((city_nodes_features.end_locs, end_locs), dim=0)

                        num_nodes += len(start_locs)
        # remove the init zero features
        city_nodes_features.nodes_id = city_nodes_features.nodes_id[1:]
        city_nodes_features.start_locs = city_nodes_features.start_locs[1:]
        city_nodes_features.end_locs = city_nodes_features.end_locs[1:]

        return city_nodes_features

    @staticmethod
    def visualize_city_centerlines_graph(city_name: str, city_nodes_features: CityNodesFeatures) -> None:
        nodes_id = city_nodes_features.nodes_id
        nodes_start_locs = city_nodes_features.start_locs
        nodes_end_locs = city_nodes_features.end_locs
        nodes_vector = nodes_end_locs - nodes_start_locs
        for i in range(len(nodes_vector)): # best for resolution=2
            plt.arrow(nodes_start_locs[i,0], nodes_start_locs[i,1], 
                    nodes_vector[i,0], nodes_vector[i,1], width=0.2, length_includes_head=True)
        plt.axis('square')
        plt.title(f'{city_name}')
        plt.show()
        plt.close('all')

if __name__ == '__main__':
    from config.config import DEEP_SCENARIO_PATH
    preprocessing_cfg = DSPreprocessingConfig()
    heading_format = HeadingFormat.from_list(("deg", "-y"))
    city_name = 'Busy Frankfurt'
    map_parser = DSMapParser(output_heading_format=heading_format)
    map_path = Path(DEEP_SCENARIO_PATH).joinpath(city_name)
    city_nodes_features = map_parser.get_centerlines_graph_features(map_path)
    city_centerlines_graph = CityCenterlinesGraph(nodes_features = city_nodes_features,
                                                    edges_indices = None)
    map_parser.visualize_city_centerlines_graph(city_name, city_nodes_features)