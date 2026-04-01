# Copyright 2022 DeepScenario GmbH

import math
from lxml import etree
import numpy as np
import open3d as o3d
from typing import List, Tuple
from tqdm import tqdm
import os

from opendrive2lanelet.opendriveparser import parser, elements
from opendrive2lanelet.plane_elements.plane_group import ParametricLaneGroup
from opendrive2lanelet.converter import OpenDriveConverter
from opendrive2lanelet.utils import decode_road_section_lane_width_id


ElevationRecords = List[elements.roadElevationProfile.ElevationRecord]


def load_map(map_file: str) -> elements.opendrive.OpenDrive:
    with open(map_file, "r") as file:
        return parser.parse_opendrive(etree.parse(file).getroot())


def get_discretization_number(length: float, discretization: float) -> int:
    return int(max(3, np.ceil(length / discretization)))


def is_forward(lane_id: int) -> bool:
    return lane_id < 0


def discretize_lane(
    parametric_lane_group: ParametricLaneGroup, s_values: np.ndarray, discretization: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute left and right vertices at specified s_values
    """
    # lane_let_format = parametric_lane_group.to_lanelet(precision=discretization)
    left_vertices = np.asarray(
        [parametric_lane_group.calc_border("inner", s_value)[0] for s_value in s_values]
    )
    right_vertices = np.asarray(
        [parametric_lane_group.calc_border("outer", s_value)[0] for s_value in s_values]
    )
    center_vertices = np.array(
            [(l + r) / 2 for (l, r) in zip(left_vertices, right_vertices)]
        )

    centerline_w_orientation = np.zeros((len(center_vertices), 3))
    centerline_w_orientation[:, :2] = center_vertices

    if len(center_vertices) > 1:
        for i in range(len(centerline_w_orientation) - 1):
            centerline_w_orientation[i, 2] = -math.atan2(
                centerline_w_orientation[i + 1, 0] - centerline_w_orientation[i, 0],
                centerline_w_orientation[i + 1, 1] - centerline_w_orientation[i, 1],
            )
        centerline_w_orientation[-1, 2] = centerline_w_orientation[-2, 2]
        centerline_w_orientation[:, 2] = np.rad2deg(centerline_w_orientation[:, 2])

    # opendrive2lanelet gives vertices of left lanes in reverse order
    _, _, lane_id, _ = decode_road_section_lane_width_id(parametric_lane_group.id_)
    if not is_forward(lane_id):
        left_vertices = np.flipud(left_vertices)
        right_vertices = np.flipud(right_vertices)

    return left_vertices, right_vertices, centerline_w_orientation


def discretize_elevations(
    elevation_records: ElevationRecords, s_values: np.ndarray
) -> np.ndarray:
    """
    Compute elevation at specified s_values
    """
    return np.array(
        [compute_elevation(elevation_records, s_value) for s_value in s_values]
    )


def compute_elevation(elevation_records: ElevationRecords, s_value: float) -> float:
    for elevation_record in reversed(elevation_records):
        if elevation_record.start_pos <= s_value:
            curr_elevation_record = elevation_record
            break
    ds = s_value - curr_elevation_record.start_pos
    a, b, c, d = curr_elevation_record.polynomial_coefficients
    elevation = a + b * ds + c * ds**2 + d * ds**3
    return elevation


class DiscretizedRoad:
    def __init__(self, road: elements.road.Road) -> None:
        for superelevation in road.lateralProfile.superelevations:
            assert np.allclose(
                superelevation.polynomial_coefficients, 0
            ), "Superelevation is not implemented"
        for shape in road.lateralProfile.shapes:
            assert np.allclose(
                shape.polynomial_coefficients, 0
            ), "Shape is not implemented"

        self.id = road.id
        self.lane_sections: List[DiscretizedLaneSection] = []
        self.elevation_records = [
            elevation_tuple[0] for elevation_tuple in road.elevationProfile.elevations
        ]


class DiscretizedLaneSection:
    def __init__(self, lane_section: elements.roadLanes.LaneSection) -> None:
        self.id = lane_section.idx
        self.lanes: List[DiscretizedLane] = []


class DiscretizedLane:
    def __init__(
        self,
        elevation_records: ElevationRecords,
        lane_section: elements.roadLanes.LaneSection,
        parametric_lane_group: ParametricLaneGroup,
        discretization: float,
    ) -> None:
        _, _, self.id, _ = decode_road_section_lane_width_id(parametric_lane_group.id_)

        lane_length = parametric_lane_group.length
        s_values = np.linspace(
            0, lane_length, num=get_discretization_number(lane_length, discretization)
        )
        left_vertices_xy, right_vertices_xy, center_vertices = discretize_lane(
            parametric_lane_group, s_values, discretization
        )
        elevations = discretize_elevations(
            elevation_records, s_values + lane_section.sPos
        )
        self.left_vertices = np.column_stack((left_vertices_xy, elevations))
        self.right_vertices = np.column_stack((right_vertices_xy, elevations))
        self.polygon = np.row_stack((self.left_vertices, self.right_vertices[::-1]))
        self.center_vertices = center_vertices
        self.lane_type = parametric_lane_group.type


class DiscretizedMap:
    def __init__(self) -> None:
        self.roads: List[DiscretizedRoad] = []

    @classmethod
    def load_from_file(
        cls, map_file: str, discretization: float = 1.0
    ) -> "DiscretizedMap":
        map = load_map(map_file)
        map_discretized = cls()
        for road in map.roads:
            road_discretized = DiscretizedRoad(road)
            ref_border = OpenDriveConverter.create_reference_border(
                road.planView, road.lanes.laneOffsets
            )
            for lane_section in road.lanes.lane_sections:
                lane_section_discretized = DiscretizedLaneSection(lane_section)
                parametric_lane_groups = (
                    OpenDriveConverter.lane_section_to_parametric_lanes(
                        lane_section, ref_border
                    )
                )
                for parametric_lane_group in parametric_lane_groups:
                    lane_discretized = DiscretizedLane(
                        road_discretized.elevation_records,
                        lane_section,
                        parametric_lane_group,
                        discretization,
                    )
                    lane_section_discretized.lanes.append(lane_discretized)
                road_discretized.lane_sections.append(lane_section_discretized)
            map_discretized.roads.append(road_discretized)
        return map_discretized


def convert_line_to_line_set(line: np.ndarray, color: tuple) -> o3d.geometry.LineSet:
    edges = [[i, i + 1] for i in range(len(line) - 1)]
    colors = [color for _ in range(len(edges))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(line), lines=o3d.utility.Vector2iVector(edges)
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def get_lane_lines(
    map: DiscretizedMap, color: tuple = (0, 1, 0)
) -> List[o3d.geometry.LineSet]:
    lane_line_sets = []
    for road in map.roads:
        for lane_section in road.lane_sections:
            for lane in lane_section.lanes:
                lane_line_set = convert_line_to_line_set(lane.polygon, color=color)
                lane_line_sets.append(lane_line_set)
    return lane_line_sets


def load_static_environment(
    data_dir: str,
    data_meta: dict,
    visualize_pcd: bool,
    visualize_mesh: bool,
    visualize_map: bool,
) -> tuple:
    with tqdm(total=1, desc="Loading static environment") as progress_bar:
        try:
            pcd_file = os.path.join(data_dir, data_meta["point_cloud_file"])
        except KeyError:
            pcd_file = os.path.join(data_dir, "point_cloud/point_cloud.ply")
        pcd = o3d.io.read_point_cloud(pcd_file) if visualize_pcd else None

        try:
            mesh_file = os.path.join(data_dir, data_meta["textured_mesh_file"])
        except KeyError:
            mesh_file = os.path.join(data_dir, "textured_mesh/textured_mesh.obj")
        mesh = o3d.io.read_triangle_model(mesh_file) if visualize_mesh else None

        if "map_file" in data_meta:
            map_file = os.path.join(data_dir, data_meta["map_file"])
        else:
            map_file = os.path.join(data_dir, "map/map.xodr")
        map = DiscretizedMap.load_from_file(map_file) if visualize_map else None
        progress_bar.update(1)

    return pcd, mesh, map
