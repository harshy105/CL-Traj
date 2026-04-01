import numpy as np
from typing import Literal, Union, List
import json
import pandas
from dataclasses import dataclass


@dataclass
class HeadingFormat:
    unit: Literal["deg", "rad"]  # units of heading
    zero: Literal["x", "-x", "y", "-y"]  # direction of zero angle

    @classmethod
    def from_list(cls, a: List):
        assert len(a) == 2
        assert a[0] in ("deg", "rad")
        assert a[1] in ("x", "-x", "y", "-y")
        return cls(unit=a[0], zero=a[1])

    def as_list(self):
        return [self.unit, self.zero]


def process_heading(
    heading: Union[float, np.ndarray],
    curr_heading: HeadingFormat,
    target_heading: HeadingFormat,
) -> float | np.ndarray:
    if curr_heading.unit == "deg" and target_heading.unit == "rad":
        heading = np.deg2rad(heading)
    elif curr_heading.unit == "rad" and target_heading.unit == "deg":
        heading = np.rad2deg(heading)

    if (
        "-" in curr_heading.zero
        and "-" not in target_heading.zero
        or "-" not in curr_heading.zero
        and "-" in target_heading.zero
    ):
        heading = -heading

    rotation_angle = 0
    if "x" in curr_heading.zero and "y" in target_heading.zero:
        rotation_angle = 90
    elif "y" in curr_heading.zero and "x" in target_heading.zero:
        rotation_angle = -90
    if target_heading.unit == "rad":
        rotation_angle = np.deg2rad(rotation_angle)
    heading += rotation_angle

    return heading


def load_json_or_parquet(file_path: str):
    if file_path.endswith(".json"):
        with open(file_path) as file:
            return json.load(file)
    elif file_path.endswith(".parquet"):
        return pandas.read_parquet(file_path)
