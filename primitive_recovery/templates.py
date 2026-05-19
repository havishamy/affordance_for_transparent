from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BeakerParams:
    cx: float
    cy: float
    z: float
    radius: float
    height: float
    yaw: float = 0.0

    def to_vector(self) -> list[float]:
        return [self.cx, self.cy, self.z, self.radius, self.height, self.yaw]

    @classmethod
    def from_vector(cls, values: list[float] | tuple[float, ...]) -> "BeakerParams":
        cx, cy, z, radius, height, yaw = values
        return cls(cx=float(cx), cy=float(cy), z=float(z), radius=float(radius), height=float(height), yaw=float(yaw))


def clip_beaker_params(
    params: BeakerParams,
    image_width: int,
    image_height: int,
) -> BeakerParams:
    params.cx = min(max(params.cx, 0.0), float(image_width - 1))
    params.cy = min(max(params.cy, 0.0), float(image_height - 1))
    params.z = max(params.z, 1.0)
    params.radius = max(params.radius, 2.0)
    params.height = max(params.height, 4.0)
    return params

