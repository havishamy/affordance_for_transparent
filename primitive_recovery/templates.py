from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BeakerParams:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    radius: float
    height: float

    def to_vector(self) -> list[float]:
        return [self.x, self.y, self.z, self.roll, self.pitch, self.yaw, self.radius, self.height]

    @classmethod
    def from_vector(cls, values: list[float] | tuple[float, ...]) -> "BeakerParams":
        x, y, z, roll, pitch, yaw, radius, height = values
        return cls(
            x=float(x),
            y=float(y),
            z=float(z),
            roll=float(roll),
            pitch=float(pitch),
            yaw=float(yaw),
            radius=float(radius),
            height=float(height),
        )


def clip_beaker_params(
    params: BeakerParams,
) -> BeakerParams:
    params.z = max(params.z, 10.0)
    params.radius = max(params.radius, 1.0)
    params.height = max(params.height, 2.0)
    return params
