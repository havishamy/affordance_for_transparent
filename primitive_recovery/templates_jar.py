from __future__ import annotations

from dataclasses import dataclass


@dataclass
class JarParams:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    body_radius: float
    body_height: float
    lid_radius: float
    lid_height: float

    def to_vector(self) -> list[float]:
        return [
            self.x,
            self.y,
            self.z,
            self.roll,
            self.pitch,
            self.yaw,
            self.body_radius,
            self.body_height,
            self.lid_radius,
            self.lid_height,
        ]

    @classmethod
    def from_vector(cls, values: list[float] | tuple[float, ...]) -> "JarParams":
        x, y, z, roll, pitch, yaw, body_radius, body_height, lid_radius, lid_height = values
        return cls(
            x=float(x),
            y=float(y),
            z=float(z),
            roll=float(roll),
            pitch=float(pitch),
            yaw=float(yaw),
            body_radius=float(body_radius),
            body_height=float(body_height),
            lid_radius=float(lid_radius),
            lid_height=float(lid_height),
        )


def clip_jar_params(params: JarParams) -> JarParams:
    params.z = max(params.z, 10.0)
    params.body_radius = max(params.body_radius, 1.0)
    params.body_height = max(params.body_height, 2.0)
    params.lid_radius = max(params.lid_radius, 1.0)
    params.lid_height = max(params.lid_height, 1.0)
    return params
