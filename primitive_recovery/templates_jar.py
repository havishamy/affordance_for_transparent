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
    shoulder_height: float
    neck_radius: float
    neck_height: float
    lip_radius: float
    lip_height: float

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
            self.shoulder_height,
            self.neck_radius,
            self.neck_height,
            self.lip_radius,
            self.lip_height,
        ]

    @classmethod
    def from_vector(cls, values: list[float] | tuple[float, ...]) -> "JarParams":
        x, y, z, roll, pitch, yaw, body_radius, body_height, shoulder_height, neck_radius, neck_height, lip_radius, lip_height = values
        return cls(
            x=float(x),
            y=float(y),
            z=float(z),
            roll=float(roll),
            pitch=float(pitch),
            yaw=float(yaw),
            body_radius=float(body_radius),
            body_height=float(body_height),
            shoulder_height=float(shoulder_height),
            neck_radius=float(neck_radius),
            neck_height=float(neck_height),
            lip_radius=float(lip_radius),
            lip_height=float(lip_height),
        )


def clip_jar_params(params: JarParams) -> JarParams:
    params.z = max(params.z, 10.0)
    params.body_radius = max(params.body_radius, 1.0)
    params.body_height = max(params.body_height, 2.0)
    params.shoulder_height = max(params.shoulder_height, 1.0)
    params.neck_radius = max(min(params.neck_radius, params.body_radius * 0.85), 1.0)
    params.neck_height = max(params.neck_height, 1.0)
    params.lip_radius = max(min(params.lip_radius, params.body_radius * 0.95), params.neck_radius)
    params.lip_height = max(params.lip_height, 1.0)
    return params
