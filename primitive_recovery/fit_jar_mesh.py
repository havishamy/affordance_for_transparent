from __future__ import annotations

import argparse
import json
import struct
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize

from affordance.utils.image_ops import ensure_dir, load_mask
from primitive_recovery.geometry import (
    CameraIntrinsics,
    backproject_pixel_to_camera,
    bbox_from_mask,
    mask_centroid,
    normalize_vector,
    project_camera_to_pixel,
    rotation_matrix_from_y_axis,
    rotation_matrix_to_rpy,
    rpy_to_rotation_matrix,
)
from primitive_recovery.losses import (
    axis_alignment_loss,
    axis_angle_loss,
    bottom_alignment_loss,
    centroid_alignment_loss,
    contour_chamfer_loss,
    mask_iou_loss,
    robust_depth_loss,
    top_width_loss,
)
from primitive_recovery.table_plane import TablePlaneEstimate, estimate_table_plane_from_depth


@dataclass
class MeshJarParams:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    scale: float


@dataclass
class MeshJarInitializationResult:
    params: MeshJarParams
    table_plane: TablePlaneEstimate | None
    init_center_world: np.ndarray
    model_extent_local: np.ndarray
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a jar mesh with global scale + pose from mask/depth.")
    parser.add_argument("--ply", type=str, required=True, help="Path to object mesh in PLY format.")
    parser.add_argument("--mask", type=str, required=True, help="Path to binary object mask.")
    parser.add_argument("--depth", type=str, required=True, help="Path to depth image.")
    parser.add_argument("--rgb", type=str, default=None, help="Optional RGB image for visualization.")
    parser.add_argument("--output-dir", type=str, default="primitive_outputs/jar_fit_mesh")
    parser.add_argument("--fx", type=float, default=605.74)
    parser.add_argument("--fy", type=float, default=605.42)
    parser.add_argument("--cx", type=float, default=335.28)
    parser.add_argument("--cy", type=float, default=249.04)
    parser.add_argument("--maxiter", type=int, default=120)
    return parser.parse_args()


def load_depth(path: str | Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Unable to read depth image: {path}")
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth.astype(np.float32)


def load_rgb(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read RGB image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_ply_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    with path.open("rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Invalid PLY: missing end_header")
            decoded = line.decode("ascii", errors="ignore").strip()
            header.append(decoded)
            if decoded == "end_header":
                break

        if header[0] != "ply":
            raise ValueError("Invalid PLY: missing ply magic")
        if "format binary_little_endian 1.0" not in header:
            raise ValueError("Only binary_little_endian PLY is supported")

        vertex_count = None
        face_count = None
        for line in header:
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif line.startswith("element face"):
                face_count = int(line.split()[-1])
        if vertex_count is None or face_count is None:
            raise ValueError("PLY must contain vertex and face counts")

        vertices = np.fromfile(f, dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")]), count=vertex_count)
        verts = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float32)

        faces = []
        for _ in range(face_count):
            n = struct.unpack("<B", f.read(1))[0]
            if n != 3:
                idx = struct.unpack("<" + "i" * n, f.read(4 * n))
                for i in range(1, n - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])
            else:
                idx = struct.unpack("<iii", f.read(12))
                faces.append(list(idx))
    return verts, np.asarray(faces, dtype=np.int32)


def canonicalize_mesh(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    axis0 = normalize_vector(eigvecs[:, 0])
    axis1 = normalize_vector(eigvecs[:, 1])
    axis2 = normalize_vector(np.cross(axis0, axis1))
    projected = centered @ np.stack([axis0, axis1, axis2], axis=1)
    extents = projected.max(axis=0) - projected.min(axis=0)
    vertical_idx = int(np.argmax(extents))
    basis = [axis0, axis1, axis2]
    y_axis = basis[vertical_idx]
    remaining = [basis[i] for i in range(3) if i != vertical_idx]
    x_axis = remaining[0]
    x_axis = normalize_vector(x_axis - np.dot(x_axis, y_axis) * y_axis)
    z_axis = normalize_vector(np.cross(x_axis, y_axis))
    x_axis = normalize_vector(np.cross(y_axis, z_axis))
    rot = np.stack([x_axis, y_axis, z_axis], axis=1)
    canonical = centered @ rot
    return canonical.astype(np.float32), rot.astype(np.float32)


def transform_mesh_vertices(vertices_local: np.ndarray, params: MeshJarParams) -> np.ndarray:
    rot = rpy_to_rotation_matrix(params.roll, params.pitch, params.yaw)
    verts = (rot @ (vertices_local * params.scale).T).T
    verts += np.asarray([params.x, params.y, params.z], dtype=np.float32)
    return verts


def rasterize_mesh(
    vertices_world: np.ndarray,
    faces: np.ndarray,
    image_shape: tuple[int, int],
    intrinsics: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    depth = np.zeros((h, w), dtype=np.float32)

    projected = []
    valid_vertex = np.ones((vertices_world.shape[0],), dtype=bool)
    for i, p in enumerate(vertices_world):
        uv = project_camera_to_pixel(p, intrinsics)
        if uv is None:
            valid_vertex[i] = False
            projected.append((0.0, 0.0))
        else:
            projected.append(uv)
    projected = np.asarray(projected, dtype=np.float32)

    face_depths = vertices_world[faces][:, :, 2].mean(axis=1)
    order = np.argsort(face_depths)[::-1]
    for idx in order:
        face = faces[idx]
        if not np.all(valid_vertex[face]):
            continue
        pts = np.round(projected[face]).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        tri_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(tri_mask, pts, 1)
        if tri_mask.sum() == 0:
            continue
        z = float(face_depths[idx])
        update = (tri_mask > 0) & ((depth == 0) | (z < depth))
        depth[update] = z
        mask[tri_mask > 0] = 1
    return mask, depth


def render_mesh_mask(
    vertices_local: np.ndarray,
    faces: np.ndarray,
    params: MeshJarParams,
    image_shape: tuple[int, int],
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    verts_world = transform_mesh_vertices(vertices_local, params)
    mask, _ = rasterize_mesh(verts_world, faces, image_shape, intrinsics)
    return mask


def render_mesh_depth(
    vertices_local: np.ndarray,
    faces: np.ndarray,
    params: MeshJarParams,
    image_shape: tuple[int, int],
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    verts_world = transform_mesh_vertices(vertices_local, params)
    _, depth = rasterize_mesh(verts_world, faces, image_shape, intrinsics)
    return depth


def initialize_from_mesh_and_mask(
    vertices_local: np.ndarray,
    faces: np.ndarray,
    observed_mask: np.ndarray,
    observed_depth: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> MeshJarInitializationResult:
    center_uv = mask_centroid(observed_mask)
    z = float(np.median(observed_depth[(observed_mask > 0) & (observed_depth > 0)]))
    if not np.isfinite(z) or z <= 0:
        z = 1000.0
    center_world = backproject_pixel_to_camera(center_uv[0], center_uv[1], z, intrinsics)
    extent = vertices_local.max(axis=0) - vertices_local.min(axis=0)
    bbox = bbox_from_mask(observed_mask)
    x1, y1, x2, y2 = bbox
    mask_h = float(y2 - y1)
    scale = mask_h * z / max(float(intrinsics.fy), 1e-6) / max(float(extent[1]), 1e-6)

    table_plane = estimate_table_plane_from_depth(observed_mask, observed_depth, intrinsics)
    roll = 0.0
    pitch = 0.0
    yaw = 0.0
    source = "mask_depth_center"
    if table_plane is not None:
        target_axis = -table_plane.normal.astype(np.float32)
        if target_axis[1] > 0:
            target_axis = -target_axis
        rot = rotation_matrix_from_y_axis(target_axis)
        roll, pitch, yaw = rotation_matrix_to_rpy(rot)
        source = table_plane.source

    params = MeshJarParams(
        x=float(center_world[0]),
        y=float(center_world[1]),
        z=float(center_world[2]),
        roll=float(roll),
        pitch=float(pitch),
        yaw=float(yaw),
        scale=float(max(scale, 1e-3)),
    )
    rendered_mask = render_mesh_mask(vertices_local, faces, params, observed_mask.shape, intrinsics)
    rendered_centroid = mask_centroid(rendered_mask)
    delta_uv = center_uv - rendered_centroid
    center_world[0] += float(delta_uv[0]) * float(params.z) / max(float(intrinsics.fx), 1e-6)
    center_world[1] += float(delta_uv[1]) * float(params.z) / max(float(intrinsics.fy), 1e-6)
    params = MeshJarParams(
        x=float(center_world[0]),
        y=float(center_world[1]),
        z=float(center_world[2]),
        roll=float(roll),
        pitch=float(pitch),
        yaw=float(yaw),
        scale=float(max(scale, 1e-3)),
    )
    return MeshJarInitializationResult(
        params=params,
        table_plane=table_plane,
        init_center_world=center_world.astype(np.float32),
        model_extent_local=extent.astype(np.float32),
        source=source,
    )


def pack_optimization_vector(params: MeshJarParams) -> np.ndarray:
    return np.asarray([params.x, params.y, params.z, params.scale], dtype=np.float64)


def unpack_optimization_vector(theta: np.ndarray, fixed_pose: MeshJarParams) -> MeshJarParams:
    x, y, z, scale = theta.tolist()
    return MeshJarParams(
        x=float(x),
        y=float(y),
        z=max(float(z), 10.0),
        roll=float(fixed_pose.roll),
        pitch=float(fixed_pose.pitch),
        yaw=float(fixed_pose.yaw),
        scale=max(float(scale), 1e-3),
    )


def mesh_scale_prior(params: MeshJarParams) -> float:
    return 0.0 if params.scale > 0 else 1000.0


def make_objective(
    vertices_local: np.ndarray,
    faces: np.ndarray,
    observed_mask: np.ndarray,
    observed_depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    history: list[dict],
    fixed_pose: MeshJarParams,
    table_plane: TablePlaneEstimate | None = None,
) -> callable:
    image_shape = observed_mask.shape
    table_axis = None if table_plane is None else -table_plane.normal.astype(np.float32)

    def objective(theta: np.ndarray) -> float:
        params = unpack_optimization_vector(theta, fixed_pose)
        rendered_mask = render_mesh_mask(vertices_local, faces, params, image_shape, intrinsics)
        rendered_depth = render_mesh_depth(vertices_local, faces, params, image_shape, intrinsics)

        l_mask = mask_iou_loss(rendered_mask, observed_mask)
        l_depth = robust_depth_loss(rendered_depth, observed_depth, observed_mask)
        l_contour = contour_chamfer_loss(rendered_mask, observed_mask)
        l_axis = axis_angle_loss(rendered_mask, observed_mask)
        l_centroid = centroid_alignment_loss(rendered_mask, observed_mask)
        l_bottom = bottom_alignment_loss(rendered_mask, observed_mask)
        l_top = top_width_loss(rendered_mask, observed_mask)
        l_prior = mesh_scale_prior(params)
        l_table_axis = 0.0
        if table_plane is not None and table_axis is not None:
            l_table_axis = axis_alignment_loss(params.roll, params.pitch, params.yaw, table_axis)

        total = (
            6.0 * l_mask
            + 0.003 * l_depth
            + 0.8 * l_contour
            + 0.2 * l_axis
            + 0.08 * l_centroid
            + 0.12 * l_bottom
            + 0.5 * l_top
            + 0.1 * l_prior
            + 1.2 * l_table_axis
        )
        history.append(
            {
                "total": float(total),
                "mask": float(l_mask),
                "depth": float(l_depth),
                "contour": float(l_contour),
                "axis": float(l_axis),
                "centroid": float(l_centroid),
                "bottom": float(l_bottom),
                "top_width": float(l_top),
                "prior": float(l_prior),
                "table_axis": float(l_table_axis),
                "params": asdict(params),
                "pose_locked": True,
            }
        )
        return float(total)

    return objective


def overlay_masks(observed_mask: np.ndarray, rendered_mask: np.ndarray) -> np.ndarray:
    h, w = observed_mask.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[observed_mask > 0] = (0, 255, 0)
    canvas[rendered_mask > 0] = (255, 0, 0)
    overlap = (observed_mask > 0) & (rendered_mask > 0)
    canvas[overlap] = (255, 255, 0)
    return canvas


def save_depth_visual(path: Path, depth: np.ndarray, mask: np.ndarray | None = None) -> None:
    vis = depth.copy()
    if mask is not None:
        vis = vis * (mask > 0)
    valid = vis[vis > 0]
    out = np.zeros_like(vis, dtype=np.uint8)
    if valid.size > 0:
        lo, hi = float(valid.min()), float(valid.max())
        if hi > lo:
            out = np.clip((vis - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), out)


def estimate_plane_anchor_pixel(
    table_plane: TablePlaneEstimate,
    intrinsics: CameraIntrinsics,
) -> tuple[tuple[int, int], np.ndarray] | None:
    ys, xs = np.where(table_plane.inlier_mask > 0)
    if len(xs) == 0:
        return None
    u = float(np.median(xs))
    v = float(np.median(ys))
    denom = float(np.dot(table_plane.normal, np.asarray([(u - intrinsics.cx) / intrinsics.fx, (v - intrinsics.cy) / intrinsics.fy, 1.0], dtype=np.float32)))
    if abs(denom) < 1e-6:
        return None
    ray = np.asarray([(u - intrinsics.cx) / intrinsics.fx, (v - intrinsics.cy) / intrinsics.fy, 1.0], dtype=np.float32)
    t = -float(table_plane.offset) / denom
    if t <= 0:
        return None
    return (int(round(u)), int(round(v))), (ray * t).astype(np.float32)


def draw_table_normal_visualization(
    canvas_bgr: np.ndarray,
    table_plane: TablePlaneEstimate,
    intrinsics: CameraIntrinsics,
    arrow_len_mm: float = 80.0,
) -> tuple[np.ndarray, dict] | tuple[None, None]:
    anchor = estimate_plane_anchor_pixel(table_plane, intrinsics)
    if anchor is None:
        return None, None
    (u0, v0), anchor_world = anchor
    normal_world = table_plane.normal.astype(np.float32)
    axis_world = -normal_world
    end_normal_world = anchor_world + normal_world * float(arrow_len_mm)
    end_axis_world = anchor_world + axis_world * float(arrow_len_mm)
    end_normal_uv = project_camera_to_pixel(end_normal_world, intrinsics)
    end_axis_uv = project_camera_to_pixel(end_axis_world, intrinsics)
    if end_normal_uv is None or end_axis_uv is None:
        return None, None

    vis = canvas_bgr.copy()
    overlay = vis.copy()
    overlay[table_plane.ring_mask > 0] = (60, 60, 60)
    overlay[table_plane.inlier_mask > 0] = (0, 180, 255)
    vis = cv2.addWeighted(vis, 0.72, overlay, 0.28, 0.0)
    p0 = (int(round(u0)), int(round(v0)))
    p_normal = (int(round(end_normal_uv[0])), int(round(end_normal_uv[1])))
    p_axis = (int(round(end_axis_uv[0])), int(round(end_axis_uv[1])))
    cv2.circle(vis, p0, 5, (255, 255, 255), -1)
    cv2.arrowedLine(vis, p0, p_normal, (0, 0, 255), 3, tipLength=0.18)
    cv2.arrowedLine(vis, p0, p_axis, (0, 255, 0), 3, tipLength=0.18)
    return vis, {
        "anchor_pixel": [int(p0[0]), int(p0[1])],
        "normal_tip_pixel": [int(p_normal[0]), int(p_normal[1])],
        "axis_tip_pixel": [int(p_axis[0]), int(p_axis[1])],
        "arrow_length_mm": float(arrow_len_mm),
    }


def write_obj_with_mtl(path: Path, vertices: np.ndarray, faces: np.ndarray, material_name: str, color_rgb: tuple[float, float, float]) -> None:
    mtl_path = path.with_suffix(".mtl")
    mtl_path.write_text(
        "\n".join(
            [
                f"newmtl {material_name}",
                f"Kd {color_rgb[0]:.4f} {color_rgb[1]:.4f} {color_rgb[2]:.4f}",
                "Ka 0.1 0.1 0.1",
                "Ks 0.0 0.0 0.0",
                "d 1.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    lines = [f"mtllib {mtl_path.name}", f"usemtl {material_name}"]
    lines.extend([f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}" for v in vertices])
    lines.extend([f"f {f[0]+1} {f[1]+1} {f[2]+1}" for f in faces])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_scene_obj(
    out_dir: Path,
    jar_vertices_world: np.ndarray,
    jar_faces: np.ndarray,
    table_plane: TablePlaneEstimate | None,
) -> None:
    vertices = []
    faces = []
    materials = []
    offset = 0

    def add_mesh(v: np.ndarray, f: np.ndarray, name: str, color: tuple[float, float, float]) -> None:
        nonlocal offset
        materials.append((name, color, offset, offset + len(v), len(f)))
        vertices.extend(v.tolist())
        faces.extend((f + offset).tolist())
        offset += len(v)

    add_mesh(jar_vertices_world.astype(np.float32), jar_faces.astype(np.int32), "jar_mesh", (0.15, 0.65, 0.95))

    if table_plane is not None:
        n = table_plane.normal.astype(np.float32)
        center = -table_plane.offset * n
        x_hint = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        x_axis = normalize_vector(x_hint - np.dot(x_hint, n) * n)
        if np.linalg.norm(x_axis) < 1e-5:
            x_axis = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
            x_axis = normalize_vector(x_axis - np.dot(x_axis, n) * n)
        z_axis = normalize_vector(np.cross(x_axis, n))
        plane_size = 220.0
        corners = np.asarray(
            [
                center + (-x_axis - z_axis) * plane_size,
                center + (x_axis - z_axis) * plane_size,
                center + (x_axis + z_axis) * plane_size,
                center + (-x_axis + z_axis) * plane_size,
            ],
            dtype=np.float32,
        )
        plane_faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        add_mesh(corners, plane_faces, "table_plane", (0.75, 0.75, 0.75))

    # Camera marker.
    cam_vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [-25.0, -15.0, 40.0],
            [25.0, -15.0, 40.0],
            [25.0, 15.0, 40.0],
            [-25.0, 15.0, 40.0],
        ],
        dtype=np.float32,
    )
    cam_faces = np.asarray([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], [1, 2, 3], [1, 3, 4]], dtype=np.int32)
    add_mesh(cam_vertices, cam_faces, "camera", (0.9, 0.25, 0.25))

    scene_path = out_dir / "scene_3d.obj"
    scene_mtl_path = out_dir / "scene_3d.mtl"
    mtl_lines = []
    obj_lines = [f"mtllib {scene_mtl_path.name}"]
    for name, color, start, end, _ in materials:
        mtl_lines.extend(
            [
                f"newmtl {name}",
                f"Kd {color[0]:.4f} {color[1]:.4f} {color[2]:.4f}",
                "Ka 0.1 0.1 0.1",
                "Ks 0.0 0.0 0.0",
                "d 1.0",
            ]
        )
    for v in vertices:
        obj_lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")

    face_offset = 0
    for name, _, start, end, face_count in materials:
        obj_lines.append(f"usemtl {name}")
        for f in faces[face_offset : face_offset + face_count]:
            obj_lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}")
        face_offset += face_count

    scene_mtl_path.write_text("\n".join(mtl_lines) + "\n", encoding="utf-8")
    scene_path.write_text("\n".join(obj_lines) + "\n", encoding="utf-8")


def assess_convergence(
    history: list[dict],
    final_params: MeshJarParams,
    final_mask_iou: float,
    table_plane: TablePlaneEstimate | None = None,
) -> dict:
    optimizer_success = len(history) > 0
    loss_stable = False
    if len(history) >= 10:
        last_vals = [h["total"] for h in history[-10:]]
        diffs = [abs(last_vals[i + 1] - last_vals[i]) for i in range(len(last_vals) - 1)]
        loss_stable = max(diffs) < 1e-2
    parameters_reasonable = final_params.scale > 0 and final_params.z > 0
    table_axis_error = None
    if table_plane is not None:
        table_axis = -table_plane.normal.astype(np.float32)
        table_axis_error = axis_alignment_loss(final_params.roll, final_params.pitch, final_params.yaw, table_axis)
        if table_axis_error > 0.15:
            parameters_reasonable = False
    visual_fit_reasonable = final_mask_iou >= 0.55
    overall = optimizer_success and parameters_reasonable and visual_fit_reasonable
    return {
        "optimizer_success": optimizer_success,
        "loss_stable": bool(loss_stable),
        "parameters_reasonable": bool(parameters_reasonable),
        "visual_fit_reasonable": bool(visual_fit_reasonable),
        "overall_trustworthy": bool(overall),
        "final_mask_iou": float(final_mask_iou),
        "table_axis_error": None if table_axis_error is None else float(table_axis_error),
    }


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    observed_mask = load_mask(args.mask)
    observed_mask = (observed_mask > 0.5).astype(np.uint8)
    observed_depth = load_depth(args.depth)
    intrinsics = CameraIntrinsics(fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy)

    verts_raw, faces = load_ply_mesh(args.ply)
    verts_local, canonical_rot = canonicalize_mesh(verts_raw)
    init_result = initialize_from_mesh_and_mask(verts_local, faces, observed_mask, observed_depth, intrinsics)
    init_params = init_result.params

    loss_history: list[dict] = []
    objective = make_objective(verts_local, faces, observed_mask, observed_depth, intrinsics, loss_history, init_params, init_result.table_plane)
    result = minimize(
        objective,
        x0=pack_optimization_vector(init_params),
        method="Powell",
        options={"maxiter": args.maxiter, "disp": True},
    )

    final_params = unpack_optimization_vector(result.x, init_params)
    rendered_mask = render_mesh_mask(verts_local, faces, final_params, observed_mask.shape, intrinsics)
    rendered_depth = render_mesh_depth(verts_local, faces, final_params, observed_mask.shape, intrinsics)
    final_mask_iou = 1.0 - mask_iou_loss(rendered_mask, observed_mask)
    convergence = assess_convergence(loss_history, final_params, final_mask_iou, init_result.table_plane)

    jar_vertices_world = transform_mesh_vertices(verts_local, final_params)
    summary = {
        "success": bool(result.success),
        "message": str(result.message),
        "final_loss": float(result.fun),
        "nfev": int(result.nfev),
        "initial_params": asdict(init_params),
        "final_params": asdict(final_params),
        "convergence": convergence,
        "mesh_model": {
            "num_vertices": int(verts_local.shape[0]),
            "num_faces": int(faces.shape[0]),
            "extent_local": (verts_local.max(axis=0) - verts_local.min(axis=0)).tolist(),
        },
        "initialization": {
            "source": init_result.source,
            "pose_locked_during_optimization": True,
            "init_center_world": init_result.init_center_world.tolist(),
        },
        "table_plane": None
        if init_result.table_plane is None
        else {
            "source": init_result.table_plane.source,
            "normal": init_result.table_plane.normal.tolist(),
            "offset": float(init_result.table_plane.offset),
            "point_count": int(init_result.table_plane.point_count),
            "rms_error": float(init_result.table_plane.rms_error),
        },
    }

    if init_result.table_plane is not None:
        base_canvas = np.zeros((observed_depth.shape[0], observed_depth.shape[1], 3), dtype=np.uint8)
        table_vis, table_vis_meta = draw_table_normal_visualization(base_canvas, init_result.table_plane, intrinsics)
        if table_vis is not None and table_vis_meta is not None:
            cv2.imwrite(str(Path(out_dir) / "table_normal_overlay.png"), table_vis)
            summary["table_plane_visualization"] = table_vis_meta
            cv2.imwrite(str(Path(out_dir) / "table_plane_inliers.png"), (init_result.table_plane.inlier_mask * 255).astype(np.uint8))
            cv2.imwrite(str(Path(out_dir) / "table_plane_ring.png"), (init_result.table_plane.ring_mask * 255).astype(np.uint8))

    export_scene_obj(Path(out_dir), jar_vertices_world, faces, init_result.table_plane)

    (Path(out_dir) / "fit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (Path(out_dir) / "loss_history.json").write_text(json.dumps(loss_history, indent=2), encoding="utf-8")
    cv2.imwrite(str(Path(out_dir) / "observed_mask.png"), (observed_mask * 255).astype(np.uint8))
    cv2.imwrite(str(Path(out_dir) / "rendered_mask.png"), (rendered_mask * 255).astype(np.uint8))
    cv2.imwrite(str(Path(out_dir) / "mask_overlay.png"), cv2.cvtColor(overlay_masks(observed_mask, rendered_mask), cv2.COLOR_RGB2BGR))
    save_depth_visual(Path(out_dir) / "observed_depth.png", observed_depth, observed_mask)
    save_depth_visual(Path(out_dir) / "rendered_depth.png", rendered_depth, rendered_mask)
    print(json.dumps(summary, indent=2))
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
