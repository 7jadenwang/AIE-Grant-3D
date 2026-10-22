"""Mesh loading and anisotropic voxelization for the 3D curing simulator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


SUPPORTED_EXTENSIONS = {".stl", ".obj", ".ply", ".glb", ".3mf"}


class GeometryError(ValueError):
    """Raised when source geometry is unusable for the simulator."""


def _as_single_mesh(geometry: trimesh.Trimesh | trimesh.Scene) -> trimesh.Trimesh:
    if isinstance(geometry, trimesh.Scene):
        geometry = geometry.to_geometry()
    if isinstance(geometry, trimesh.Trimesh):
        return geometry
    meshes = [item for item in geometry if isinstance(item, trimesh.Trimesh)]
    if not meshes:
        raise GeometryError("The model is empty or contains no mesh geometry.")
    return trimesh.util.concatenate(meshes)


def load_mesh(
    filepath: str | Path,
    rotations_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    require_watertight: bool = False,
) -> trimesh.Trimesh:
    """Load, merge, validate, and rotate a supported mesh."""
    path = Path(filepath)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise GeometryError(f"Unsupported model format: {path.suffix or '<none>'}")
    try:
        mesh = _as_single_mesh(trimesh.load(path, force="scene", process=True))
    except Exception as exc:
        raise GeometryError(f"Could not load mesh: {path}") from exc
    if mesh.is_empty or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise GeometryError("The model is empty or contains no faces.")
    if require_watertight and not mesh.is_watertight:
        raise GeometryError("The model must be watertight; automatic repair is disabled.")
    angles = np.radians(np.asarray(rotations_deg, dtype=float))
    mesh = mesh.copy()
    mesh.apply_transform(trimesh.transformations.euler_matrix(*angles, axes="sxyz"))
    return mesh


def _scaled_mesh(
    mesh: trimesh.Trimesh, active_pixels: int, xy_pitch: float
) -> tuple[trimesh.Trimesh, np.ndarray, float]:
    extents = np.asarray(mesh.extents, dtype=float)
    if np.any(extents <= 0) or max(extents[:2]) <= 0:
        raise GeometryError("The model must have nonzero X, Y, and Z extents.")
    scale = (active_pixels - 1) * xy_pitch / max(extents[:2])
    scaled = mesh.copy()
    scaled.apply_scale(scale)
    scaled.apply_translation(-scaled.bounds[0])
    return scaled, np.asarray(scaled.extents), scale


def _slice_mask(
    mesh: trimesh.Trimesh, z_position: float, shape_yx: tuple[int, int], xy_pitch: float
) -> np.ndarray:
    import cv2

    section = mesh.section(plane_origin=[0.0, 0.0, z_position], plane_normal=[0.0, 0.0, 1.0])
    mask = np.zeros(shape_yx, dtype=np.uint8)
    if section is None:
        return mask.astype(bool)
    contours = []
    for loop in section.discrete:
        points = np.rint(np.asarray(loop)[:, :2] / xy_pitch).astype(np.int32)
        contours.append(points.reshape((-1, 1, 2)))
    if contours:
        cv2.fillPoly(mask, contours, color=1)
    return mask.astype(bool)


def voxelize_mesh(
    mesh: trimesh.Trimesh,
    canvas_size: int = 500,
    padding: int = 50,
    xy_pitch: float = 7.6e-6,
    layer_height: float = 50e-6,
) -> tuple[np.ndarray, dict[str, object]]:
    """Uniformly scale a mesh in physical space and voxelize it on an XYZ grid."""
    active = canvas_size - 2 * padding
    if active < 2 or xy_pitch <= 0 or layer_height <= 0:
        raise GeometryError("Canvas, padding, and voxel spacing must be positive and usable.")
    scaled, extents, scale = _scaled_mesh(mesh, active, xy_pitch)
    xy_counts = np.rint(extents[:2] / xy_pitch).astype(int) + 1
    z_ratio = extents[2] / layer_height
    exact_multiple = np.isclose(z_ratio, round(z_ratio), rtol=0.0, atol=1e-10)
    z_count = int(round(z_ratio)) if exact_multiple else int(np.ceil(z_ratio))
    counts = np.array([xy_counts[0], xy_counts[1], max(1, z_count)], dtype=int)
    if counts[0] > active or counts[1] > active:
        raise GeometryError("Scaled mesh exceeds the padded XY canvas.")
    x_offset = (canvas_size - counts[0]) // 2
    y_offset = (canvas_size - counts[1]) // 2
    target = np.zeros((counts[2], canvas_size, canvas_size), dtype=bool)
    for z_index in range(counts[2]):
        lower = z_index * layer_height
        upper = min((z_index + 1) * layer_height, extents[2])
        z_position = 0.5 * (lower + upper)
        local = _slice_mask(scaled, z_position, (counts[1], counts[0]), xy_pitch)
        target[z_index, y_offset : y_offset + counts[1], x_offset : x_offset + counts[0]] = local
    if not target.any():
        raise GeometryError("Mesh voxelization produced an empty target.")
    info: dict[str, object] = {
        "source_extents": mesh.extents.tolist(),
        "scale_factor": float(scale),
        "scaled_extents_m": extents.tolist(),
        "voxel_shape_zyx": list(target.shape),
        "xy_offsets_pixels": [int(x_offset), int(y_offset)],
        "spacing_m_zyx": [float(layer_height), float(xy_pitch), float(xy_pitch)],
    }
    return target, info
