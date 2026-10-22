"""Cross-section recording and final numeric/3D output generation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
import plotly.graph_objects as go
import trimesh
from skimage import measure

from aie_3d_physics import SimulationSnapshot


def extract_cross_section(volume: np.ndarray, axis: str, index: int) -> np.ndarray:
    """Select an X, Y, or Z plane from a volume stored in [Z, Y, X] order."""
    if axis not in {"x", "y", "z"}:
        raise ValueError("Cross-section axis must be 'x', 'y', or 'z'.")
    dimension = {"x": 2, "y": 1, "z": 0}[axis]
    if index < 0 or index >= volume.shape[dimension]:
        raise IndexError(f"Cross-section {axis} index {index} is outside the volume.")
    if axis == "x":
        return volume[:, :, index]
    if axis == "y":
        return volume[:, index, :]
    return volume[index]


def _render_frame(
    section: np.ndarray, value_range: tuple[float, float], title: str, path: Path, colormap: int
) -> None:
    low, high = value_range
    scaled = np.clip((section - low) / max(high - low, 1e-30), 0.0, 1.0)
    invalid = ~np.isfinite(section)
    colored = cv2.applyColorMap(np.nan_to_num(scaled * 255).astype(np.uint8), colormap)
    colored[invalid] = (35, 35, 35)
    width = 700
    height = max(240, int(width * colored.shape[0] / max(colored.shape[1], 1)))
    colored = cv2.resize(colored, (width, height), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((height + 82, width, 3), 245, dtype=np.uint8)
    canvas[52 : 52 + height] = colored
    cv2.putText(canvas, title, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 1)
    cv2.putText(
        canvas, f"fixed range: {low:g} to {high:g}", (12, height + 74),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1,
    )
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"Could not write image: {path}")


def _write_gif(paths: list[Path], destination: Path, fps: float) -> None:
    if not paths:
        return
    with imageio.get_writer(destination, mode="I", duration=1000.0 / fps, loop=0) as writer:
        for path in paths:
            writer.append_data(imageio.imread(path))


class CrossSectionRecorder:
    """Write only the selected 2D plane at each requested simulation snapshot."""

    def __init__(
        self,
        output_dir: str | Path,
        axis: str,
        index: int,
        final_shape: tuple[int, int, int],
        initial_capacity: float,
        gif_fps: float = 10.0,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.axis, self.index = axis.lower(), int(index)
        self.final_shape = tuple(int(value) for value in final_shape)
        self.initial_capacity = float(initial_capacity)
        self.gif_fps = float(gif_fps)
        dimension = {"x": 2, "y": 1, "z": 0}.get(self.axis)
        if dimension is None:
            raise ValueError("Cross-section axis must be 'x', 'y', or 'z'.")
        if self.index < 0 or self.index >= self.final_shape[dimension]:
            raise IndexError(f"Cross-section {self.axis} index {self.index} is outside the volume.")
        self.doc_dir = self.output_dir / "DoC_frames"
        self.o2_dir = self.output_dir / "O2_frames"
        self.doc_dir.mkdir(parents=True, exist_ok=True)
        self.o2_dir.mkdir(parents=True, exist_ok=True)
        self.doc_paths: list[Path] = []
        self.o2_paths: list[Path] = []

    def _padded(self, volume: np.ndarray) -> np.ndarray:
        if volume.shape[0] == self.final_shape[0]:
            return volume
        padded = np.full(self.final_shape, np.nan, dtype=np.float32)
        padded[: volume.shape[0]] = volume
        return padded

    def record(self, snapshot: SimulationSnapshot) -> None:
        doc = extract_cross_section(self._padded(snapshot.doc), self.axis, self.index)
        oxygen = extract_cross_section(self._padded(snapshot.oxygen), self.axis, self.index)
        self._save_sections(doc, oxygen, snapshot.elapsed_s, snapshot.phase, snapshot.layer_index)

    def _save_sections(
        self, doc: np.ndarray, oxygen: np.ndarray, elapsed_s: float, phase: str, layer_index: int
    ) -> None:
        number = len(self.doc_paths)
        doc_path, o2_path = (
            self.doc_dir / f"DoC_{number:06d}.png",
            self.o2_dir / f"O2_{number:06d}.png",
        )
        title = (
            f"t={elapsed_s:.2f} s | {phase} | "
            f"layer {layer_index}/{self.final_shape[0]}"
        )
        _render_frame(doc, (0.0, 1.0), f"DoC | {title}", doc_path, cv2.COLORMAP_VIRIDIS)
        _render_frame(
            oxygen, (0.0, self.initial_capacity), f"O2 capacity | {title}",
            o2_path, cv2.COLORMAP_VIRIDIS,
        )
        self.doc_paths.append(doc_path)
        self.o2_paths.append(o2_path)

    def _tensor_section(self, field: Any) -> np.ndarray:
        if self.axis == "x":
            section = field[:, :, self.index]
        elif self.axis == "y":
            section = field[:, self.index, :]
        elif self.index >= field.shape[0]:
            return np.full(self.final_shape[1:], np.nan, dtype=np.float32)
        else:
            section = field[self.index]
        array = section.detach().cpu().numpy()
        if self.axis in {"x", "y"} and array.shape[0] < self.final_shape[0]:
            width = self.final_shape[1] if self.axis == "x" else self.final_shape[2]
            padded = np.full((self.final_shape[0], width), np.nan, dtype=np.float32)
            padded[: array.shape[0]] = array
            return padded
        return np.array(array, copy=True)

    def capture_tensors(
        self, doc: Any, oxygen: Any, *, elapsed_s: float, phase: str,
        layer_index: int, step_index: int,
    ) -> None:
        del step_index
        self._save_sections(
            self._tensor_section(doc), self._tensor_section(oxygen), elapsed_s, phase, layer_index
        )

    def finish(self) -> None:
        _write_gif(self.doc_paths, self.output_dir / "allDoC.gif", self.gif_fps)
        _write_gif(self.o2_paths, self.output_dir / "allO2.gif", self.gif_fps)


def downsample_volume(volume: np.ndarray, max_dim: int = 128) -> np.ndarray:
    if max_dim < 1:
        raise ValueError("max_dim must be positive.")
    strides = tuple(max(1, math.ceil(size / max_dim)) for size in volume.shape)
    reduced = volume[tuple(slice(None, None, stride) for stride in strides)]
    return np.array(reduced, copy=True)


def prepare_volume_render(
    volume: np.ndarray, spacing: tuple[float, float, float], max_dim: int = 128
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    strides = tuple(max(1, math.ceil(size / max_dim)) for size in volume.shape)
    reduced = volume[tuple(slice(None, None, stride) for stride in strides)].copy()
    z_index, y_index, x_index = np.indices(reduced.shape, dtype=np.float64)
    z = z_index * strides[0] * spacing[0]
    y = y_index * strides[1] * spacing[1]
    x = x_index * strides[2] * spacing[2]
    return reduced, x, y, z


def _volume_html(
    doc: np.ndarray,
    destination: Path,
    max_dim: int,
    spacing: tuple[float, float, float],
) -> None:
    reduced, x, y, z = prepare_volume_render(doc, spacing, max_dim)
    figure = go.Figure(
        go.Volume(
            x=x.ravel(), y=y.ravel(), z=z.ravel(), value=reduced.ravel(),
            isomin=0.01, isomax=1.0, opacity=0.12, surface_count=18,
            colorscale="Viridis", colorbar={"title": "DoC"},
        )
    )
    figure.update_layout(
        title="Final degree of conversion", scene_aspectmode="data",
        scene={"xaxis_title": "X (m)", "yaxis_title": "Y (m)", "zaxis_title": "Z (m)"},
    )
    figure.write_html(destination, include_plotlyjs=True, full_html=True)


def _isosurface(
    doc: np.ndarray,
    threshold: float,
    spacing: tuple[float, float, float],
    html_path: Path,
    stl_path: Path,
) -> bool:
    if not 0 < threshold <= 1 or float(np.max(doc)) < threshold:
        return False
    padded = np.pad(doc.astype(np.float32), 1, mode="constant")
    level = min(threshold, float(np.max(padded)) - np.finfo(np.float32).eps)
    vertices_zyx, faces, _, _ = measure.marching_cubes(padded, level=level, spacing=spacing)
    vertices_zyx -= np.asarray(spacing)
    vertices_xyz = vertices_zyx[:, [2, 1, 0]]
    trimesh.Trimesh(vertices=vertices_xyz, faces=faces, process=False).export(stl_path)
    figure = go.Figure(
        go.Mesh3d(
            x=vertices_xyz[:, 0], y=vertices_xyz[:, 1], z=vertices_xyz[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], intensity=vertices_zyx[:, 0],
            colorscale="Viridis", opacity=0.75,
        )
    )
    figure.update_layout(title=f"Cured isosurface (DoC >= {threshold:g})", scene_aspectmode="data")
    figure.write_html(html_path, include_plotlyjs=True, full_html=True)
    return True


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def save_final_outputs(
    output_dir: str | Path,
    target: np.ndarray,
    doc: np.ndarray,
    oxygen: np.ndarray,
    summary: dict[str, Any],
    threshold: float,
    spacing_m: tuple[float, float, float],
    visualization_max_dim: int = 128,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "final_fields.npz", target=target, doc=doc, oxygen=oxygen)
    _volume_html(doc, output / "final_DoC_volume.html", visualization_max_dim, spacing_m)
    surface = _isosurface(
        doc, threshold, spacing_m,
        output / "cured_isosurface.html", output / "cured_isosurface.stl",
    )
    full_summary = dict(summary)
    full_summary["surface_produced"] = surface
    full_summary["surface_threshold"] = threshold
    (output / "run_summary.json").write_text(
        json.dumps(full_summary, indent=2, default=_json_default), encoding="utf-8"
    )
    return {"surface_produced": surface, "output_dir": str(output)}
