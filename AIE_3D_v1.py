"""Editable entry point for the 3D vat photopolymerization simulator v1."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

from aie_3d_geometry import load_mesh, voxelize_mesh
from aie_3d_outputs import CrossSectionRecorder, save_final_outputs
from aie_3d_physics import (
    PhysicsConfig,
    diffusion_stability_limit,
    estimate_peak_memory_bytes,
    run_simulation,
)


# ----------------------------- EDIT PARAMETERS HERE -----------------------------
MODEL_FILEPATH = r"D:/Research/AIE Future/AIE Grant 3D/AMME CAD Models/bench.stl"  # Required: STL, OBJ, PLY, GLB, or 3MF.
ROTATION_XYZ_DEG = (0.0, 0.0, 0.0)  # File Z remains the build axis after rotation.
REQUIRE_WATERTIGHT = False  # Set True only for strict closed-mesh validation.

CANVAS_SIZE = 500
PADDING_PIXELS = 50
XY_PITCH_M = 7.6e-6
LAYER_HEIGHT_M = 50e-6

CROSS_SECTION_AXIS = "y"
CROSS_SECTION_INDEX = None  # None selects the central plane for the chosen axis.

INTENSITY_MW_CM2 = 30.0
EXPOSURE_S_PER_LAYER = 6.0
DARK_S_PER_LAYER = 2.0
# 0.05 s is unstable for the stated pitch and diffusivity; 0.025 s is the
# smallest simple correction and keeps the requested 0.10 s output cadence.
DT_S = 0.025
GAUSSIAN_KERNEL_WIDTH_M = 50e-6
AXIAL_DECAY_LENGTH_M = 100e-6  # Explicitly uncalibrated v1 assumption.
LIGHT_CUTOFF_FRACTION = 0.01
OXYGEN_BUFFER_LAYERS = 2

INHIBITION_REFERENCE_MJ_CM2 = 12.6124
INHIBITION_REFERENCE_THICKNESS_M = 100e-6
OXYGEN_DIFFUSIVITY_M2_S = 400e-12
PRE_SLOPE = 0.5e-3
PRE_CAP = 0.015
B_COEFFICIENT = 0.1131
B_EXPONENT = 0.6038

RECORD_EVERY_STEPS = 200
GIF_FPS = 10.0
CURED_THRESHOLD = 0.77
VISUALIZATION_MAX_DIM = 128
RESULTS_ROOT = Path("results")

DEVICE = "cuda"
ALLOW_CPU_MAIN_RUN = False  # CPU is intended only for small tests.
GPU_FREE_MEMORY_FRACTION = 0.90
# -------------------------------------------------------------------------------


def _physics_config() -> PhysicsConfig:
    return PhysicsConfig(
        intensity_mw_cm2=INTENSITY_MW_CM2,
        exposure_s=EXPOSURE_S_PER_LAYER,
        dark_s=DARK_S_PER_LAYER,
        dt_s=DT_S,
        oxygen_diffusivity_m2_s=OXYGEN_DIFFUSIVITY_M2_S,
        xy_pitch_m=XY_PITCH_M,
        layer_height_m=LAYER_HEIGHT_M,
        inhibition_reference_mj_cm2=INHIBITION_REFERENCE_MJ_CM2,
        inhibition_reference_thickness_m=INHIBITION_REFERENCE_THICKNESS_M,
        blur_width_m=GAUSSIAN_KERNEL_WIDTH_M,
        axial_decay_length_m=AXIAL_DECAY_LENGTH_M,
        light_cutoff_fraction=LIGHT_CUTOFF_FRACTION,
        oxygen_buffer_layers=OXYGEN_BUFFER_LAYERS,
        b_coefficient=B_COEFFICIENT,
        b_exponent=B_EXPONENT,
        pre_slope=PRE_SLOPE,
        pre_cap=PRE_CAP,
        record_every_steps=RECORD_EVERY_STEPS,
    )


def _resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; no CPU fallback was applied.")
    if device.type == "cpu" and not ALLOW_CPU_MAIN_RUN:
        raise RuntimeError("Set ALLOW_CPU_MAIN_RUN=True to explicitly allow a small CPU run.")
    return device


def _check_stability(config: PhysicsConfig) -> float:
    spacing = (config.layer_height_m, config.xy_pitch_m, config.xy_pitch_m)
    limit = diffusion_stability_limit(config.oxygen_diffusivity_m2_s, spacing)
    if config.dt_s > limit:
        raise ValueError(
            f"DT_S={config.dt_s:g} s exceeds the explicit 3D diffusion stability limit "
            f"{limit:.6g} s. Reduce DT_S; the simulator will not silently substep."
        )
    return limit


def _memory_preflight(shape: tuple[int, int, int], device: torch.device) -> dict[str, int | float]:
    estimated = estimate_peak_memory_bytes(shape)
    report: dict[str, int | float] = {"estimated_peak_bytes": estimated}
    if device.type != "cuda":
        return report
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    allowed = int(free_bytes * GPU_FREE_MEMORY_FRACTION)
    report.update(free_bytes=free_bytes, total_bytes=total_bytes, allowed_bytes=allowed)
    if estimated > allowed:
        raise MemoryError(
            f"Estimated peak GPU memory is {estimated / 2**30:.2f} GiB, above "
            f"{GPU_FREE_MEMORY_FRACTION:.0%} of currently free memory ({allowed / 2**30:.2f} GiB)."
        )
    return report


def _section_index(axis: str, shape: tuple[int, int, int]) -> int:
    dimension = {"x": 2, "y": 1, "z": 0}.get(axis.lower())
    if dimension is None:
        raise ValueError("CROSS_SECTION_AXIS must be 'x', 'y', or 'z'.")
    return shape[dimension] // 2 if CROSS_SECTION_INDEX is None else int(CROSS_SECTION_INDEX)


def _output_directory(model_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = RESULTS_ROOT / f"{model_path.stem}_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    return output


def _assumptions() -> list[str]:
    return [
        "0 mM TEMPO; oxygen is the only inhibitor",
        "closed no-flux 3D oxygen reservoir",
        "uniform oxygen diffusivity in liquid and cured regions",
        "no air influx, fluid flow, peel deformation, or future-layer evolution",
        "constant lateral blur with propagation depth",
        "axial decay length is an uncalibrated v1 parameter",
        "layers below the optical window plus oxygen buffer are frozen",
        "the frozen-window boundary is no-flux",
    ]


def main() -> None:
    model_path = Path(MODEL_FILEPATH)
    if not MODEL_FILEPATH.strip() or not model_path.is_file():
        raise FileNotFoundError("Set MODEL_FILEPATH to an existing STL, OBJ, PLY, GLB, or 3MF file.")
    config = _physics_config()
    stability_limit = _check_stability(config)
    device = _resolve_device(DEVICE)
    mesh = load_mesh(model_path, ROTATION_XYZ_DEG, require_watertight=REQUIRE_WATERTIGHT)
    target, geometry = voxelize_mesh(
        mesh, CANVAS_SIZE, PADDING_PIXELS, XY_PITCH_M, LAYER_HEIGHT_M
    )
    memory = _memory_preflight(target.shape, device)
    output = _output_directory(model_path)
    index = _section_index(CROSS_SECTION_AXIS, target.shape)
    total_steps = round((EXPOSURE_S_PER_LAYER + DARK_S_PER_LAYER) / DT_S) * target.shape[0]
    print(
        f"Grid Z/Y/X: {target.shape}; active Z window: up to {config.active_layers} layers; "
        f"estimated recorded frames: {total_steps // RECORD_EVERY_STEPS}"
    )
    recorder = CrossSectionRecorder(
        output, CROSS_SECTION_AXIS, index, target.shape, config.bulk_capacity, GIF_FPS
    )
    result = run_simulation(target, config, device=device, snapshot_callback=recorder)
    summary = {
        "model_filepath": str(model_path.resolve()), "rotation_xyz_deg": ROTATION_XYZ_DEG,
        "require_watertight": REQUIRE_WATERTIGHT,
        "physics": asdict(config), "geometry": geometry, "cross_section": {
            "axis": CROSS_SECTION_AXIS, "index": index, "record_every_steps": RECORD_EVERY_STEPS,
        }, "device": str(device), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "memory_estimate": memory, "diffusion_stability_limit_s": stability_limit,
        "derived_active_window": {
            "light_layers": config.light_layers, "oxygen_buffer_layers": config.oxygen_buffer_layers,
            "maximum_active_layers": config.active_layers,
        },
        "simulation_elapsed_s": result.elapsed_s, "runtime_s": result.runtime_s,
        "assumptions": _assumptions(),
    }
    artifacts = save_final_outputs(
        output, target, result.doc, result.oxygen, summary, CURED_THRESHOLD,
        (LAYER_HEIGHT_M, XY_PITCH_M, XY_PITCH_M), VISUALIZATION_MAX_DIM,
    )
    recorder.finish()
    print(f"Results: {output.resolve()}")
    if not artifacts["surface_produced"]:
        print(f"No voxel reached DoC={CURED_THRESHOLD:g}; no isosurface files were produced.")


if __name__ == "__main__":
    main()
