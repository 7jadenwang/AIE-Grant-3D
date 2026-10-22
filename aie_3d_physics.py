"""Optics, oxygen diffusion, and incremental curing physics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from time import perf_counter
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PhysicsConfig:
    intensity_mw_cm2: float = 30.0
    exposure_s: float = 6.0
    dark_s: float = 2.0
    dt_s: float = 0.025
    oxygen_diffusivity_m2_s: float = 400e-12
    xy_pitch_m: float = 7.6e-6
    layer_height_m: float = 50e-6
    initial_capacity_mj_cm2: float | None = None
    inhibition_reference_mj_cm2: float = 12.6124
    inhibition_reference_thickness_m: float = 100e-6
    blur_width_m: float = 50e-6
    axial_decay_length_m: float = 100e-6
    light_cutoff_fraction: float = 0.01
    oxygen_buffer_layers: int = 2
    b_coefficient: float = 0.1131
    b_exponent: float = 0.6038
    pre_slope: float = 0.5e-3
    pre_cap: float = 0.015
    record_every_steps: int = 4

    @property
    def bulk_capacity(self) -> float:
        if self.initial_capacity_mj_cm2 is not None:
            return self.initial_capacity_mj_cm2
        return initial_oxygen_capacity(
            self.inhibition_reference_mj_cm2,
            self.layer_height_m,
            self.inhibition_reference_thickness_m,
        )

    @property
    def light_layers(self) -> int:
        return light_window_layers(
            self.light_cutoff_fraction, self.axial_decay_length_m, self.layer_height_m
        )

    @property
    def active_layers(self) -> int:
        return self.light_layers + self.oxygen_buffer_layers


@dataclass(frozen=True)
class SimulationSnapshot:
    doc: np.ndarray
    oxygen: np.ndarray
    elapsed_s: float
    phase: str
    layer_index: int
    step_index: int


@dataclass(frozen=True)
class SimulationResult:
    doc: np.ndarray
    oxygen: np.ndarray
    elapsed_s: float
    runtime_s: float
    steps: int


def initial_oxygen_capacity(
    reference_energy_mj_cm2: float, layer_height_m: float, reference_thickness_m: float
) -> float:
    if min(reference_energy_mj_cm2, layer_height_m, reference_thickness_m) <= 0:
        raise ValueError("Inhibition reference values and layer height must be positive.")
    return reference_energy_mj_cm2 * layer_height_m / reference_thickness_m


def build_depth_intensity(
    focal_plane: torch.Tensor, depth: int, layer_height: float, axial_decay_length: float
) -> torch.Tensor:
    if depth < 1 or layer_height <= 0 or axial_decay_length <= 0:
        raise ValueError("Depth and optical lengths must be positive.")
    distances = torch.arange(depth - 1, -1, -1, device=focal_plane.device, dtype=focal_plane.dtype)
    attenuation = torch.exp(-distances * layer_height / axial_decay_length)
    return attenuation[:, None, None] * focal_plane[None, :, :]


def light_window_layers(
    cutoff_fraction: float, axial_decay_length: float, layer_height: float
) -> int:
    """Number of layers, including the current layer, above the optical cutoff."""
    if not 0.0 < cutoff_fraction < 1.0:
        raise ValueError("Light cutoff fraction must be between zero and one.")
    if axial_decay_length <= 0 or layer_height <= 0:
        raise ValueError("Axial decay length and layer height must be positive.")
    maximum_distance = -axial_decay_length * math.log(cutoff_fraction)
    return math.floor(maximum_distance / layer_height) + 1


def active_window_start(layer_count: int, light_layers: int, buffer_layers: int) -> int:
    """Return the first zero-based layer updated in the current cycle."""
    if layer_count < 1 or light_layers < 1 or buffer_layers < 0:
        raise ValueError("Layer count and light window must be positive; buffer cannot be negative.")
    return max(0, layer_count - light_layers - buffer_layers)


def gaussian_blur_mask(mask: torch.Tensor, kernel_width_m: float, xy_pitch_m: float) -> torch.Tensor:
    mask = mask.to(dtype=torch.float32)
    if kernel_width_m <= 0:
        return mask
    kernel_size = max(1, int(kernel_width_m / xy_pitch_m))
    kernel_size += int(kernel_size % 2 == 0)
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    coords = torch.arange(kernel_size, device=mask.device, dtype=mask.dtype) - kernel_size // 2
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    kernel = kernel_1d[:, None] * kernel_1d[None, :]
    pad = kernel_size // 2
    mode = "reflect" if min(mask.shape) > pad else "replicate"
    image = F.pad(mask[None, None], (pad, pad, pad, pad), mode=mode)
    return F.conv2d(image, kernel[None, None])[0, 0]


def diffusion_stability_limit(diffusivity: float, spacing: tuple[float, float, float]) -> float:
    if diffusivity < 0 or min(spacing) <= 0:
        raise ValueError("Diffusivity cannot be negative and spacing must be positive.")
    if diffusivity == 0:
        return float("inf")
    inverse_sum = sum(1.0 / (distance * distance) for distance in spacing)
    return 1.0 / (2.0 * diffusivity * inverse_sum)


def _add_axis_flux(laplacian: torch.Tensor, field: torch.Tensor, axis: int, spacing: float) -> None:
    left = [slice(None)] * 3
    right = [slice(None)] * 3
    left[axis], right[axis] = slice(None, -1), slice(1, None)
    flux = (field[tuple(right)] - field[tuple(left)]) / (spacing * spacing)
    laplacian[tuple(left)] += flux
    laplacian[tuple(right)] -= flux


def diffuse_no_flux(
    field: torch.Tensor, diffusivity: float, dt: float, spacing: tuple[float, float, float]
) -> torch.Tensor:
    limit = diffusion_stability_limit(diffusivity, spacing)
    if dt < 0 or dt > limit * (1.0 + 1e-12):
        raise ValueError(f"dt={dt:g} s exceeds the explicit diffusion stability limit {limit:g} s.")
    if diffusivity == 0 or dt == 0:
        return field.clone()
    laplacian = torch.zeros_like(field)
    for axis, distance in enumerate(spacing):
        if field.shape[axis] > 1:
            _add_axis_flux(laplacian, field, axis, distance)
    return torch.clamp_min(field + diffusivity * dt * laplacian, 0.0)


def reaction_step(
    oxygen: torch.Tensor,
    doc: torch.Tensor,
    cumulative_light: torch.Tensor,
    intensity: torch.Tensor,
    dt: float,
    b_coefficient: float = 0.1131,
    b_exponent: float = 0.6038,
    pre_slope: float = 0.5e-3,
    pre_cap: float = 0.015,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dose = torch.clamp_min(intensity, 0.0) * dt
    consumed = torch.minimum(torch.clamp_min(oxygen, 0.0), dose)
    next_oxygen = torch.clamp_min(oxygen - consumed, 0.0)
    residual = torch.clamp_min(dose - consumed, 0.0)
    cure_time = torch.where(intensity > 0, residual / torch.clamp_min(intensity, 1e-30), 0.0)
    rate = b_coefficient * torch.clamp_min(intensity, 0.0).pow(b_exponent)
    cured = 1.0 - (1.0 - doc) * torch.exp(-rate * cure_time)
    next_cumulative = cumulative_light + dose
    precure = torch.clamp(pre_slope * next_cumulative, max=pre_cap)
    next_doc = torch.where(residual > 0, cured, torch.maximum(doc, precure))
    return next_oxygen, torch.clamp(next_doc, 0.0, 1.0), next_cumulative


def append_resin_layer(
    field: torch.Tensor, fill_value: float, height: int, width: int
) -> torch.Tensor:
    layer = torch.full((1, height, width), fill_value, dtype=field.dtype, device=field.device)
    return torch.cat((field, layer), dim=0)


def _step_count(duration: float, dt: float) -> int:
    if duration < 0 or dt <= 0:
        raise ValueError("Durations cannot be negative and dt must be positive.")
    steps = int(round(duration / dt))
    if not np.isclose(steps * dt, duration, rtol=0, atol=1e-9):
        raise ValueError("Exposure and dark durations must be integer multiples of dt.")
    return steps


def _emit_snapshot(
    callback: Callable[[SimulationSnapshot], None] | Any | None,
    doc: torch.Tensor,
    oxygen: torch.Tensor,
    elapsed: float,
    phase: str,
    layer_index: int,
    step_index: int,
) -> None:
    if callback is None:
        return
    tensor_recorder = getattr(callback, "capture_tensors", None)
    if callable(tensor_recorder):
        tensor_recorder(
            doc, oxygen, elapsed_s=elapsed, phase=phase,
            layer_index=layer_index, step_index=step_index,
        )
        return
    callback(
        SimulationSnapshot(
            doc.detach().cpu().numpy().copy(),
            oxygen.detach().cpu().numpy().copy(),
            elapsed,
            phase,
            layer_index,
            step_index,
        )
    )


def _validate_config(config: PhysicsConfig) -> tuple[int, int]:
    exposure_steps = _step_count(config.exposure_s, config.dt_s)
    dark_steps = _step_count(config.dark_s, config.dt_s)
    if config.intensity_mw_cm2 < 0 or config.record_every_steps < 1:
        raise ValueError("Intensity cannot be negative and recording interval must be positive.")
    if isinstance(config.oxygen_buffer_layers, bool) or not isinstance(
        config.oxygen_buffer_layers, Integral
    ):
        raise ValueError("Oxygen buffer layer count must be an integer.")
    if config.oxygen_buffer_layers < 0:
        raise ValueError("Oxygen buffer layer count cannot be negative.")
    config.light_layers
    spacing = (config.layer_height_m, config.xy_pitch_m, config.xy_pitch_m)
    if config.dt_s > diffusion_stability_limit(config.oxygen_diffusivity_m2_s, spacing):
        raise ValueError("dt exceeds the anisotropic explicit-diffusion stability limit.")
    return exposure_steps, dark_steps


def _window_intensity(
    focal: torch.Tensor, layer_count: int, config: PhysicsConfig
) -> tuple[int, torch.Tensor]:
    start = active_window_start(layer_count, config.light_layers, config.oxygen_buffer_layers)
    active_depth = layer_count - start
    light_depth = min(layer_count, config.light_layers)
    intensity = torch.zeros(
        (active_depth, *focal.shape), dtype=focal.dtype, device=focal.device
    )
    intensity[-light_depth:] = build_depth_intensity(
        focal * config.intensity_mw_cm2,
        light_depth,
        config.layer_height_m,
        config.axial_decay_length_m,
    )
    return start, intensity


def _advance_window(
    oxygen: torch.Tensor,
    doc: torch.Tensor,
    cumulative: torch.Tensor,
    intensity: torch.Tensor,
    start: int,
    config: PhysicsConfig,
    spacing: tuple[float, float, float],
    expose: bool,
) -> None:
    active_oxygen = diffuse_no_flux(
        oxygen[start:], config.oxygen_diffusivity_m2_s, config.dt_s, spacing
    )
    if expose:
        active_oxygen, active_doc, active_cumulative = reaction_step(
            active_oxygen, doc[start:], cumulative[start:], intensity, config.dt_s,
            config.b_coefficient, config.b_exponent, config.pre_slope, config.pre_cap,
        )
        doc[start:], cumulative[start:] = active_doc, active_cumulative
    oxygen[start:] = active_oxygen


def run_simulation(
    target_volume: np.ndarray,
    config: PhysicsConfig,
    device: str | torch.device = "cuda",
    snapshot_callback: Callable[[SimulationSnapshot], None] | Any | None = None,
) -> SimulationResult:
    """Run the layer-growing simulator without allocating future resin layers."""
    exposure_steps, dark_steps = _validate_config(config)
    target = np.asarray(target_volume, dtype=bool)
    if target.ndim != 3 or 0 in target.shape:
        raise ValueError("target_volume must be a nonempty [Z, Y, X] array.")
    torch_device = torch.device(device)
    height, width = target.shape[1:]
    fields = [torch.empty((0, height, width), device=torch_device) for _ in range(3)]
    oxygen, doc, cumulative = fields
    elapsed = 0.0
    step_index = 0
    started = perf_counter()
    spacing = (config.layer_height_m, config.xy_pitch_m, config.xy_pitch_m)
    for layer_index, mask in enumerate(target, start=1):
        oxygen = append_resin_layer(oxygen, config.bulk_capacity, height, width)
        doc = append_resin_layer(doc, 0.0, height, width)
        cumulative = append_resin_layer(cumulative, 0.0, height, width)
        focal = gaussian_blur_mask(torch.as_tensor(mask, device=torch_device), config.blur_width_m, config.xy_pitch_m)
        active_start, intensity = _window_intensity(focal, layer_index, config)
        for phase, count in (("exposure", exposure_steps), ("dark", dark_steps)):
            for _ in range(count):
                _advance_window(
                    oxygen, doc, cumulative, intensity, active_start,
                    config, spacing, phase == "exposure",
                )
                elapsed += config.dt_s
                step_index += 1
                if step_index % config.record_every_steps == 0:
                    _emit_snapshot(snapshot_callback, doc, oxygen, elapsed, phase, layer_index, step_index)
    return SimulationResult(
        doc.detach().cpu().numpy(),
        oxygen.detach().cpu().numpy(),
        elapsed,
        perf_counter() - started,
        step_index,
    )


def estimate_peak_memory_bytes(shape_zyx: tuple[int, int, int]) -> int:
    """Conservative peak estimate: target plus sixteen float work arrays."""
    voxels = int(np.prod(shape_zyx))
    return voxels * (1 + 16 * np.dtype(np.float32).itemsize)
