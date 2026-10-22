from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import trimesh

import AIE_3D_v1 as entrypoint

from aie_3d_geometry import GeometryError, load_mesh, voxelize_mesh
from aie_3d_outputs import (
    CrossSectionRecorder,
    downsample_volume,
    extract_cross_section,
    prepare_volume_render,
    save_final_outputs,
)
from aie_3d_physics import (
    PhysicsConfig,
    active_window_start,
    append_resin_layer,
    build_depth_intensity,
    diffuse_no_flux,
    diffusion_stability_limit,
    initial_oxygen_capacity,
    light_window_layers,
    reaction_step,
    run_simulation,
)


def _export_mesh(mesh: trimesh.Trimesh, path: Path) -> Path:
    payload = mesh.export(file_type=path.suffix.lstrip("."))
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_bytes(payload)
    return path


@pytest.mark.parametrize("extension", ["stl", "obj", "ply", "glb", "3mf"])
def test_loads_supported_mesh_formats(tmp_path: Path, extension: str) -> None:
    source = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
    mesh = load_mesh(_export_mesh(source, tmp_path / f"box.{extension}"))
    assert mesh.is_watertight
    assert np.allclose(np.sort(mesh.extents), [1.0, 2.0, 3.0], rtol=1e-3)


def test_voxelization_preserves_complete_aspect_ratio() -> None:
    mesh = trimesh.creation.box(extents=(1.0, 2.0, 5.0))
    target, info = voxelize_mesh(mesh, canvas_size=30, padding=5, xy_pitch=1.0, layer_height=0.5)
    scaled = np.asarray(info["scaled_extents_m"])
    assert np.allclose(scaled / scaled[0], [1.0, 2.0, 5.0], rtol=1e-6)
    assert target.shape[1:] == (30, 30)
    assert target.shape[0] == math.ceil(scaled[2] / 0.5)


def test_exact_height_multiple_does_not_add_duplicate_layer() -> None:
    mesh = trimesh.creation.box(extents=(2.0, 1.0, 2.0))
    target, info = voxelize_mesh(
        mesh, canvas_size=30, padding=5, xy_pitch=1.0, layer_height=9.5
    )
    assert info["scaled_extents_m"][2] == pytest.approx(19.0)
    assert target.shape[0] == 2


def test_partial_top_slab_is_not_rounded_away() -> None:
    mesh = trimesh.creation.box(extents=(2.0, 1.0, 2.00001))
    target, _ = voxelize_mesh(
        mesh, canvas_size=30, padding=5, xy_pitch=1.0, layer_height=9.5
    )
    assert target.shape[0] == 3


def test_voxelization_places_model_inside_500_canvas_padding() -> None:
    thin_box = trimesh.creation.box(extents=(1.0, 1.0, 1e-3))
    target, _ = voxelize_mesh(
        thin_box, canvas_size=500, padding=50, xy_pitch=1.0, layer_height=1.0
    )
    z, y, x = np.nonzero(target)
    assert (x.min(), x.max(), y.min(), y.max()) == (50, 449, 50, 449)
    assert len(z) > 0


def test_geometry_rejects_empty_and_unsupported_inputs(tmp_path: Path) -> None:
    unsupported = tmp_path / "part.step"
    unsupported.write_text("not a mesh", encoding="utf-8")
    with pytest.raises(GeometryError, match="Unsupported"):
        load_mesh(unsupported)

    empty = tmp_path / "empty.obj"
    empty.write_text("# empty", encoding="utf-8")
    with pytest.raises(GeometryError, match="(?i)empty|load"):
        load_mesh(empty)

def test_non_watertight_mesh_is_allowed_by_default_but_strict_mode_rejects_it(
    tmp_path: Path,
) -> None:
    open_mesh = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]]
    )
    path = _export_mesh(open_mesh, tmp_path / "open.stl")
    assert load_mesh(path).is_watertight is False
    with pytest.raises(GeometryError, match="watertight"):
        load_mesh(path, require_watertight=True)


def test_entrypoint_allows_non_watertight_meshes_by_default() -> None:
    assert entrypoint.REQUIRE_WATERTIGHT is False


def test_rotations_are_applied_before_voxelization(tmp_path: Path) -> None:
    path = _export_mesh(trimesh.creation.box(extents=(1.0, 2.0, 3.0)), tmp_path / "box.stl")
    rotated = load_mesh(path, rotations_deg=(0.0, 0.0, 90.0))
    assert np.allclose(rotated.extents[:2], [2.0, 1.0], atol=1e-5)


def test_initial_capacity_scales_with_layer_height() -> None:
    assert initial_oxygen_capacity(12.6124, 50e-6, 100e-6) == pytest.approx(6.3062)


def test_entrypoint_time_step_is_stable() -> None:
    config = entrypoint._physics_config()
    spacing = (config.layer_height_m, config.xy_pitch_m, config.xy_pitch_m)
    assert config.dt_s <= diffusion_stability_limit(config.oxygen_diffusivity_m2_s, spacing)
    assert PhysicsConfig().dt_s * PhysicsConfig().record_every_steps == pytest.approx(0.10)


def test_one_percent_cutoff_and_two_buffer_layers_make_twelve_layer_window() -> None:
    light_layers = light_window_layers(0.01, axial_decay_length=100e-6, layer_height=50e-6)
    assert light_layers == 10
    assert active_window_start(layer_count=20, light_layers=light_layers, buffer_layers=2) == 8


def test_layers_outside_light_window_are_frozen() -> None:
    cfg = PhysicsConfig(
        intensity_mw_cm2=10.0,
        exposure_s=0.1,
        dark_s=0.0,
        dt_s=0.1,
        oxygen_diffusivity_m2_s=0.0,
        xy_pitch_m=1.0,
        layer_height_m=1.0,
        initial_capacity_mj_cm2=0.0,
        blur_width_m=0.0,
        axial_decay_length_m=1.0,
        light_cutoff_fraction=0.5,
        oxygen_buffer_layers=1,
        record_every_steps=1,
    )
    result = run_simulation(np.ones((4, 2, 2), dtype=bool), cfg, device="cpu")
    per_layer = result.doc[:, 0, 0]
    assert np.allclose(per_layer, per_layer[0], rtol=1e-6)


def test_oxygen_stops_changing_below_no_flux_active_window() -> None:
    history: list[tuple[int, np.ndarray]] = []
    target = np.zeros((4, 1, 1), dtype=bool)
    target[0] = True
    cfg = PhysicsConfig(
        intensity_mw_cm2=1.0,
        exposure_s=0.2,
        dark_s=0.0,
        dt_s=0.1,
        oxygen_diffusivity_m2_s=0.1,
        xy_pitch_m=1.0,
        layer_height_m=1.0,
        initial_capacity_mj_cm2=1.0,
        blur_width_m=0.0,
        axial_decay_length_m=1.0,
        light_cutoff_fraction=0.5,
        oxygen_buffer_layers=1,
        record_every_steps=1,
    )
    run_simulation(
        target,
        cfg,
        device="cpu",
        snapshot_callback=lambda snapshot: history.append(
            (snapshot.layer_index, snapshot.oxygen[:, 0, 0].copy())
        ),
    )
    layer_two_final = [state for layer, state in history if layer == 2][-1]
    later_layer_zero = [state[0] for layer, state in history if layer >= 3]
    assert np.allclose(later_layer_zero, layer_two_final[0], atol=1e-7)
    layer_three = [state for layer, state in history if layer == 3]
    assert not np.isclose(layer_three[0][1], layer_three[-1][1])
    assert np.allclose([state[1:].sum() for state in layer_three], layer_three[0][1:].sum())


@pytest.mark.parametrize(
    ("cutoff", "buffer"), [(0.0, 2), (1.0, 2), (0.01, -1), (0.01, 1.5)]
)
def test_invalid_active_window_configuration_is_rejected(
    cutoff: float, buffer: int | float
) -> None:
    cfg = PhysicsConfig(
        exposure_s=0.0,
        dark_s=0.0,
        light_cutoff_fraction=cutoff,
        oxygen_buffer_layers=buffer,
    )
    with pytest.raises(ValueError, match="cutoff|buffer"):
        run_simulation(np.ones((1, 1, 1), dtype=bool), cfg, device="cpu")


def test_depth_attenuation_uses_newest_layer_as_focus() -> None:
    plane = torch.full((2, 2), 30.0)
    stack = build_depth_intensity(plane, depth=3, layer_height=50e-6, axial_decay_length=100e-6)
    assert torch.allclose(stack[2], plane)
    assert torch.allclose(stack[1], plane * math.exp(-0.5))
    assert torch.allclose(stack[0], plane * math.exp(-1.0))


def test_append_layer_preserves_old_state_and_initializes_new_layer() -> None:
    old = torch.tensor([[[2.0, 1.0], [0.5, 0.0]]])
    grown = append_resin_layer(old, fill_value=6.3062, height=2, width=2)
    assert grown.shape == (2, 2, 2)
    assert torch.equal(grown[0], old[0])
    assert torch.allclose(grown[1], torch.full((2, 2), 6.3062))


def test_future_layers_are_absent_until_their_cycle() -> None:
    target = np.ones((3, 3, 3), dtype=bool)
    depths: list[int] = []
    cfg = PhysicsConfig(
        intensity_mw_cm2=1.0,
        exposure_s=0.1,
        dark_s=0.0,
        dt_s=0.05,
        oxygen_diffusivity_m2_s=0.0,
        xy_pitch_m=1.0,
        layer_height_m=1.0,
        initial_capacity_mj_cm2=2.0,
        blur_width_m=0.0,
        record_every_steps=1,
    )
    run_simulation(target, cfg, device="cpu", snapshot_callback=lambda s: depths.append(s.doc.shape[0]))
    assert depths == [1, 1, 2, 2, 3, 3]


def test_tensor_recorder_path_receives_growing_fields_without_full_snapshots() -> None:
    class TensorOnlyRecorder:
        def __init__(self) -> None:
            self.depths: list[int] = []

        def capture_tensors(self, doc: torch.Tensor, oxygen: torch.Tensor, **metadata: object) -> None:
            assert doc.device.type == "cpu"
            assert oxygen.shape == doc.shape
            assert metadata["phase"] == "exposure"
            self.depths.append(doc.shape[0])

        def __call__(self, snapshot: object) -> None:
            raise AssertionError("The full-snapshot callback path should not be used.")

    recorder = TensorOnlyRecorder()
    cfg = PhysicsConfig(
        intensity_mw_cm2=1.0, exposure_s=0.05, dark_s=0.0, dt_s=0.05,
        oxygen_diffusivity_m2_s=0.0, xy_pitch_m=1.0, layer_height_m=1.0,
        initial_capacity_mj_cm2=1.0, blur_width_m=0.0, record_every_steps=1,
    )
    run_simulation(np.ones((2, 2, 2), dtype=bool), cfg, "cpu", recorder)
    assert recorder.depths == [1, 2]


def test_no_flux_diffusion_conserves_total_capacity() -> None:
    field = torch.zeros((4, 5, 6), dtype=torch.float64)
    field[1, 2, 3] = 1.0
    out = diffuse_no_flux(field, diffusivity=0.1, dt=0.1, spacing=(1.0, 1.0, 1.0))
    assert out.sum().item() == pytest.approx(field.sum().item(), abs=1e-12)


def test_anisotropic_diffusion_and_stability_validation() -> None:
    field = torch.zeros((3, 3, 3), dtype=torch.float64)
    field[1, 1, 1] = 1.0
    out = diffuse_no_flux(field, diffusivity=0.1, dt=0.01, spacing=(4.0, 2.0, 1.0))
    assert out[1, 1, 2] > out[1, 2, 1] > out[2, 1, 1]
    limit = diffusion_stability_limit(0.1, spacing=(4.0, 2.0, 1.0))
    with pytest.raises(ValueError, match="stability"):
        diffuse_no_flux(field, diffusivity=0.1, dt=limit * 1.001, spacing=(4.0, 2.0, 1.0))


@pytest.mark.parametrize(
    ("oxygen", "intensity", "dt", "expected_oxygen", "expect_cure"),
    [
        (10.0, 2.0, 1.0, 8.0, False),
        (0.0, 2.0, 1.0, 0.0, True),
        (1.0, 2.0, 1.0, 0.0, True),
    ],
)
def test_reaction_handles_no_full_and_partial_depletion(
    oxygen: float, intensity: float, dt: float, expected_oxygen: float, expect_cure: bool
) -> None:
    o2 = torch.tensor([oxygen])
    doc = torch.zeros(1)
    cumulative = torch.zeros(1)
    next_o2, next_doc, _ = reaction_step(o2, doc, cumulative, torch.tensor([intensity]), dt)
    assert next_o2.item() == pytest.approx(expected_oxygen)
    assert (next_doc.item() > 0.015) is expect_cure


def test_variable_intensity_incremental_conversion_matches_hand_calculation() -> None:
    o2 = torch.zeros(1)
    doc = torch.zeros(1)
    cumulative = torch.zeros(1)
    for intensity, dt in [(4.0, 0.25), (9.0, 0.5)]:
        o2, doc, cumulative = reaction_step(
            o2, doc, cumulative, torch.tensor([intensity]), dt
        )
    exponent = 0.1131 * (4.0**0.6038) * 0.25 + 0.1131 * (9.0**0.6038) * 0.5
    assert doc.item() == pytest.approx(1.0 - math.exp(-exponent), rel=1e-6)


def test_partial_depletion_uses_only_residual_fraction_of_step_for_curing() -> None:
    oxygen, doc, _ = reaction_step(
        torch.tensor([1.0]), torch.zeros(1), torch.zeros(1), torch.tensor([2.0]), 1.0
    )
    expected = 1.0 - math.exp(-0.1131 * (2.0**0.6038) * 0.5)
    assert oxygen.item() == pytest.approx(0.0)
    assert doc.item() == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(
    ("axis", "index", "expected"),
    [("x", 3, (2, 3)), ("y", 2, (2, 4)), ("z", 1, (3, 4))],
)
def test_cross_section_indexing(axis: str, index: int, expected: tuple[int, int]) -> None:
    volume = np.zeros((2, 3, 4))
    assert extract_cross_section(volume, axis, index).shape == expected
    with pytest.raises(IndexError):
        extract_cross_section(volume, axis, 99)
    with pytest.raises(ValueError):
        extract_cross_section(volume, "q", 0)


def test_downsampling_does_not_modify_full_resolution_array() -> None:
    volume = np.arange(20 * 30 * 40, dtype=np.float32).reshape(20, 30, 40)
    before = volume.copy()
    reduced = downsample_volume(volume, max_dim=10)
    assert max(reduced.shape) <= 10
    assert np.array_equal(volume, before)
    assert not np.shares_memory(reduced, volume)


def test_volume_render_coordinates_preserve_anisotropic_spacing_after_downsampling() -> None:
    volume = np.zeros((4, 6, 8), dtype=np.float32)
    reduced, x, y, z = prepare_volume_render(volume, (5.0, 2.0, 1.0), max_dim=4)
    assert reduced.shape == (4, 3, 4)
    assert x[0, 0, 1] - x[0, 0, 0] == pytest.approx(2.0)
    assert y[0, 1, 0] - y[0, 0, 0] == pytest.approx(4.0)
    assert z[1, 0, 0] - z[0, 0, 0] == pytest.approx(5.0)


def test_tiny_cpu_workflow_writes_all_outputs(tmp_path: Path) -> None:
    target = np.ones((3, 8, 8), dtype=bool)
    cfg = PhysicsConfig(
        intensity_mw_cm2=100.0,
        exposure_s=0.2,
        dark_s=0.1,
        dt_s=0.1,
        oxygen_diffusivity_m2_s=0.0,
        xy_pitch_m=1.0,
        layer_height_m=1.0,
        initial_capacity_mj_cm2=0.0,
        blur_width_m=0.0,
        record_every_steps=1,
    )
    recorder = CrossSectionRecorder(
        tmp_path, axis="y", index=4, final_shape=target.shape, initial_capacity=1.0
    )
    result = run_simulation(target, cfg, device="cpu", snapshot_callback=recorder)
    recorder.finish()
    artifacts = save_final_outputs(
        tmp_path,
        target,
        result.doc,
        result.oxygen,
        summary={"device": "cpu"},
        threshold=0.01,
        spacing_m=(1.0, 1.0, 1.0),
        visualization_max_dim=8,
    )
    required = [
        tmp_path / "allDoC.gif",
        tmp_path / "allO2.gif",
        tmp_path / "final_fields.npz",
        tmp_path / "run_summary.json",
        tmp_path / "final_DoC_volume.html",
        tmp_path / "cured_isosurface.html",
        tmp_path / "cured_isosurface.stl",
    ]
    assert all(path.exists() and path.stat().st_size > 0 for path in required)
    assert list((tmp_path / "DoC_frames").glob("*.png"))
    assert list((tmp_path / "O2_frames").glob("*.png"))
    assert artifacts["surface_produced"] is True
    with np.load(tmp_path / "final_fields.npz") as saved:
        assert saved["doc"].shape == target.shape
        assert saved["target"].shape == target.shape
        assert saved["oxygen"].shape == target.shape
        assert np.array_equal(saved["oxygen"], result.oxygen)
    assert json.loads((tmp_path / "run_summary.json").read_text())["device"] == "cpu"


def test_no_threshold_crossing_preserves_non_surface_outputs(tmp_path: Path) -> None:
    target = np.zeros((2, 3, 4), dtype=bool)
    result = save_final_outputs(
        tmp_path,
        target,
        np.zeros_like(target, dtype=np.float32),
        np.ones_like(target, dtype=np.float32),
        summary={"case": "uncured"},
        threshold=0.77,
        spacing_m=(1.0, 1.0, 1.0),
        visualization_max_dim=4,
    )
    assert result["surface_produced"] is False
    assert (tmp_path / "final_fields.npz").exists()
    assert (tmp_path / "final_DoC_volume.html").exists()
    assert (tmp_path / "run_summary.json").exists()
    assert not (tmp_path / "cured_isosurface.stl").exists()
    assert not (tmp_path / "cured_isosurface.html").exists()
