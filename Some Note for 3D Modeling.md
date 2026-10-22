# 3D Vat Photopolymerization Simulator v1

## Summary

Build a GPU-based, simplified bottom-up curing simulator derived from `AIE_TEMPOv1.1.py`. It will import a user-selected 3D mesh, create binary layer masks, simulate oxygen-limited curing through a growing 3D layer stack, and generate cross-section histories plus final volumetric and thresholded 3D results.

The implementation will use a clean runnable entry point with three small helper modules, not a packaged application:

- `AIE_3D_v1.py` — editable parameters and execution flow.
- `aie_3d_geometry.py` — mesh loading and voxelization.
- `aie_3d_physics.py` — optics, oxygen diffusion, and conversion.
- `aie_3d_outputs.py` — PNG/GIF and 3D output generation.
- `tests/test_aie_3d.py` — focused numerical and workflow tests.

## Configuration and Geometry

- Expose the following parameters near the top of `AIE_3D_v1.py`:
  - Required model filepath; accepted formats: STL, OBJ, PLY, GLB, and 3MF.
  - X/Y/Z rotations in degrees; file Z is the build axis.
  - `canvas_size=500`, `padding=50`, `xy_pitch=7.6e-6 m`.
  - Editable `layer_height`, defaulting to `50e-6 m`.
  - Cross-section axis and integer voxel index; default to the central Y plane.
  - Exposure, optical, oxygen, output, threshold, and device parameters listed below.
- Merge transformed scene components, require a nonempty watertight mesh, and fail rather than silently repairing it.
- Uniformly rescale the physical geometry so its largest XY extent fits within 400×400 pixels, center it in the 500×500 canvas, and preserve its complete XYZ aspect ratio.
- Derive Z-layer count from physical model height and `layer_height`. Apply no artificial Z limit or Z rescaling.
- Use anisotropic voxelization so XY and Z spacing remain physically correct.
- Slice from low to high model Z and create binary projection masks.

## Optical and Curing Simulation

- Use a growing object-coordinate stack:
  - Append one full 500×500 resin layer at each cycle.
  - Initialize it at the bulk oxygen-equivalent inhibition capacity.
  - Retain all prior layers’ oxygen and DoC states.
  - Do not allocate or evolve future layers prematurely.
- Default exposure schedule:
  - Intensity: `30 mW/cm²`.
  - Exposure: `6 s/layer`.
  - Dark lift and local gap-replacement interval: `2 s/layer`.
  - Physics step: `dt=0.05 s`.
- Apply the existing 2D Gaussian construction using a `50 µm` kernel width. Keep lateral blur constant with propagation depth.
- For current layer `k`, apply full focal-plane intensity. Older layer `j` receives:
  - `I_j = I_blurred × exp(-(k-j) × layer_height / Lz)`
  - `Lz` is editable and defaults to an explicitly uncalibrated `100 µm`.
- Use oxygen as the only inhibitor:
  - Reference inhibition energy: `12.6124 mJ/cm²`.
  - Reference thickness: `100 µm`.
  - Initial per-layer capacity: `12.6124 × layer_height / 100 µm`.
  - Oxygen diffusivity: `400×10⁻¹² m²/s`, uniform in liquid and cured regions.
  - Closed 3D reservoir with no-flux boundaries.
  - No TEMPO, air-interface influx, fluid flow, peel deformation, or conversion-dependent diffusivity.
- Replace cumulative-dose/current-intensity conversion with an incremental update:
  - Each step’s local dose first consumes oxygen-equivalent capacity.
  - Any residual dose cures for the corresponding fraction of that time step.
  - Retain `B(I)=0.1131·I^0.6038`, `pre_slope=0.5e-3`, and `pre_cap=0.015`.
  - Clamp inhibition capacity to nonnegative values and DoC to `[0,1]`.
- Implement anisotropic finite-difference diffusion and reject `dt` values above its stability limit.
- Default to CUDA. Estimate peak memory against 75% of currently free GPU memory and stop before simulation when unsafe. Do not silently rescale, fall back to CPU, or truncate Z. CPU remains available for small tests only.

## Outputs and Interfaces

- Write each run to `results/<model_name>_<timestamp>/`.
- Record the selected cross-section every two steps by default, reproducing the 2D code’s 0.10 s output cadence:
  - Separate DoC and oxygen-capacity PNG folders.
  - `allDoC.gif` and `allO2.gif`.
  - Fixed color ranges of `[0,1]` for DoC and `[0, initial_capacity]` for oxygen capacity.
  - Include elapsed time, exposure/dark state, and layer index on every frame.
- Save full-resolution numeric results:
  - Target voxel volume.
  - Final DoC and oxygen-capacity fields.
  - JSON summary containing inputs, derived dimensions, assumptions, device, memory estimate, and runtime.
- Generate both requested final 3D representations:
  - Interactive volumetric DoC HTML, downsampled only for browser rendering.
  - Cured isosurface HTML and STL using an editable threshold defaulting to `0.77`.
- If no voxel reaches the threshold, preserve all numeric and volumetric outputs and report that no surface was produced.
- Do not add geometry-accuracy metrics or grayscale optimization in v1.

## Test and Review Plan

- Geometry tests:
  - Load all five supported formats.
  - Preserve cube and tall-box aspect ratios.
  - Verify 500×500 canvas placement and 50-pixel XY padding.
  - Reject empty, unsupported, and non-watertight models.
- Physics tests:
  - Verify 50 µm layers produce `6.3062 mJ/cm²` initial inhibition capacity.
  - Verify exponential attenuation between current and older layers.
  - Verify future layers do not evolve before being appended.
  - Verify new layers enter at bulk capacity while prior states persist.
  - Confirm no-flux diffusion conserves total capacity without reaction.
  - Confirm anisotropic diffusion and stability validation.
  - Test full, zero, and partial within-step oxygen depletion.
  - Verify variable-intensity incremental conversion against hand-calculated cases.
- Output tests:
  - Validate X/Y/Z cross-section indexing and out-of-range errors.
  - Run a tiny CPU smoke simulation that produces PNGs, GIFs, numeric fields, HTML, and STL.
  - Confirm visualization downsampling does not change saved full-resolution arrays.
- Run the complete tests, a representative GPU smoke run, and a focused regression/robustness review. Use an independent review subagent where appropriate, without allowing it to overwrite unrelated work.
