# Synthetic Sonar Dataset Guide

This guide documents Dataset A (`A_clean`, sphere) and Dataset C (`C_clean`, cube)
workflows for synthetic sonar validation.

## Current Dataset Inventory

Use this section as the source of truth for currently maintained synthetic datasets.

- `synthetic_sphere_A_clean` (`A_clean`, sphere in vacuum)
- `synthetic_cube_C_clean` (`C_clean`, cube in vacuum)

## Scope

- Datasets:
  - Dataset A: analytic sphere in vacuum (`synthetic_sphere_A_clean`)
  - Dataset C: analytic cube in vacuum (`synthetic_cube_C_clean`)
- Goal: validate end-to-end sonar training behavior with known geometry and deterministic data
- Primary scripts:
  - `scripts/generate_synthetic_sonar_dataset.py`
  - `debug_multiframe.py`
  - `scripts/eval_synthetic_sphere.py`
  - `scripts/eval_synthetic_cube.py`

## Environment

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate unbiased_surfel_sonar
```

## 1) Generate Dataset A_clean

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate unbiased_surfel_sonar && \
python scripts/generate_synthetic_sonar_dataset.py \
  --output-dir ./synthetic_datasets/synthetic_sphere_A_clean \
  --variant A_clean \
  --pose-mode sonar_equivalent \
  --num-frames 500 \
  --seed 42 \
  --elevation-samples 64 \
  --overwrite
```

Expected dataset outputs:

- `synthetic_datasets/synthetic_sphere_A_clean/sparse/0/cameras.txt`
- `synthetic_datasets/synthetic_sphere_A_clean/sparse/0/images.txt`
- `synthetic_datasets/synthetic_sphere_A_clean/sparse/0/points3D.txt`
- `synthetic_datasets/synthetic_sphere_A_clean/sonar/*.png`
- `synthetic_datasets/synthetic_sphere_A_clean/manifest.json`
- `synthetic_datasets/synthetic_sphere_A_clean/DATASET_SETTINGS.md`
- `synthetic_datasets/synthetic_sphere_A_clean/consistency_gate.json`
- `synthetic_datasets/synthetic_sphere_A_clean/consistency_backprojected_points.ply`

Backward-projection consistency gate pass criteria:

- no NaN/Inf
- median pixel round-trip error <= 1 px
- mean radial residual <= 0.15 m
- p95 radial residual <= 0.30 m
- fitted sphere center error <= 0.10 m

## 2) Run synthetic debug training

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate unbiased_surfel_sonar && \
SONAR_DATASET=synthetic_a_clean \
SONAR_DATASET_PATH=/home/gavin/Unbiased_Surfel_sonar/synthetic_datasets/synthetic_sphere_A_clean \
SONAR_OUTPUT_DIR=./output/debug_multiframe_synth_run1 \
SONAR_NUM_FRAMES=500 \
SONAR_STAGE2_ITERS=1000 \
SONAR_STAGE3_ITERS=1 \
SONAR_FREEZE_SCALE=1 \
python debug_multiframe.py
```

Expected behavior:

- run completes without crash
- final scale remains `1.000000`
- final training CSV and support metrics are written

## 3) Evaluate geometry quality

Evaluate final surfel cloud:

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate unbiased_surfel_sonar && \
python scripts/eval_synthetic_sphere.py \
  --reconstruction ./output/debug_multiframe_synth_run1/surfels_after_training.ply \
  --dataset-root ./synthetic_datasets/synthetic_sphere_A_clean \
  --output-dir ./output/debug_multiframe_synth_run1/eval_surfel \
  --fit-mode both
```

Outputs:

- `output/debug_multiframe_synth_run1/eval_surfel/sphere_eval.json`
- `output/debug_multiframe_synth_run1/eval_surfel/sphere_residual_hist.png`

Default pass thresholds:

- mean radial error <= 0.05 m
- p95 radial error <= 0.10 m
- center error <= 0.03 m

## 4) Reproducibility check

Run the same training command with a different output folder (for example `debug_multiframe_synth_run2`) and compare:

- `final_eval_train_frames.csv` aggregate means
- `eval_surfel/sphere_eval.json` metrics

Metric drift should be very small under fixed seed and identical config.

## 5) One-command full gate runner

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate unbiased_surfel_sonar && \
python scripts/run_synthetic_a_gate.py \
  --dataset-root ./synthetic_datasets/synthetic_sphere_A_clean \
  --pose-mode sonar_equivalent \
  --num-frames 500 \
  --stage2-iters 1000 \
  --stage3-iters 1 \
  --overwrite-runs
```

Summary artifacts:

- `output/debug_multiframe_synth_gate_summary.json`
- `output/debug_multiframe_synth_gate_summary.md`

## 6) Dataset C (cube) quickstart

Generate Dataset C with multi-band pose coverage:

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate unbiased_surfel_sonar && \
python scripts/generate_synthetic_sonar_dataset.py \
  --output-dir ./synthetic_datasets/synthetic_cube_C_clean \
  --variant C_clean \
  --pose-mode sonar_equivalent \
  --pose-policy multi_band \
  --num-frames 500 \
  --seed 42 \
  --elevation-samples 64 \
  --overwrite
```

Train on Dataset C:

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate unbiased_surfel_sonar && \
SONAR_DATASET=synthetic_c_clean \
SONAR_DATASET_PATH=/home/gavin/Unbiased_Surfel_sonar/synthetic_datasets/synthetic_cube_C_clean \
SONAR_OUTPUT_DIR=./output/debug_multiframe_synth_c_run1 \
SONAR_NUM_FRAMES=500 \
SONAR_STAGE2_ITERS=1000 \
SONAR_STAGE3_ITERS=1 \
SONAR_FREEZE_SCALE=1 \
python debug_multiframe.py
```

Evaluate cube geometry:

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate unbiased_surfel_sonar && \
python scripts/eval_synthetic_cube.py \
  --reconstruction ./output/debug_multiframe_synth_c_run1/surfels_after_training.ply \
  --dataset-root ./synthetic_datasets/synthetic_cube_C_clean \
  --output-dir ./output/debug_multiframe_synth_c_run1/eval_surfel \
  --fit-mode both
```

One-command Dataset C gate:

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate unbiased_surfel_sonar && \
python scripts/run_synthetic_c_gate.py \
  --dataset-root ./synthetic_datasets/synthetic_cube_C_clean \
  --pose-mode sonar_equivalent \
  --pose-policy multi_band \
  --num-frames 500 \
  --stage2-iters 1000 \
  --stage3-iters 1 \
  --overwrite-runs
```

Dataset C default pass thresholds:

- mean surface error <= 0.05 m
- p95 surface error <= 0.10 m
- center error <= 0.03 m

## Renderer Semantic Fingerprint Policy (v2)

Once renderer-v2 semantics are introduced for a run, pre-v2 synthetic gate claims become historical-only for cross-run comparison.

For every synthetic gate summary used in Chunk 4 / renderer-remediation / Chunk 5 decisions, record at minimum:

- `renderer_semantics_version`
- `render_sonar_contract_hash` (or equivalent commit SHA)
- `normal_init_mode`
- `lambertian_transfer`
- `sonar_render_mode`
- `sonar_occlusion_mode`
- `occlusion_space` (active runs must report `ray_binned`)
- `occlusion_footprint_policy`
- `occlusion_support_cap_config`
- `occlusion_support_cap_mass_loss`
- `compat_reference_id` for frozen v2 off-mode comparisons
- `surfel_size_stats_schema_version`
- `elevation_bin_count`, `elevation_bin_policy`, `elevation_weight_mode`
- `sigma_point_config` and `sigma_point_fallback_fraction` when `sonar_render_mode=2dgs_nonlinear`

Comparator rule:

- Cross-run deltas are valid only when renderer semantic fingerprints match.
- The frozen v2 compatibility reference uses:
  - `SONAR_RENDER_MODE=2dgs`
  - `SONAR_OCCLUSION_MODE=none`
  - `SONAR_LAMBERTIAN_MODE=clamp0`

Current governance note:

- Renderer remediation was implemented after Chunk-4 investigation, but post-v2 active-path validation / synthetic re-baselining remains a separate gate before Chunk 5 is interpreted as current.

## Implementation Notes

- **Sonar image channel contract**: synthetic sonar PNGs are saved as 3-channel grayscale RGB. This avoids SSIM channel mismatch in the existing training path that expects 3 channels.
- **Pose modes**: use `sonar_equivalent` for the canonical Dataset A acceptance gate. `camera_with_extrinsic` is optional diagnostic mode for transform-path checks.
- **Pose policy**: Dataset C defaults to multi-band coverage (`--pose-policy auto` resolves to `multi_band`) to avoid equatorial-only coverage bias.
- **Evaluator fit mode**: `scripts/eval_synthetic_sphere.py` and `scripts/eval_synthetic_cube.py` default to `--fit-mode gt_trimmed` for robust center estimation while still reporting full-cloud GT residual statistics.

## Commands Used in Practice

- Generator: `scripts/generate_synthetic_sonar_dataset.py`
- Training: `debug_multiframe.py` with `SONAR_DATASET_PATH` and `SONAR_FREEZE_SCALE=1`
- Evaluator: `scripts/eval_synthetic_sphere.py`
- Evaluator (cube): `scripts/eval_synthetic_cube.py`
