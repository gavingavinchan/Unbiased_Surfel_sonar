# Synthetic Sonar Dataset Guide

This guide documents the Dataset A (`A_clean`) workflow for synthetic sonar validation.

## Scope

- Dataset: analytic sphere in vacuum (`synthetic_sphere_A_clean`)
- Goal: validate end-to-end sonar training behavior with known geometry and deterministic data
- Primary scripts:
  - `scripts/generate_synthetic_sonar_dataset.py`
  - `debug_multiframe.py`
  - `scripts/eval_synthetic_sphere.py`

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

## Implementation Notes

- **Sonar image channel contract**: synthetic sonar PNGs are saved as 3-channel grayscale RGB. This avoids SSIM channel mismatch in the existing training path that expects 3 channels.
- **Pose modes**: use `sonar_equivalent` for the canonical Dataset A acceptance gate. `camera_with_extrinsic` is optional diagnostic mode for transform-path checks.
- **Evaluator fit mode**: `scripts/eval_synthetic_sphere.py` defaults to `--fit-mode gt_trimmed`. It uses GT-centered radial trimming before sphere fitting to reduce outlier bias in center estimation while still reporting full-cloud GT radial statistics.

## Commands Used in Practice

- Generator: `scripts/generate_synthetic_sonar_dataset.py`
- Training: `debug_multiframe.py` with `SONAR_DATASET_PATH` and `SONAR_FREEZE_SCALE=1`
- Evaluator: `scripts/eval_synthetic_sphere.py`
