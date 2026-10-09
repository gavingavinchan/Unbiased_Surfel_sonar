# Frozen echo-support extraction diagnostic (007)

These standalone scripts implement the permitted SonarSplat fallback when the
004 convention gate fails and005/006 supply no validated oriented checkpoint.
They do not certify the native surfel renderer or invent normals for3D Gaussians.

From the retained007 experiment directory and environment:

```bash
scripts/python.sh fork/scripts/echo_support_extraction/support_filter.py --config configs/extraction.json --out-dir outputs/support
scripts/python.sh fork/scripts/echo_support_extraction/extract_supported.py --config configs/extraction.json --masks outputs/support/masks.npz --population support3 --out-dir outputs/support3
```

Dependencies: NumPy/SciPy/Plyfile for support; PyTorch CUDA/Open3D/Trimesh/Matplotlib
and the pinned SonarSplat gsplat covariance helper for extraction. The existing
013 environment is reused read-only. The unchanged meshing function bodies come
from `/home/gavin/sonar-recon/third_party/sonar_splat/scripts/mesh_gaussian.py`
(commit8a46a2a11e4be08a1cf4bb76ee51aed29ee94dd3); their hash is recorded.

The only conditional AST substitution replaces `valid_opacities >0.2` with the
measured support mask. Opacity weights, covariances, reflectance normalization,
density0.9, ten samples/Gaussian and smoothing remain unchanged. `all` disables
primitive rejection as an ablation; `opacity` reproduces013 byte-for-byte.

Explicit sim_cube_v1 metadata is essential: centre pixels, range origin0/span3m,
FLU world_T_sonar poses, no second mount,200range rows×256azimuth columns. Projection
integrates elevation membership; it never treats a polar pixel as a pinhole ray.
Training support counts72 greedily selected poses, pairwise translation≥0.10m
AND rotation≥5degrees; no validation frame or GT geometry enters selection.
Two settings (count≥2/count≥3), echo>.05, range±1bin and azimuth±2columns are fixed.

The original013 initialization included validation images. Validation projection
F1 and filtered image scores are transductive, not clean held-out evidence.
ROI bounds are evaluation-only and never passed to these scripts. Gaussian
covariance axes are not validated surface normals. No Poisson/TSDF route is used.
See007 RESULTS.md, configs/extraction.json and extraction_comparison.json for
negative surface results, resolution controls and exact reproducibility.

Additional standalone diagnostics retain the original protocol and separately
label equal-pitch (`common_pitch.py`), shared per-primitive sample draws
(`paired_samples.py --seed42/7`), null correspondence (`null_support.py`), density
stage survival (`density_trace.py`), exact triangle/edge metrics
(`score_extractions.py`), and no-training filtered rasterization
(`evaluate_filtered_images.py`). Inputs/outputs are relative to the007 experiment
working directory. `density_trace.py` asserts exact agreement with retained
original accepted point sets. Paired controls do not alter the originally
reproduced013 baseline. None changes support threshold choices usingGT scores.
