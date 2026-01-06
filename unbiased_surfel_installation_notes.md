# Unbiased Surfel installation notes

## What worked

### Forked install (unbiased_surfel_sonar)
This install was much faster with these notes.

From the fork repo root (`/home/gavin/Unbiased_Surfel_sonar`):
1) Create a new env:
   - `conda env create -f environment.yml -n unbiased_surfel_sonar`
   - `conda activate unbiased_surfel_sonar`
2) Install CUDA + PyTorch (CUDA 11.8):
   - `conda install -c nvidia cuda-nvcc=11.8`
   - `conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`
   - `conda install -c nvidia cuda-toolkit=11.8`
3) Set CUDA_HOME in this env and reactivate:
   - `conda env config vars set CUDA_HOME="$CONDA_PREFIX"`
   - `conda deactivate`
   - `conda activate unbiased_surfel_sonar`
4) Build the extensions:
   - `pip install ./submodules/diff-surfel-rasterization`
   - `pip install ./submodules/simple-knn`
5) Sanity check:
   - `nvcc --version`
   - `python - <<'PY'
import torch
print(torch.version.cuda, torch.cuda.is_available())
import diff_surfel_rasterization, simple_knn
print("extensions import OK")
PY`
6) If training fails with missing Python modules, do NOT install one-by-one.
   - Install the full pip set at once:
     - `python -m pip install --upgrade open3d==0.18.0 mediapy==1.1.1 lpips==0.1.4 scikit-image==0.21.0 tqdm==4.66.2 trimesh==4.3.2 plyfile==1.0.3 opencv-python==4.10.0.84 matplotlib`

### 1) Update the existing conda env
- The `unbiased_surfel` env already existed, so it was updated with:
  - `conda env update -f environment.yml --prune`

### 2) Install CUDA compiler tools in the env
- `nvcc` was missing. Installing it via conda worked:
  - `conda install -c nvidia cuda-nvcc=11.8`

### 3) Install CUDA-enabled PyTorch
- The env initially had CPU-only PyTorch (`torch.version.cuda=None`).
- Reinstalled with CUDA 11.8 support:
  - `conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`
- After this:
  - `torch.version.cuda = 11.8`
  - `torch.cuda.is_available = True`

### 4) Install CUDA headers and libraries (needed for build)
- The build failed with `cuda_runtime.h` missing.
- Installing the CUDA toolkit headers fixed it:
  - `conda install -c nvidia cuda-toolkit=11.8`

### 5) Build and install the CUDA extensions
- With CUDA headers present, both submodules built and installed:
  - `pip install ./submodules/diff-surfel-rasterization`
  - `pip install ./submodules/simple-knn`

## Notes

### CUDA environment variables
- Do not set `PATH` or `LD_LIBRARY_PATH` with `conda env config vars set`, because it overrides PATH and breaks activation scripts.
- Only set `CUDA_HOME` persistently:
  - `conda env config vars set CUDA_HOME="$CONDA_PREFIX"`
- If PATH ever gets broken, fix it in a new shell:
  - `exec -l bash`
  - `source ~/anaconda3/etc/profile.d/conda.sh`
  - `conda activate unbiased_surfel`

### PATH sanity check
- In a fresh shell, `command -v ls` should resolve and PATH should include `/usr/bin`.
- For CUDA: `which nvcc` should point at the active env (`.../envs/unbiased_surfel_sonar/bin/nvcc`).

### Submodule setup
- If `pip install ./submodules/...` fails with "not installable", init submodules:
  - `git submodule update --init --recursive`

### If nvcc comes from base
- The build will fail if nvcc resolves to `/home/gavin/anaconda3/bin/nvcc`.
- Fix by unsetting CUDA_HOME in base (if set) and ensuring the env sets it:
  - `conda activate base`
  - `conda env config vars unset CUDA_HOME`
  - `conda deactivate`
  - `conda activate unbiased_surfel_sonar`
  - `export CUDA_HOME="$CONDA_PREFIX"`

### If conda install fails with `InvalidSpec: cuda-compiler==12.6.2=0`
- This can happen due to a pinned `cuda-compiler` spec in the env.
- If `nvcc` already exists in the env and points to it, you can skip re-installing:
  - `which nvcc`
  - `nvcc --version`

## Current status
- `diff_surfel_rasterization` and `simple_knn` are installed successfully.
- CUDA headers are present at:
  - `$CONDA_PREFIX/include/cuda_runtime.h`

## Recorded commands (forked run)
Run from the fork repo root (`/home/gavin/Unbiased_Surfel_sonar`) so outputs stay under `./output`:

Training:
```
source ~/anaconda3/etc/profile.d/conda.sh
conda activate unbiased_surfel_sonar

python train.py \
  -s /home/gavin/datasets/ns_docker_data_input/cubePool_colmap_ws/dense/0 \
  --images images/images \
  -m ./output/cubePool_colmap_ws \
  --data_device cpu \
  --resolution 2
```

Rendering (mesh extraction):
```
source ~/anaconda3/etc/profile.d/conda.sh
conda activate unbiased_surfel_sonar

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nice -n 10 ionice -c2 -n7 \
python render.py \
  -m ./output/cubePool_colmap_ws \
  -s /home/gavin/datasets/ns_docker_data_input/cubePool_colmap_ws/dense/0 \
  --images images/images \
  --skip_train --skip_test \
  --data_device cpu \
  --resolution 2 \
  --mesh_res 896
```

## Known-good run checklist
- CWD is `/home/gavin/Unbiased_Surfel_sonar`
- Env is active: `conda activate unbiased_surfel_sonar`
- `which nvcc` points at `.../envs/unbiased_surfel_sonar/bin/nvcc`
- `nvcc --version` shows CUDA 11.8
- `python - <<'PY'` check shows `torch.version.cuda=11.8` and `torch.cuda.is_available=True`
- Full pip set installed (see step 6 in "Forked install")

## Training run that worked

Used this command to avoid CUDA OOM on an 8 GB GPU (downscale + CPU-backed images):

```
python train.py \
  -s /home/gavin/datasets/ns_docker_data_input/cubePool_colmap_ws/dense/0 \
  --images images/images \
  -m ./output/cubePool_colmap_ws \
  --data_device cpu \
  --resolution 2
```

Notes:
- This completed training successfully.
- The dataset used is the undistorted COLMAP output under `dense/0`.

## Mesh extraction that worked (higher quality)

Ran successfully with higher mesh resolution without crashing:

```
source ~/anaconda3/etc/profile.d/conda.sh
conda activate unbiased_surfel

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nice -n 10 ionice -c2 -n7 \
python render.py \
  -m ./output/cubePool_colmap_ws \
  -s /home/gavin/datasets/ns_docker_data_input/cubePool_colmap_ws/dense/0 \
  --images images/images \
  --skip_train --skip_test \
  --data_device cpu \
  --resolution 2 \
  --mesh_res 896
```
