# Architecture Diagram (Commit 0d41037 and Earlier)

```mermaid
flowchart TB
  subgraph "Inputs"
    Colmap["COLMAP dataset"]
    Blender["Blender dataset"]
    Checkpoint["point_cloud.ply / checkpoints"]
  end

  subgraph "Data Loading"
    DatasetReaders["scene/dataset_readers.py"]
    CameraUtils["utils/camera_utils.py"]
    Cameras["scene/cameras.py"]
    Scene["scene.Scene"]
  end

  Colmap --> DatasetReaders
  Blender --> DatasetReaders
  DatasetReaders --> CameraUtils
  CameraUtils --> Cameras
  Cameras --> Scene
  Checkpoint --> Scene

  subgraph "Core Model"
    GaussianModel["scene/gaussian_model.GaussianModel"]
    SimpleKNN["submodules/simple-knn (distCUDA2)"]
  end
  Scene --> GaussianModel
  SimpleKNN --> GaussianModel

  subgraph "Rendering"
    Renderer["gaussian_renderer.render()"]
    Rasterizer["diff-surfel-rasterization"]
    PointUtils["utils/point_utils.py (depth_to_normal)"]
  end
  Cameras --> Renderer
  GaussianModel --> Renderer
  Renderer --> Rasterizer
  PointUtils --> Renderer

  subgraph "Training Loop"
    Train["train.py"]
    Losses["utils/loss_utils.py (SSIM + L1 + regularizers)"]
    Optimizer["Adam + densify/prune"]
  end
  Train --> Renderer
  Renderer --> Losses
  Losses --> Optimizer
  Optimizer --> GaussianModel

  subgraph "Mesh Extraction"
    Render["render.py"]
    GaussianExtractor["utils/mesh_utils.GaussianExtractor"]
    Open3D["Open3D TSDF fusion"]
    Mesh["mesh outputs (fuse.ply)"]
  end
  Render --> GaussianExtractor
  GaussianExtractor --> Renderer
  GaussianExtractor --> Open3D
  Open3D --> Mesh

  subgraph "Viewer"
    View["view.py"]
    NetworkGUI["gaussian_renderer/network_gui.py"]
  end
  View --> Renderer
  View <--> NetworkGUI

  subgraph "Evaluation"
    EvalScripts["scripts/*_eval.py"]
    Metrics["metrics.py"]
  end
  EvalScripts --> Metrics
```

- Scope matches the repository layout and module wiring present at commit `0d41037`.
- Sonar-specific modules are intentionally excluded because they are not part of that commit.
