#!/bin/bash
# Train sonar model and extract meshes at specific iterations
# Iterations: 1, 2, 5, 10, 20, 50, 100

set -e  # Exit on error

# Configuration
DATASET_PATH="/home/gavin/ros2_ws/outputs/session_2025-12-08_16-35-13_sonar_data_for_2dgs"
OUTPUT_DIR="./output/sonar_mesh_progression"
ITERATIONS="1 2 5 10 20 50 100"
MESH_RES=128  # Safe for laptop GPU

# Clear previous output
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "SONAR TRAINING WITH MESH EXTRACTION"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR"
echo "Iterations: $ITERATIONS"
echo "============================================================"

# Convert space-separated list to comma-separated for save_iterations
SAVE_ITERS=$(echo "$ITERATIONS" | tr ' ' ',')

# Step 1: Train with checkpoints at specific iterations
echo ""
echo ">>> STEP 1: Training (max 100 iterations)..."
python train.py \
    -s "$DATASET_PATH" \
    -m "$OUTPUT_DIR" \
    --sonar_mode \
    --sonar_images sonar \
    --data_device cpu \
    --resolution 2 \
    --iterations 100 \
    --save_iterations $ITERATIONS \
    --test_iterations 100 \
    --densify_until_iter 50 \
    --seed 42

echo ""
echo ">>> Training complete. Extracting meshes..."

# Step 2: Extract mesh for each iteration
for iter in $ITERATIONS; do
    echo ""
    echo ">>> Extracting mesh for iteration $iter..."
    
    # Check if checkpoint exists
    CKPT_PATH="$OUTPUT_DIR/point_cloud/iteration_${iter}/point_cloud.ply"
    if [ ! -f "$CKPT_PATH" ]; then
        echo "    WARNING: Checkpoint not found at $CKPT_PATH, skipping..."
        continue
    fi
    
    # Extract mesh
    python render.py \
        -m "$OUTPUT_DIR" \
        -s "$DATASET_PATH" \
        --iteration $iter \
        --skip_train --skip_test \
        --data_device cpu \
        --mesh_res $MESH_RES \
        --quiet
    
    # Rename mesh to include iteration number
    MESH_SRC="$OUTPUT_DIR/train/ours_${iter}/fuse.ply"
    MESH_DST="$OUTPUT_DIR/mesh_iter_$(printf '%03d' $iter).ply"
    
    if [ -f "$MESH_SRC" ]; then
        mv "$MESH_SRC" "$MESH_DST"
        echo "    Saved: $MESH_DST"
    else
        echo "    WARNING: Mesh not found at $MESH_SRC"
    fi
done

echo ""
echo "============================================================"
echo "COMPLETE"
echo "============================================================"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Generated meshes:"
ls -la "$OUTPUT_DIR"/mesh_iter_*.ply 2>/dev/null || echo "  (no meshes found)"
echo ""
echo "Point clouds saved at iterations:"
ls -d "$OUTPUT_DIR"/point_cloud/iteration_* 2>/dev/null | xargs -I{} basename {} || echo "  (none)"
