#!/usr/bin/env python3
"""
Interpolate sonar poses from camera trajectory.

This script reads camera poses from a COLMAP images.bin file and interpolates
poses for sonar frames based on timestamp proximity. Sonar frames without
camera poses within the specified threshold are discarded.

Usage:
    python scripts/interpolate_sonar_poses.py \
        --camera_model /path/to/camera/sparse/0 \
        --sonar_images /path/to/sonar/images \
        --output_dir /path/to/output/sparse/0 \
        --threshold_ms 100
"""

import os
import sys
import struct
import shutil
import argparse
import bisect
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, NamedTuple
import numpy as np


# =============================================================================
# COLMAP Data Structures
# =============================================================================

class Image(NamedTuple):
    """COLMAP image data structure."""
    id: int
    qvec: np.ndarray  # Quaternion (w, x, y, z)
    tvec: np.ndarray  # Translation (x, y, z)
    camera_id: int
    name: str
    xys: np.ndarray  # 2D points
    point3D_ids: np.ndarray  # Corresponding 3D point IDs


# =============================================================================
# Binary I/O Helpers
# =============================================================================

def read_next_bytes(fid, num_bytes: int, format_char_sequence: str, 
                    endian_character: str = "<") -> tuple:
    """Read and unpack bytes from a binary file."""
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def write_next_bytes(fid, values: tuple, format_char_sequence: str,
                     endian_character: str = "<") -> None:
    """Pack and write bytes to a binary file."""
    data = struct.pack(endian_character + format_char_sequence, *values)
    fid.write(data)


# =============================================================================
# COLMAP Binary Readers
# =============================================================================

def read_images_binary(path: str) -> Dict[int, Image]:
    """
    Read COLMAP images.bin file.
    
    Returns:
        Dictionary mapping image_id to Image namedtuple
    """
    images = {}
    with open(path, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            if num_points2D > 0:
                x_y_id_s = read_next_bytes(
                    fid, num_bytes=24 * num_points2D,
                    format_char_sequence="ddq" * num_points2D)
                xys = np.column_stack([
                    tuple(map(float, x_y_id_s[0::3])),
                    tuple(map(float, x_y_id_s[1::3]))])
                point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            else:
                xys = np.zeros((0, 2))
                point3D_ids = np.zeros(0, dtype=np.int64)
            
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    
    return images


def write_images_binary(images: Dict[int, Image], path: str) -> None:
    """
    Write COLMAP images.bin file.
    
    Args:
        images: Dictionary mapping image_id to Image namedtuple
        path: Output file path
    """
    with open(path, "wb") as fid:
        write_next_bytes(fid, (len(images),), "Q")
        
        for image_id, image in images.items():
            write_next_bytes(fid, (
                image.id,
                image.qvec[0], image.qvec[1], image.qvec[2], image.qvec[3],
                image.tvec[0], image.tvec[1], image.tvec[2],
                image.camera_id
            ), "idddddddi")
            
            # Write image name as null-terminated string
            fid.write(image.name.encode("utf-8"))
            fid.write(b"\x00")
            
            # Write 2D points
            write_next_bytes(fid, (len(image.xys),), "Q")
            for i in range(len(image.xys)):
                write_next_bytes(fid, (
                    image.xys[i, 0],
                    image.xys[i, 1],
                    int(image.point3D_ids[i])
                ), "ddq")


# =============================================================================
# Quaternion Operations
# =============================================================================

def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """Normalize a quaternion to unit length."""
    return q / np.linalg.norm(q)


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation between two quaternions.
    
    Args:
        q0: Start quaternion (w, x, y, z)
        q1: End quaternion (w, x, y, z)
        t: Interpolation parameter [0, 1]
    
    Returns:
        Interpolated quaternion (w, x, y, z)
    """
    q0 = normalize_quaternion(q0)
    q1 = normalize_quaternion(q1)
    
    # Compute dot product
    dot = np.dot(q0, q1)
    
    # If dot product is negative, negate one quaternion to take shorter path
    if dot < 0:
        q1 = -q1
        dot = -dot
    
    # If quaternions are very close, use linear interpolation
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return normalize_quaternion(result)
    
    # SLERP formula
    theta_0 = np.arccos(dot)  # Angle between quaternions
    theta = theta_0 * t       # Interpolated angle
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    
    return s0 * q0 + s1 * q1


def lerp(v0: np.ndarray, v1: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation between two vectors."""
    return (1 - t) * v0 + t * v1


# =============================================================================
# Timestamp Extraction
# =============================================================================

def extract_timestamp(filename: str, prefix: str) -> Optional[int]:
    """
    Extract timestamp (milliseconds) from filename.
    
    Args:
        filename: Image filename (e.g., "camera_1765233408026.png")
        prefix: Expected prefix (e.g., "camera_" or "sonar_")
    
    Returns:
        Timestamp in milliseconds, or None if parsing fails
    """
    try:
        basename = os.path.basename(filename)
        name = os.path.splitext(basename)[0]
        if name.startswith(prefix):
            return int(name[len(prefix):])
    except (ValueError, IndexError):
        pass
    return None


# =============================================================================
# Pose Interpolation
# =============================================================================

def interpolate_pose(
    camera_poses: Dict[int, Tuple[np.ndarray, np.ndarray]],
    sorted_timestamps: List[int],
    target_ts: int,
    threshold_ms: int
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Interpolate pose for a target timestamp from camera trajectory.
    
    Args:
        camera_poses: Dict mapping timestamp to (qvec, tvec)
        sorted_timestamps: Sorted list of camera timestamps
        target_ts: Target timestamp to interpolate
        threshold_ms: Maximum allowed distance to nearest camera pose
    
    Returns:
        Tuple of (qvec, tvec) or None if no poses within threshold
    """
    if not sorted_timestamps:
        return None
    
    # Find insertion point
    idx = bisect.bisect_left(sorted_timestamps, target_ts)
    
    # Get candidate timestamps (before and after)
    candidates = []
    if idx > 0:
        candidates.append((sorted_timestamps[idx - 1], abs(target_ts - sorted_timestamps[idx - 1])))
    if idx < len(sorted_timestamps):
        candidates.append((sorted_timestamps[idx], abs(target_ts - sorted_timestamps[idx])))
    
    if not candidates:
        return None
    
    # Find nearest camera timestamp
    nearest_ts, nearest_dist = min(candidates, key=lambda x: x[1])
    
    # Check threshold
    if nearest_dist > threshold_ms:
        return None
    
    # If exact match or only one candidate, return that pose
    if nearest_dist == 0 or len(candidates) == 1:
        return camera_poses[nearest_ts]
    
    # Get bracketing timestamps for interpolation
    if idx == 0:
        # Target is before first camera frame - use first pose
        return camera_poses[sorted_timestamps[0]]
    elif idx >= len(sorted_timestamps):
        # Target is after last camera frame - use last pose
        return camera_poses[sorted_timestamps[-1]]
    else:
        # Interpolate between two poses
        t_before = sorted_timestamps[idx - 1]
        t_after = sorted_timestamps[idx]
        
        # Compute interpolation factor
        alpha = (target_ts - t_before) / (t_after - t_before)
        
        qvec_before, tvec_before = camera_poses[t_before]
        qvec_after, tvec_after = camera_poses[t_after]
        
        # Interpolate
        qvec_interp = slerp(qvec_before, qvec_after, alpha)
        tvec_interp = lerp(tvec_before, tvec_after, alpha)
        
        return qvec_interp, tvec_interp


# =============================================================================
# Main Processing
# =============================================================================

def process_sonar_poses(
    camera_model_dir: str,
    sonar_images_dir: str,
    output_dir: str,
    threshold_ms: int = 100,
    max_frames: int = -1,
    seed: int = 42
) -> dict:
    """
    Generate interpolated poses for sonar images.
    
    Args:
        camera_model_dir: Path to camera COLMAP model (sparse/0)
        sonar_images_dir: Path to sonar images directory
        output_dir: Output directory for sonar COLMAP model
        threshold_ms: Maximum time gap to nearest camera pose (ms)
        max_frames: Maximum number of frames to output (-1 for all)
        seed: Random seed for reproducible sampling
    
    Returns:
        Statistics dictionary
    """
    stats = {
        "total_camera_frames": 0,
        "total_sonar_frames": 0,
        "valid_sonar_frames": 0,
        "rejected_sonar_frames": 0,
        "sampled_frames": 0,
        "rejection_reasons": []
    }
    
    # Read camera images.bin
    camera_images_path = os.path.join(camera_model_dir, "images.bin")
    if not os.path.exists(camera_images_path):
        raise FileNotFoundError(f"Camera images.bin not found: {camera_images_path}")
    
    print(f"Reading camera poses from: {camera_images_path}")
    camera_images = read_images_binary(camera_images_path)
    stats["total_camera_frames"] = len(camera_images)
    print(f"  Found {len(camera_images)} camera frames")
    
    # Build camera pose lookup by timestamp
    camera_poses: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    camera_id_to_use = None  # We'll use the same camera_id for all sonar images
    
    for img in camera_images.values():
        ts = extract_timestamp(img.name, "camera_")
        if ts is not None:
            camera_poses[ts] = (img.qvec, img.tvec)
            if camera_id_to_use is None:
                camera_id_to_use = img.camera_id
    
    sorted_camera_ts = sorted(camera_poses.keys())
    print(f"  Camera timestamp range: {sorted_camera_ts[0]} - {sorted_camera_ts[-1]}")
    
    # Get sonar image files
    sonar_files = sorted([
        f for f in os.listdir(sonar_images_dir) 
        if f.endswith('.png') and f.startswith('sonar_')
    ])
    stats["total_sonar_frames"] = len(sonar_files)
    print(f"\nFound {len(sonar_files)} sonar images in: {sonar_images_dir}")
    
    # Extract sonar timestamps
    sonar_timestamps = []
    for f in sonar_files:
        ts = extract_timestamp(f, "sonar_")
        if ts is not None:
            sonar_timestamps.append((ts, f))
    
    print(f"  Sonar timestamp range: {sonar_timestamps[0][0]} - {sonar_timestamps[-1][0]}")
    
    # Interpolate poses for each sonar frame
    print(f"\nInterpolating poses (threshold: ±{threshold_ms}ms)...")
    sonar_images: Dict[int, Image] = {}
    image_id = 1
    
    for sonar_ts, sonar_filename in sonar_timestamps:
        result = interpolate_pose(
            camera_poses, sorted_camera_ts, sonar_ts, threshold_ms)
        
        if result is None:
            stats["rejected_sonar_frames"] += 1
            # Find actual distance for reporting
            idx = bisect.bisect_left(sorted_camera_ts, sonar_ts)
            dists = []
            if idx > 0:
                dists.append(abs(sonar_ts - sorted_camera_ts[idx - 1]))
            if idx < len(sorted_camera_ts):
                dists.append(abs(sonar_ts - sorted_camera_ts[idx]))
            min_dist = min(dists) if dists else float('inf')
            stats["rejection_reasons"].append((sonar_filename, min_dist))
        else:
            qvec, tvec = result
            sonar_images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id_to_use,
                name=sonar_filename,
                xys=np.zeros((0, 2)),  # No 2D-3D correspondences for sonar
                point3D_ids=np.zeros(0, dtype=np.int64)
            )
            image_id += 1
            stats["valid_sonar_frames"] += 1
    
    print(f"  Valid frames: {stats['valid_sonar_frames']}")
    print(f"  Rejected frames: {stats['rejected_sonar_frames']}")
    
    # Random sampling if max_frames is specified
    if max_frames > 0 and len(sonar_images) > max_frames:
        print(f"\nRandomly sampling {max_frames} frames from {len(sonar_images)} valid frames (seed={seed})...")
        random.seed(seed)
        sampled_ids = sorted(random.sample(list(sonar_images.keys()), max_frames))
        sonar_images_sampled = {}
        for new_id, old_id in enumerate(sampled_ids, start=1):
            img = sonar_images[old_id]
            # Re-assign sequential IDs
            sonar_images_sampled[new_id] = Image(
                id=new_id,
                qvec=img.qvec,
                tvec=img.tvec,
                camera_id=img.camera_id,
                name=img.name,
                xys=img.xys,
                point3D_ids=img.point3D_ids
            )
        sonar_images = sonar_images_sampled
        stats["sampled_frames"] = len(sonar_images)
        print(f"  Sampled {len(sonar_images)} frames")
    else:
        stats["sampled_frames"] = len(sonar_images)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Write sonar images.bin
    output_images_path = os.path.join(output_dir, "images.bin")
    print(f"\nWriting sonar poses to: {output_images_path}")
    write_images_binary(sonar_images, output_images_path)
    
    # Copy cameras.bin and points3D.bin
    for filename in ["cameras.bin", "points3D.bin"]:
        src = os.path.join(camera_model_dir, filename)
        dst = os.path.join(output_dir, filename)
        if os.path.exists(src):
            print(f"Copying {filename}...")
            shutil.copy2(src, dst)
        else:
            print(f"Warning: {filename} not found in camera model")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Interpolate sonar poses from camera trajectory")
    parser.add_argument(
        "--camera_model", required=True,
        help="Path to camera COLMAP model directory (containing images.bin)")
    parser.add_argument(
        "--sonar_images", required=True,
        help="Path to sonar images directory")
    parser.add_argument(
        "--output_dir", required=True,
        help="Output directory for sonar COLMAP model")
    parser.add_argument(
        "--threshold_ms", type=int, default=100,
        help="Maximum time gap to nearest camera pose in milliseconds (default: 100)")
    parser.add_argument(
        "--max_frames", type=int, default=-1,
        help="Maximum number of frames to output; randomly samples if exceeded (default: -1 for all)")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling (default: 42)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Sonar Pose Interpolation")
    print("=" * 60)
    print(f"Camera model: {args.camera_model}")
    print(f"Sonar images: {args.sonar_images}")
    print(f"Output dir:   {args.output_dir}")
    print(f"Threshold:    ±{args.threshold_ms}ms")
    print(f"Max frames:   {args.max_frames if args.max_frames > 0 else 'all'}")
    print(f"Seed:         {args.seed}")
    print("=" * 60)
    
    stats = process_sonar_poses(
        args.camera_model,
        args.sonar_images,
        args.output_dir,
        args.threshold_ms,
        args.max_frames,
        args.seed
    )
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Camera frames:        {stats['total_camera_frames']}")
    print(f"Total sonar frames:   {stats['total_sonar_frames']}")
    print(f"Valid sonar frames:   {stats['valid_sonar_frames']} ({100*stats['valid_sonar_frames']/stats['total_sonar_frames']:.1f}%)")
    print(f"Rejected frames:      {stats['rejected_sonar_frames']} ({100*stats['rejected_sonar_frames']/stats['total_sonar_frames']:.1f}%)")
    if stats['sampled_frames'] != stats['valid_sonar_frames']:
        print(f"Output frames:        {stats['sampled_frames']} (randomly sampled)")
    
    if stats["rejection_reasons"]:
        print(f"\nSample rejections (first 5):")
        for filename, dist in stats["rejection_reasons"][:5]:
            print(f"  {filename}: nearest camera {dist:.0f}ms away")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
