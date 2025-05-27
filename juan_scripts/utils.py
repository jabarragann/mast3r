import numpy as np
import numpy.typing as npt
from pathlib import Path
import open3d as o3d
from typing import TypeAlias

# Stubs
float32_arr: TypeAlias = npt.NDArray[np.float32]
uint8_arr: TypeAlias = npt.NDArray[np.uint8]
bool_arr: TypeAlias = npt.NDArray[np.bool_]

def save_pc_with_open3d(outfile: Path, pts: float32_arr, colors: float32_arr):
    valid_msk = np.isfinite(pts.sum(axis=1))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[valid_msk])
    pcd.colors = o3d.utility.Vector3dVector(colors[valid_msk])
    # Save the point cloud to a file
    o3d.io.write_point_cloud(str(outfile), pcd)

def project_points_3d(
    points_2d: float32_arr, depth: float32_arr, K: float32_arr
) -> float32_arr:
    
    assert K.shape == (3, 3), "K must be a 3x3 intrinsic camera matrix"     
    assert points_2d.shape[1] == 2, "points_2d must be Nx2 array"

    K_inv = np.linalg.inv(K)
    points_2d_hom = np.hstack(
        (points_2d, np.ones((points_2d.shape[0], 1), dtype=np.float32))
    )
    points_3d = (K_inv @ points_2d_hom.T) * depth
    return points_3d.T


def project_points_2d(points_3d: float32_arr, K: float32_arr) -> float32_arr:
    """
    points_3d: Nx3 array of 3D points in camera coordinates.
    K: 3x3 intrinsic camera matrix.
    """

    assert points_3d.shape[1] == 3, "points_3d must be Nx3 array"   
    assert K.shape == (3, 3), "K must be a 3x3 intrinsic camera matrix"

    points_2d = K @ points_3d.T
    points_2d = points_2d[:2, :] / points_2d[2, :]  # Normalize by the third coordinate
    points_2d += 0.5

    return points_2d.T


def create_img_from_projected_pc(
    points_2d: float32_arr, colors: uint8_arr, img_shape: tuple[int, ...]
) -> uint8_arr:
    u, v = points_2d[:, 0].astype(int), points_2d[:, 1].astype(int)
    W = img_shape[1]
    H = img_shape[0]

    # Filter out points that are outside the image bounds
    # Juan note: most of the points are falling outside the image bounds.
    mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[mask]
    v = v[mask]
    colors = colors[mask]

    # Assign colors. There might be overlapping points which are just overwritten
    rgb_img = np.zeros((H, W, 3), dtype=np.uint8)
    rgb_img[v, u] = colors

    return rgb_img