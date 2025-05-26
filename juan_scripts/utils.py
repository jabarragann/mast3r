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