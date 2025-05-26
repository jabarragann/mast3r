import json
import numpy as np
import numpy.typing as npt
from pathlib import Path
from imageio.v2 import imread
import open3d as o3d
import matplotlib.pyplot as plt
from utils import float32_arr, uint8_arr, save_pc_with_open3d


def project_points_3d(
    points_2d: float32_arr, depth: float32_arr, K: float32_arr
) -> float32_arr:
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


def main():
    data_dir = Path("juan_out/clip06")

    with open(data_dir / "intrinsics.json", "r") as f:
        _intrinsic = json.load(f)
        K = np.array(_intrinsic["intrinsics"], dtype=np.float32)

    rgb_img: npt.NDArray[np.uint8] = imread(
        data_dir / "pc/frame_pc_vis/full_reconstruction_000.png"
    )
    pc_o3d = o3d.io.read_point_cloud(
        data_dir / "pc/frame_pc_vis/full_reconstruction_000.ply"
    )
    points_3d = np.asarray(pc_o3d.points, dtype=np.float32)
    colors_float = np.asarray(pc_o3d.colors, dtype=np.float32)
    colors_uint8 = (colors_float * 255).astype(np.uint8)

    points_2d = project_points_2d(points_3d, K)
    rgb_from_pc = create_img_from_projected_pc(points_2d, colors_uint8, rgb_img.shape)

    points_3d_from_2d = project_points_3d(points_2d, points_3d[:, 2], K)

    print("Points 3D from 2D shape:", points_3d_from_2d.shape)
    save_pc_with_open3d(
        data_dir / "test_reprojected_pc.ply",
        points_3d_from_2d,
        colors_float,
    )

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(rgb_from_pc)
    ax[1].imshow(rgb_img)
    plt.axis("off")
    plt.show()

    print(type(rgb_img))
    print(rgb_img.dtype)

    print(K)


if __name__ == "__main__":
    main()
