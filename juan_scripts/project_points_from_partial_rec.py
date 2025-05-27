import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import open3d as o3d
from imageio.v2 import imread
from utils import (
    create_img_from_projected_pc,
    project_points_2d,
    project_points_3d,
    save_pc_with_open3d,
)


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
