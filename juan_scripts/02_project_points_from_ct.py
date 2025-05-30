import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import utils
from imageio.v2 import imread


def main():
    data_dir = Path("juan_out/clip06")

    # Load intrinsic
    with open(data_dir / "intrinsics.json", "r") as f:
        _intrinsic = json.load(f)
        K = np.array(_intrinsic["intrinsics"], dtype=np.float32)

    # Load extrinsic
    cam2world = np.load(data_dir / "pc/cams2world.npy")

    rgb_img: npt.NDArray[np.uint8] = imread(
        data_dir / "pc/frame_pc_vis/full_reconstruction_000.png"
    )

    ct_path = data_dir / "pc/aligned_workflow2/liver_preoperative.ply"
    pc_path = data_dir / "pc/liver_intraoperative_manual_clean.ply"
    video_points, video_colors = utils.load_pc_with_open3d(pc_path)
    video_colors_uint8 = (video_colors * 255).astype(np.uint8)

    ct_points, ct_colors = utils.load_pc_with_open3d(ct_path)
    ct_colors_uint8 = (ct_colors * 255).astype(np.uint8)

    # Transforms
    T_world2cam = np.linalg.inv(cam2world[0])
    # fmt: off
    T_world2ct = np.array([[-38.641998,-27.865482,-6.559163,-17.142851],
                         [28.342777,-38.789444,-2.185884,-58.964081],
                         [-4.023947,-5.622119,47.591110,-230.046738],
                         [0.000000,0.000000,0.000000,1.000000]], dtype=np.float32)
    # fmt: on
    T_ct2world = np.linalg.inv(T_world2ct)
    T_ct2cam = T_world2cam @ T_ct2world

    # Transform points
    video_points_in_cam = (T_world2cam[:3, :3] @ video_points.T + T_world2cam[:3, 3:]).T
    ct_points_in_cam = (T_ct2cam[:3, :3] @ ct_points.T + T_ct2cam[:3, 3:]).T

    # video_points_in_ct = (T_world2ct[:3, :3] @ video_points.T + T_world2ct[:3, 3:]).T
    # combined_points = np.vstack((video_points_in_ct, ct_points))
    # combined_colors = np.vstack((video_colors, ct_colors))
    # save_pc_with_open3d(
    #     data_dir / "temp/video_points_in_ct.ply",
    #     video_points_in_ct,
    #     video_colors,
    # )
    # save_pc_with_open3d(
    #     data_dir / "temp/combined_points_in_ct.ply",
    #     combined_points,
    #     combined_colors,
    # )

    points_2d = utils.project_points_2d(video_points_in_cam, K)
    points_2d_ct = utils.project_points_2d(ct_points_in_cam, K)

    rgb_from_video_pc = utils.create_img_from_projected_pc(
        points_2d, video_colors_uint8, rgb_img.shape
    )
    rgb_from_ct = utils.create_img_from_projected_pc(
        points_2d_ct, ct_colors_uint8, rgb_img.shape
    )

    # blended_img = cv2.addWeighted(rgb_img, 0.5, rgb_from_pc, 0.5, 0)

    blended_ct_video = utils.blend_image_and_mask(rgb_img, rgb_from_video_pc, alpha=0.5)
    blended_ct_img = utils.blend_image_and_mask(rgb_img, rgb_from_ct, alpha=0.5)

    fig, ax = plt.subplots(2, 3)
    ax: np.ndarray

    ax[0, 0].imshow(rgb_from_ct)
    ax[1, 0].imshow(rgb_from_video_pc)
    ax[0, 1].imshow(blended_ct_img)
    ax[1, 1].imshow(blended_ct_video)
    ax[0, 2].imshow(rgb_img)

    [a.axis("off") for a in ax.ravel()]
    plt.tight_layout()
    plt.show()

    print(type(rgb_img))
    print(rgb_img.dtype)

    # print(K)


if __name__ == "__main__":
    main()
