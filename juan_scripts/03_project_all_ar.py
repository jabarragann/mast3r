import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import utils
from imageio.v2 import imread, imwrite
from utils import float32_arr, uint8_arr


def get_rgb_frames(data_dir: Path) -> uint8_arr:
    file_list = sorted(data_dir.glob("*.png"))

    _all_rgb: List[uint8_arr] = []
    for f in file_list:
        rgb_img: uint8_arr = imread(f)
        rgb_img = np.expand_dims(rgb_img, axis=0)
        _all_rgb.append(rgb_img)

    all_rgb = np.concatenate(_all_rgb, axis=0)

    return all_rgb


@dataclass
class ProjectCTLiver:
    ct_points: float32_arr
    ct_colors: float32_arr
    K: float32_arr
    T_ct2world: float32_arr
    width: int
    height: int

    def __post_init__(self):
        self.ct_colors_uint8 = (self.ct_colors * 255).astype(np.uint8)

    def project_points(self, world2cam: float32_arr) -> uint8_arr:
        T_ct2cam = world2cam @ self.T_ct2world
        ct_points_in_cam = (T_ct2cam[:3, :3] @ self.ct_points.T + T_ct2cam[:3, 3:]).T

        points_2d_ct = utils.project_points_2d(ct_points_in_cam, self.K)
        rgb_with_ct_ar = utils.create_img_from_projected_pc(
            points_2d_ct, self.ct_colors_uint8, (self.height, self.width)
        )

        return rgb_with_ct_ar


def main():
    data_dir = Path("juan_out/clip06")
    out_dir = data_dir / "pc/ar_vis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load intrinsic
    with open(data_dir / "intrinsics.json", "r") as f:
        _intrinsic = json.load(f)
        K = np.array(_intrinsic["intrinsics"], dtype=np.float32)

    # Load extrinsic
    cam2world = np.load(data_dir / "pc/cams2world.npy")

    # Load CT to video
    T_world2ct = np.array(
        [
            [-38.641998, -27.865482, -6.559163, -17.142851],
            [28.342777, -38.789444, -2.185884, -58.964081],
            [-4.023947, -5.622119, 47.591110, -230.046738],
            [0.000000, 0.000000, 0.000000, 1.000000],
        ],
        dtype=np.float32,
    )
    T_ct2world = np.linalg.inv(T_world2ct)

    # Load images
    all_rgb = get_rgb_frames(data_dir / "pc/frame_pc_vis")

    # Load CT
    ct_path = data_dir / "pc/aligned_workflow2/liver_preoperative.ply"
    ct_points, ct_colors = utils.load_pc_with_open3d(ct_path)

    # AR projection
    ar_projection = ProjectCTLiver(
        ct_points,
        ct_colors,
        K,
        T_ct2world,
        width=all_rgb.shape[2],
        height=all_rgb.shape[1],
    )

    for i in range(all_rgb.shape[0]):
        rgb_img_i = all_rgb[i]
        cam2world_i = cam2world[i]
        world2cam_i = np.linalg.inv(cam2world_i)

        project_points = ar_projection.project_points(world2cam_i)
        blended_ct_video = utils.blend_image_and_mask(
            rgb_img_i, project_points, alpha=0.5
        )

        imwrite(
            out_dir / f"ar_frame_{i:03d}.png",
            blended_ct_video,
        )


if __name__ == "__main__":
    main()
