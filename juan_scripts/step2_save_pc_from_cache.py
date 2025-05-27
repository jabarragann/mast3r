import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
import trimesh
from scipy.spatial.transform import Rotation
import imageio.v2

sys.path.append(str(Path(__file__).resolve().parent / ".."))
import pickle
from utils import save_pc_with_open3d, float32_arr, uint8_arr, bool_arr
from mast3r.cloud_opt.sparse_ga import SparseGA
from dust3r.utils.device import to_numpy
import open3d as o3d


OPENGL = np.array(
    [
        [1, 0, 0, 0],  # fmt: skip
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ]
)


def save_img(outfile: Path, img: np.ndarray):
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    imageio.v2.imwrite(str(outfile), img)


def save_pc(
    outfile: Path,
    imgs: List[float32_arr],
    pts3d: List[Tensor],
    mask: npt.NDArray,
    focals: Tensor,
    cams2world: Tensor,
    labels: Optional[np.ndarray] = None,
    labels_binary: Optional[np.ndarray] = None,
    save_intermediate_pc: bool = False,
):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)

    pts3d_np: float32_arr = to_numpy(pts3d)
    imgs_np: float32_arr = to_numpy(imgs)
    focals_np: float32_arr = to_numpy(focals)
    cams2world_np: float32_arr = to_numpy(cams2world)

    all_pts: List[float32_arr] = []
    all_col: List[float32_arr] = []
    all_labels: List[bool_arr] = []

    for i in range(len(imgs_np)):
        pi = pts3d_np[i]
        mi = mask[i]
        ci = imgs_np[i]

        if labels is not None and labels_binary is not None:
            li = labels[i]
            li_binary = labels_binary[i]
            all_labels.append(li_binary[mi].reshape(-1, 1))

        all_pts.append(pi[mi.ravel()].reshape(-1, 3))
        all_col.append(ci[mi].reshape(-1, 3))

        # save intermediate point clouds
        if save_intermediate_pc:
            path_frame_pc = outfile.parent / "frame_pc_vis"
            path_frame_pc.mkdir(exist_ok=True)

            # Return pc to camera frame
            T = np.linalg.inv(cams2world_np[i])
            pts_in_cam = (T[:3, :3] @ all_pts[-1].T + T[:3, 3:]).T

            outfile_i = path_frame_pc / f"{outfile.stem}_{i:03d}.ply"
            save_pc_with_open3d(outfile_i, pts_in_cam, all_col[-1])

            outfile_i = path_frame_pc / f"{outfile.stem}_{i:03d}.png"
            save_img(outfile_i, imgs_np[i])

        # save label image blend
        if labels is not None:
            path_label_vis = outfile.parent / "label_vis"
            outfile_i = path_label_vis / f"{outfile.stem}_{i:03d}_label.png"
            alpha = 0.5
            beta = 1.0 - alpha
            img_mask = cv2.addWeighted(imgs_np[i], alpha, li, beta, 0)
            save_img(outfile_i, img_mask)

        # save sample image points
        sampled_img = np.zeros_like(imgs_np[i])
        sampled_img[mi] = imgs_np[i][mi]
        outfile_i = outfile.parent / "sample_vis"
        outfile_i.mkdir(exist_ok=True)
        outfile_i = outfile_i / f"{outfile.stem}_{i:03d}_sampled.png"
        save_img(outfile_i, sampled_img)

    ## Original implementation
    # pts = np.concatenate([p[m.ravel()] for p, m in zip(pts3d, mask)]).reshape(-1, 3)
    # col = np.concatenate([p[m] for p, m in zip(imgs, mask)]).reshape(-1, 3)
    # valid_msk = np.isfinite(pts.sum(axis=1))

    pts = np.concatenate(all_pts, axis=0)
    col = np.concatenate(all_col, axis=0)

    save_pc_with_open3d(outfile, pts, col)

    # Filter points with label
    if labels is not None:
        labels = np.concatenate(all_labels, axis=0)
        liver_pts = pts[labels.ravel()]
        liver_col = col[labels.ravel()]
        save_pc_with_open3d(
            outfile.with_name("liver_intraoperative.ply"), liver_pts, liver_col
        )

    # Open3D
    # valid_msk = np.isfinite(pts.sum(axis=1))
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pts[valid_msk])
    # pcd.colors = o3d.utility.Vector3dVector(col[valid_msk])
    # # Save the point cloud to a file
    # o3d.io.write_point_cloud(str(outfile), pcd)

    # Save with Trimesh
    # scene = trimesh.Scene()
    # pct = trimesh.PointCloud(pts[valid_msk], colors=col[valid_msk])
    # scene.add_geometry(pct)
    # rot = np.eye(4)
    # rot[:3, :3] = Rotation.from_euler("y", np.deg2rad(180)).as_matrix()
    # scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    # print("(exporting 3D scene to", outfile, ")")
    # scene.export(file_obj=outfile)

    return outfile


def load_labels(width: int, height: int) -> Tuple[float32_arr, bool_arr]:
    # labels_path = Path(
    #     "/media/juan95/b0ad3209-9fa7-42e8-a070-b02947a78943/home/camma/JuanData/OvarianCancerDataset/video_16052/mast3r/labels"
    # )
    labels_path = Path(
        "/media/juan95/b0ad3209-9fa7-42e8-a070-b02947a78943/home/camma/JuanData/OvarianCancerDataset/video_16052/mast3r/mast3r_clip06/labels"
    )

    labels_paths = sorted(labels_path.glob("*.png"))

    all_labels: List[float32_arr] = []
    all_labels_binary: List[bool_arr] = []
    for i, label in enumerate(labels_paths):
        img: uint8_arr = imageio.v2.imread(label)

        resized_label, label_binary = process_labels(img, width, height)

        resized_label = np.expand_dims(resized_label, axis=0)
        label_binary = np.expand_dims(label_binary, axis=0)
        all_labels.append(resized_label)
        all_labels_binary.append(label_binary)

    labels = np.concatenate(all_labels, axis=0)
    labels_binary = np.concatenate(all_labels_binary, axis=0)

    return labels, labels_binary


def process_labels(
    label: uint8_arr, width: int, height: int
) -> Tuple[float32_arr, bool_arr]:
    label_resized = np.asarray(
        cv2.resize(
            label,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ),
        dtype=np.float32,
    )
    label_resized /= 255.0  # Normalize to [0, 1]

    _label_binary = cv2.cvtColor(label_resized, cv2.COLOR_RGB2GRAY)
    _, _label_binary = cv2.threshold(_label_binary, 0.1, 1.0, cv2.THRESH_BINARY)
    label_binary = _label_binary.astype(bool)

    return label_resized, label_binary


# Parameters
clean_depth = False
min_conf_thr = 1.5


def main():
    # Load scene state
    data_dir = Path("./juan_out/clip06")

    with open(data_dir / "scene_state.pkl", "rb") as f:
        scene_state = pickle.load(f)

    scene: SparseGA = scene_state.sparse_ga
    scene.modify_root_path_of_canon(data_dir)
    rgbimg: List[float32_arr] = scene.imgs  # type: ignore
    focals: Tensor = scene.get_focals().cpu()
    cams2world: Tensor = scene.get_im_poses().cpu()

    print(scene.intrinsics.shape)
    K: npt.NDArray[np.float32] = scene.intrinsics.detach().cpu().numpy()
    with data_dir.joinpath("intrinsics.json").open("w") as f:
        json.dump({"intrinsics": K[0].tolist()}, f, indent=4)

    width = rgbimg[0].shape[1]
    height = rgbimg[0].shape[0]
    labels = None
    labels, labels_binary = load_labels(width=width, height=height)

    pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=clean_depth))
    msk = to_numpy([c > min_conf_thr for c in confs])

    outfile = data_dir / "pc/full_reconstruction.ply"
    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile = save_pc(
        outfile,
        rgbimg,
        pts3d,
        msk,
        focals,
        cams2world,
        labels,
        labels_binary,
        save_intermediate_pc=False,
    )

    cams2world_np: float32_arr = cams2world.detach().cpu().numpy()
    np.save(data_dir / "pc/cams2world.npy", cams2world_np)

    print("finished")


if __name__ == "__main__":
    main()
