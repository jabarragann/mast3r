import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt
import trimesh
from scipy.spatial.transform import Rotation
import imageio.v2

sys.path.append(str(Path(__file__).resolve().parent / ".."))
import pickle
from utils import save_pc_with_open3d
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
    outfile,
    imgs,
    pts3d,
    mask,
    focals,
    cams2world,
    labels: Optional[np.ndarray] = None,
    save_intermediate_pc: bool = False,
):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    all_pts = []
    all_col = []
    all_labels = []
    for i in range(len(imgs)):
        pi = pts3d[i]
        mi = mask[i]
        ci = imgs[i]

        # Label processing
        if labels is not None:
            li = labels[i]
            li = cv2.resize(
                li,
                (imgs[i].shape[1], imgs[i].shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            li_binary = cv2.cvtColor(li, cv2.COLOR_RGB2GRAY)
            _, li_binary = cv2.threshold(li_binary, 0.1, 1.0, cv2.THRESH_BINARY)
            li_binary = li_binary.astype(bool)

            all_labels.append(li_binary[mi].reshape(-1, 1))

        all_pts.append(pi[mi.ravel()].reshape(-1, 3))
        all_col.append(ci[mi].reshape(-1, 3))

        # save intermediate point clouds
        if save_intermediate_pc:
            path_frame_pc = outfile.parent / "frame_pc_vis"
            path_frame_pc.mkdir(exist_ok=True)

            outfile_i = path_frame_pc / f"{outfile.stem}_{i:03d}.ply"
            save_pc_with_open3d(outfile_i, all_pts[-1], all_col[-1])

            outfile_i = path_frame_pc / f"{outfile.stem}_{i:03d}.png"
            save_img(outfile_i, imgs[i])

        # save label image blend
        if labels is not None:
            path_label_vis = outfile.parent / "label_vis"
            outfile_i = path_label_vis / f"{outfile.stem}_{i:03d}_label.png"
            alpha = 0.5
            beta = 1.0 - alpha
            img_mask = cv2.addWeighted(imgs[i], alpha, li, beta, 0)
            save_img(outfile_i, img_mask)

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


def load_labels() -> np.ndarray:
    # labels_path = Path(
    #     "/media/juan95/b0ad3209-9fa7-42e8-a070-b02947a78943/home/camma/JuanData/OvarianCancerDataset/video_16052/mast3r/labels"
    # )
    labels_path = Path(
        "/media/juan95/b0ad3209-9fa7-42e8-a070-b02947a78943/home/camma/JuanData/OvarianCancerDataset/video_16052/mast3r/mast3r_clip06/labels"
    )

    labels = sorted(labels_path.glob("*.png"))

    all_labels = []
    for i, label in enumerate(labels):
        img = imageio.v2.imread(label)
        img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
        all_labels.append(img)

    all_labels = np.concatenate(all_labels, axis=0)
    return all_labels


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
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()

    print(scene.intrinsics.shape)
    K: npt.NDArray[np.float32] = scene.intrinsics.detach().cpu().numpy()
    with data_dir.joinpath("intrinsics.json").open("w") as f:
        json.dump({"intrinsics": K[0].tolist()}, f, indent=4)

    labels = None
    labels = load_labels()

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
        save_intermediate_pc=False,
    )

    print("finished")


if __name__ == "__main__":
    main()
