import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
import imageio.v2

sys.path.append(str(Path(__file__).resolve().parent / ".."))
import pickle
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

def save_pc_with_open3d(outfile: Path, pts: np.ndarray, colors: np.ndarray):

    valid_msk = np.isfinite(pts.sum(axis=1))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[valid_msk])
    pcd.colors = o3d.utility.Vector3dVector(colors[valid_msk])  
    # Save the point cloud to a file
    o3d.io.write_point_cloud(str(outfile), pcd)

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
):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    all_pts = []
    all_col = []
    for i in range(len(imgs)):
        pi = pts3d[i]
        mi = mask[i]
        ci = imgs[i]

        all_pts.append(pi[mi.ravel()].reshape(-1, 3))
        all_col.append(ci[mi].reshape(-1, 3))

        # save intermediate point clouds
        outfile_i = outfile.with_name(f"{outfile.stem}_{i:03d}.ply")
        save_pc_with_open3d(outfile_i, all_pts[-1], all_col[-1])
        outfile_i = outfile.with_name(f"{outfile.stem}_{i:03d}.png")
        save_img(outfile_i, imgs[i])

    ## Origianl implementation
    # pts = np.concatenate([p[m.ravel()] for p, m in zip(pts3d, mask)]).reshape(-1, 3)
    # col = np.concatenate([p[m] for p, m in zip(imgs, mask)]).reshape(-1, 3)
    # valid_msk = np.isfinite(pts.sum(axis=1))

    pts = np.concatenate(all_pts, axis=0)
    col = np.concatenate(all_col, axis=0)

    save_pc_with_open3d(outfile, pts, col)

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


# Parameters
clean_depth = False
min_conf_thr = 1.5


def main():
    # Load scene state
    with open("./juan_out/scene_state.pkl", "rb") as f:
        scene_state = pickle.load(f)

    scene: SparseGA = scene_state.sparse_ga
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()

    pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=clean_depth))

    msk = to_numpy([c > min_conf_thr for c in confs])

    outfile = Path("./juan_out/pc/juan_pc.ply")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile = save_pc(outfile, rgbimg, pts3d, msk, focals, cams2world)

    print("finished")


if __name__ == "__main__":
    main()
