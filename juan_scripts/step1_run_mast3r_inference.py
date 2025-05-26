import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / ".."))
from mast3r.demo import get_reconstructed_scene
from mast3r.model import AsymmetricMASt3R


def main():
    weights_path = "/media/juan95/b0ad3209-9fa7-42e8-a070-b02947a78943/home/camma/JuanDocuments/Mast3r_checkpoints/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    # img_path = Path(
    #     "/media/juan95/b0ad3209-9fa7-42e8-a070-b02947a78943/home/camma/JuanData/OvarianCancerDataset/video_16052/mast3r/img_subset"
    # )

    img_path = Path(
        "/media/juan95/b0ad3209-9fa7-42e8-a070-b02947a78943/home/camma/JuanData/OvarianCancerDataset/video_16052/clip06_frames"
    )
    outdir = Path("./juan_out")
    img_list = list(map(str, sorted(img_path.glob("*.png"))))

    img_list = img_list[::4]
 
    # Parameters
    optim_level = "refine+depth"
    lr1 = 0.07
    niter1 = 300
    lr2 = 0.01
    niter2 = 300
    min_conf_thr = 1.5
    matching_conf_thr = 0
    as_pointcloud = True
    mask_sky = False
    clean_depth = True
    transparent_cams = False
    cam_size = 0.2
    scenegraph_type = "swin" #"complete"
    winsize = 8 
    win_cyclic = False
    refid = 0
    TSDF_thresh = 0
    shared_intrinsics = True

    # Model and scenestate
    device = "cuda:0"
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(device)
    current_scene_state = None

    # Run the reconstruction
    scene_state, outfile = get_reconstructed_scene(
        outdir=outdir,
        gradio_delete_cache=None,
        model=model,
        retrieval_model=None,
        device=device,
        silent=False,
        image_size=512,
        current_scene_state=current_scene_state,
        filelist=img_list,
        optim_level=optim_level,
        lr1=lr1,
        niter1=niter1,
        lr2=lr2,
        niter2=niter2,
        min_conf_thr=min_conf_thr,
        matching_conf_thr=matching_conf_thr,
        as_pointcloud=as_pointcloud,
        mask_sky=mask_sky,
        clean_depth=clean_depth,
        transparent_cams=transparent_cams,
        cam_size=cam_size,
        scenegraph_type=scenegraph_type,
        winsize=winsize,
        win_cyclic=win_cyclic,
        refid=refid,
        TSDF_thresh=TSDF_thresh,
        shared_intrinsics=shared_intrinsics,
    )


if __name__ == "__main__":
    main()
