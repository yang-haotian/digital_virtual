# -*- coding: utf-8 -*-
# @Time    : 2024/8/18 17:00
# @Project : AnimateMaster
# @FileName: process_youtube_videos.py
import os
import pdb

"""
python scripts/data_processing/process_youtube_videos.py \
data/youtube_man_0907/raw_videos
"""


def crop_videos(src_vdir):
    """
    对横屏视频进行裁剪
    :return:
    """
    from animate_master.common.crop_video import crop_video_by_line
    cnt = 0
    for sub_name in os.listdir(src_vdir):
        vsubdir = os.path.join(src_vdir, sub_name)
        for vname in os.listdir(vsubdir):
            if vname.endswith(".mp4"):
                vpath = os.path.join(vsubdir, vname)
                print(cnt, vpath)
                crop_video_by_line(vpath, None, detect_horizon=False, output_h=1280, output_w=720,
                                   delete_src_video=True)
                cnt += 1
    print(f"{src_vdir} have {cnt} videos")


def split_videos(src_vdir, save_dir):
    from animate_master.common.split_video import VideoSceneDetectSplitPipeline
    os.makedirs(save_dir, exist_ok=True)
    pipe_kwargs = {
        "models": {
            "yolo_detect": {
                "name": "YoloHumanDetectModel",
                "predict_type": "trt",
                "model_path": "./checkpoints/AnimateMaster/yolov10x.trt"
            }
        }
    }
    pipe = VideoSceneDetectSplitPipeline(**pipe_kwargs)
    cnt = 0
    split_video_set = set()
    for vname_ in os.listdir(save_dir):
        split_video_set.add(vname_.split("-")[0])
    for sub_name in os.listdir(src_vdir):
        vsubdir = os.path.join(src_vdir, sub_name)
        for vname in os.listdir(vsubdir):
            if vname.endswith(".mp4"):
                # 如果之前split过就不要再分割了
                if os.path.splitext(vname)[0] in split_video_set:
                    continue
                vpath = os.path.join(vsubdir, vname)
                print(cnt, vpath)
                scene_video_list = pipe.run(vpath, save_dir=save_dir, min_duration=2, threshold=20)
                cnt += 1
                print(scene_video_list)


def draw_openpose_videos(src_vdir, save_dir):
    from animate_master.common.draw_video import DrawOpenposeVideoPipeline
    os.makedirs(save_dir, exist_ok=True)
    pipe_kwargs = {
        "models": {
            "det_model": {
                "name": "YoloHumanDetectModel",
                "predict_type": "trt",
                "model_path": "./checkpoints/AnimateMaster/yolov10x.trt"
            },
            "pose_model": {
                "name": "RTMWBodyPose2dModel",
                "predict_type": "trt",
                "model_path": "./checkpoints/AnimateMaster/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.trt"
            }
        }
    }
    pipe = DrawOpenposeVideoPipeline(**pipe_kwargs)
    cnt = 0
    for vname in os.listdir(src_vdir):
        if vname.endswith(".mp4"):
            vpath = os.path.join(src_vdir, vname)
            pose_vpath = os.path.join(save_dir, vname)
            if os.path.exists(pose_vpath):
                continue
            print(cnt, vpath)
            pipe.run(vpath, save_dir=save_dir)
            cnt += 1


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process URLs from a txt file using yt-dlp.')
    parser.add_argument('raw_vdir', type=str, help='Raw video dir')

    args = parser.parse_args()
    raw_vdir = args.raw_vdir
    # crop_videos(raw_vdir)
    split_vdir = os.path.join(raw_vdir, "..", "videos")
    # split_videos(raw_vdir, split_vdir)
    openpose_vdir = os.path.join(raw_vdir, "..", "openpose")
    draw_openpose_videos(split_vdir, openpose_vdir)
