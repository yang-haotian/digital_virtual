import pdb
import numpy as np
import cv2, os
import moviepy
import moviepy.video.io.ImageSequenceClip


def check_frame_count(src_vpath, pose_vpath):
    src_video = cv2.VideoCapture(src_vpath)
    pose_video = cv2.VideoCapture(pose_vpath)
    src_frame_count = int(src_video.get(cv2.CAP_PROP_FRAME_COUNT))
    pose_frame_count = int(pose_video.get(cv2.CAP_PROP_FRAME_COUNT))
    src_video.get(cv2.CAP_PROP_FPS)
    src_video.release()
    pose_video.release()
    frame = np.zeros(shape=(768, 512, 3)).astype(np.uint8)
    ## 小数帧会导致视频帧增多一帧
    kps_results = [frame] * 300
    fps = 29.97
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(kps_results, fps=fps)
    out_path = src_vpath + "-test.mp4"
    clip.write_videofile(out_path, fps=fps, codec='libx264')
    out_video = cv2.VideoCapture(out_path)
    out_frame_count = int(out_video.get(cv2.CAP_PROP_FRAME_COUNT))
    pdb.set_trace()


if __name__ == '__main__':
    # collect all video_folder paths
    src_vpath = "/root/yht/test/_J3LCIS3Vz4_segment_02.mp4"
    pose_vpath = "/root/yht/test_dwpose/_J3LCIS3Vz4_segment_02.mp4"
    check_frame_count(src_vpath, pose_vpath)
