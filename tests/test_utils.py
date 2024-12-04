# -*- coding: utf-8 -*-
# @Time    : 2024/9/7 11:38
# @Project : AnimateMaster
# @FileName: test_utils.py
import pdb
import cv2
import os

from animate_master.infer_models.rtmw_bodypose2d_model import RTMWBodyPose2dModel
from animate_master.infer_models.yolo_human_detect_model import YoloHumanDetectModel

det_kwargs = dict(
    predict_type="trt",
    model_path="./checkpoints/AnimateMaster/yolov10x.trt",
)

det_model = YoloHumanDetectModel(**det_kwargs)

# tensorrt 模型加载
pose_kwargs = dict(
    predict_type="trt",
    model_path="./checkpoints/AnimateMaster/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.trt",
)
pose_model = RTMWBodyPose2dModel(**pose_kwargs)


def test_preprocess_openpose(ref_img_path, src_vpath):
    import decord
    from animate_master.common import preprocess

    ref_pose = preprocess.preprocess_openpose(src_vpath, ref_img_path, draw_foot=True, draw_hand=True, draw_face=False,
                                              to_rgb=False,
                                              score_thred=0.3, det_model=det_model, pose_model=pose_model)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    H, W = ref_pose.shape[1:3]
    save_vpath = src_vpath[:-4] + "-pose.mp4"
    decord_src = decord.VideoReader(src_vpath)
    vfps = decord_src.get_avg_fps()
    vcom = cv2.VideoWriter(save_vpath, fourcc, vfps, (W, H))

    for i in range(ref_pose.shape[0]):
        vcom.write(ref_pose[i])
    vcom.release()
    print(save_vpath)

    ref_pose = preprocess.preprocess_openpose(src_vpath, ref_img_path, draw_foot=False, draw_hand=True, draw_face=True,
                                              to_rgb=False,
                                              score_thred=0.3, det_model=det_model, pose_model=pose_model)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    H, W = ref_pose.shape[1:3]
    save_vpath = src_vpath[:-4] + "-mimic-pose.mp4"
    decord_src = decord.VideoReader(src_vpath)
    vfps = decord_src.get_avg_fps()
    vcom = cv2.VideoWriter(save_vpath, fourcc, vfps, (W, H))

    for i in range(ref_pose.shape[0]):
        vcom.write(ref_pose[i])
    vcom.release()
    print(save_vpath)


def test_preprocess_openpose_v2(ref_img_path, src_vpath):
    import decord
    from animate_master.common import preprocess

    ref_pose = preprocess.preprocess_openpose_v2(src_vpath, ref_img_path, draw_foot=True, draw_hand=True,
                                                 draw_face=False,
                                                 to_rgb=False,
                                                 score_thred=0.3, det_model=det_model, pose_model=pose_model)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    H, W = ref_pose.shape[1:3]
    save_vpath = src_vpath[:-4] + "-pose.mp4"
    decord_src = decord.VideoReader(src_vpath)
    vfps = decord_src.get_avg_fps()
    vcom = cv2.VideoWriter(save_vpath, fourcc, vfps, (W, H))

    for i in range(ref_pose.shape[0]):
        vcom.write(ref_pose[i])
    vcom.release()
    print(save_vpath)

    ref_pose = preprocess.preprocess_openpose_v2(src_vpath, ref_img_path, draw_foot=False, draw_hand=True,
                                                 draw_face=True,
                                                 to_rgb=False,
                                                 score_thred=0.3, det_model=det_model, pose_model=pose_model)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    H, W = ref_pose.shape[1:3]
    save_vpath = src_vpath[:-4] + "-mimic-pose.mp4"
    decord_src = decord.VideoReader(src_vpath)
    vfps = decord_src.get_avg_fps()
    vcom = cv2.VideoWriter(save_vpath, fourcc, vfps, (W, H))

    for i in range(ref_pose.shape[0]):
        vcom.write(ref_pose[i])
    vcom.release()
    print(save_vpath)


def test_preprocess_openpose_image(ref_img_path):
    from animate_master.common import preprocess

    ref_pose = preprocess.preprocess_openpose_image(ref_img_path, draw_foot=True, draw_hand=False, draw_face=False,
                                                    to_rgb=False, score_thred=0.3, det_model=det_model,
                                                    pose_model=pose_model)
    img_save_path = ref_img_path[:-4] + "-pose.png"
    cv2.imwrite(img_save_path, ref_pose)
    print(img_save_path)
    ref_pose = preprocess.preprocess_openpose_image(ref_img_path, draw_foot=False, draw_hand=False, draw_face=True,
                                                    to_rgb=False, score_thred=0.3, det_model=det_model,
                                                    pose_model=pose_model)
    img_save_path = ref_img_path[:-4] + "-mimic-pose.png"
    cv2.imwrite(img_save_path, ref_pose)
    print(img_save_path)


if __name__ == '__main__':
    ref_img_path = "data/tuokouxiu/demo6/img.png"
    src_vpath = "data/tuokouxiu/demo6/001.mp4"
    test_preprocess_openpose(ref_img_path, src_vpath)
    test_preprocess_openpose_image(ref_img_path)
