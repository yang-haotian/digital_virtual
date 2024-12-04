# -*- coding: utf-8 -*-
# @Time    : 2024/8/17 14:12
# @Project : AnimateMaster
# @FileName: test_models.py
import os
import pdb
import cv2
import time
import numpy as np
from datetime import datetime


def test_yolo_human_detect_model(draw=True):
    """
    测试 YoloHumanDetectModel
    Returns:

    """
    from animate_master.infer_models.yolo_human_detect_model import YoloHumanDetectModel

    # tensorrt 模型加载
    det_kwargs = dict(
        predict_type="trt",
        model_path="./checkpoints/AnimateMaster/yolov10x.trt",
    )

    det_model = YoloHumanDetectModel(**det_kwargs)

    img_path = "assets/examples/ref_images/01.jpeg"
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    trt_rets = det_model.predict(img_rgb)

    # infer_times = []
    # for _ in range(100):
    #     t0 = time.time()
    #     trt_rets = det_model.predict(img_rgb)
    #     infer_times.append(time.time() - t0)
    # print("{} inference time: min: {}, max: {}, mean: {}".format(YoloHumanDetectModel.__name__, np.min(infer_times),
    #                                                              np.max(infer_times), np.mean(infer_times)))

    if draw:
        date_str = datetime.now().strftime("%m-%d-%H-%M")
        result_dir = "./results/{}-{}".format(YoloHumanDetectModel.__name__, date_str)
        os.makedirs(result_dir, exist_ok=True)

        for i, box in enumerate(trt_rets):
            img_bgr = cv2.rectangle(img_bgr, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 3)
        print(os.path.join(result_dir, os.path.basename(img_path)))
        cv2.imwrite(os.path.join(result_dir, os.path.basename(img_path)), img_bgr)


def test_rtmw_bodypose2d_model():
    from animate_master.infer_models.rtmw_bodypose2d_model import RTMWBodyPose2dModel
    from animate_master.infer_models.yolo_human_detect_model import YoloHumanDetectModel
    from animate_master.common import utils
    from animate_master.common import draw

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
    img_path = "assets/examples/ref_images/04.jpeg"
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]

    bbox = det_model.predict(img_rgb)
    bbox = bbox.tolist()[0]
    bbox = utils.xyxy2xywh(bbox)
    keypoints = pose_model.predict(img_rgb, bbox)

    # infer_times = []
    # for _ in range(100):
    #     t0 = time.time()
    #     trt_rets = pose_model.predict(img_rgb, bbox)
    #     infer_times.append(time.time() - t0)
    # print(
    #     "{} inference time: min: {}, max: {}, mean: {}".format(RTMWBodyPose2dModel.__name__, np.min(infer_times),
    #                                                            np.max(infer_times), np.mean(infer_times)))

    date_str = datetime.now().strftime("%m-%d-%H-%M")
    result_dir = "./results/{}-{}".format(RTMWBodyPose2dModel.__name__, date_str)
    os.makedirs(result_dir, exist_ok=True)

    img_draw = draw.draw_pose_v2(keypoints, H, W, draw_foot=True)
    cv2.imwrite(os.path.join(result_dir, os.path.basename(img_path)), img_draw)
    print(os.path.join(result_dir, os.path.basename(img_path)))


def test_rtmw_with_handpose2d_model():
    from animate_master.infer_models.rtmw_bodypose2d_model import RTMWBodyPose2dModel
    from animate_master.infer_models.topdown_handpose2d_model import TopdownHandPose2dModel
    from animate_master.infer_models.yolo_human_detect_model import YoloHumanDetectModel
    from animate_master.common import utils
    from animate_master.common import draw

    det_kwargs = dict(
        predict_type="trt",
        model_path="./checkpoints/AnimateMaster/yolov10x.trt",
    )

    det_model = YoloHumanDetectModel(**det_kwargs)

    pose_kwargs = dict(
        predict_type="trt",
        model_path="./checkpoints/AnimateMaster/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.trt",
    )
    pose_model = RTMWBodyPose2dModel(**pose_kwargs)

    handpose_kwargs = dict(
        predict_type="trt",
        model_path="./checkpoints/AnimateMaster/hrw48_mix_train_0401_best_1x256x256x3.trt",
    )
    handpose_model = TopdownHandPose2dModel(**handpose_kwargs)

    img_path = "assets/examples/ref_images/img.png"
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]

    bbox = det_model.predict(img_rgb)
    bbox = bbox.tolist()[0]
    keypoints = pose_model.predict(img_rgb, utils.xyxy2xywh(bbox))[0]
    lhand_box, rhand_box = utils.get_hand_boxes_from_poses(keypoints, H, W, bbox)
    if lhand_box is not None:
        keypoints[92:112] = handpose_model.predict(img_rgb, utils.xyxy2xywh(lhand_box))[0, 1:]
    if rhand_box is not None:
        keypoints[113:133] = handpose_model.predict(img_rgb, utils.xyxy2xywh(rhand_box))[0, 1:]

    date_str = datetime.now().strftime("%m-%d-%H-%M")
    result_dir = "./results/{}-{}".format(TopdownHandPose2dModel.__name__, date_str)
    os.makedirs(result_dir, exist_ok=True)

    img_draw = draw.draw_pose_v2(keypoints[None], H, W, draw_foot=True)
    cv2.imwrite(os.path.join(result_dir, os.path.basename(img_path)), img_draw)
    print(os.path.join(result_dir, os.path.basename(img_path)))


if __name__ == '__main__':
    # test_yolo_human_detect_model()
    test_rtmw_bodypose2d_model()
    # test_rtmw_with_handpose2d_model()
