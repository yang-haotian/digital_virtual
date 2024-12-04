# -*- coding: utf-8 -*-
# @Time    : 2024/8/24 12:10
# @Project : AnimateMaster
# @FileName: draw_video.py
import cv2
import numpy as np
import os
from tqdm import tqdm
import pickle
from .. import infer_models
from . import draw
from . import utils


class DrawOpenposeVideoPipeline:
    def __init__(self, **kwargs):
        self.init_models(**kwargs)

    def init_models(self, **kwargs):
        """
        初始化模型
        """
        models_kwargs = kwargs["models"]
        self.models_dict = {}
        for model_name in models_kwargs:
            model_kwargs_ = models_kwargs[model_name]
            self.models_dict[model_name] = getattr(infer_models, model_kwargs_["name"])(**model_kwargs_)

    def run(self, vpath, **kwargs):
        try:
            if not os.path.exists(vpath):
                return
            save_dir = kwargs.get("save_dir", os.path.dirname(vpath))
            os.makedirs(save_dir, exist_ok=True)

            vname = os.path.splitext(os.path.basename(vpath))[0]
            keypoints_pkl = os.path.join(save_dir, f"{vname}.pkl")
            keypoints_info = {}
            if os.path.exists(keypoints_pkl):
                with open(keypoints_pkl, "rb") as fin:
                    keypoints_info = pickle.load(fin)
            else:
                keypoints = []
                vcap = cv2.VideoCapture(vpath)
                fps = int(vcap.get(cv2.CAP_PROP_FPS))
                frame_num = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
                for i in tqdm(range(frame_num), total=frame_num, desc="pose estimation"):
                    ret, frame = vcap.read()
                    if not ret:
                        break
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    H, W = img_rgb.shape[:2]
                    bbox = self.models_dict["det_model"].predict(img_rgb)
                    if bbox is None:
                        bbox = [0, 0, W, H]
                    else:
                        bbox = bbox.tolist()[0]
                    bbox = utils.xyxy2xywh(bbox)
                    keypoints_ = self.models_dict["pose_model"].predict(img_rgb, bbox)
                    keypoints.append(keypoints_)
                keypoints = np.stack(keypoints)
                keypoints_info = {
                    "keypoints": keypoints,
                    "fps": fps,
                    "vdim": [H, W]
                }
                with open(keypoints_pkl, "wb") as fw:
                    pickle.dump(keypoints_info, fw)
                print(keypoints_pkl)
                vcap.release()

            if kwargs.get("draw_pose", True):
                keypoints = keypoints_info["keypoints"]
                vfps = keypoints_info["fps"]
                H, W = keypoints_info["vdim"]
                frame_num = keypoints.shape[0]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                save_vpath = os.path.join(save_dir, f"{vname}.mp4")
                vcom = cv2.VideoWriter(save_vpath, fourcc, vfps, (W, H))
                for i in tqdm(range(frame_num), total=frame_num, desc="draw pose"):
                    img_draw = draw.draw_pose_v2(keypoints[i], H, W, draw_foot=True, draw_face=False, to_rgb=False)
                    vcom.write(img_draw)
                vcom.release()
                print(save_vpath)

        except Exception as e:
            import traceback
            traceback.print_exc()
