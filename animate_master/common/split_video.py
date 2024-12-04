# -*- coding: utf-8 -*-
# @Time    : 2024/8/18 22:17
# @Project : AnimateMaster
# @FileName: split_video.py
import copy
import pdb

import cv2
import tqdm
import os
from scenedetect import open_video, SceneManager, split_video_ffmpeg
from scenedetect.detectors import ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg

from animate_master import infer_models
from . import utils


class VideoSceneDetectSplitPipeline:
    """
    视频转场检测和分割
    """

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
                print(f"{vpath} not exist!")
                return None
            min_duration = kwargs.get("min_duration", 1)
            # Open our video, create a scene manager, and add a detector.
            video = open_video(vpath)
            vname = os.path.basename(vpath)
            vcap = cv2.VideoCapture(vpath)
            vfps = int(vcap.get(cv2.CAP_PROP_FPS))
            vh = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            vw = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
            vframe_num = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))

            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector(threshold=kwargs.get("threshold", 27)))
            scene_manager.detect_scenes(video, show_progress=True)
            scene_list = scene_manager.get_scene_list()
            scene_frames = []
            for i, scene in enumerate(scene_list):
                if scene[1].get_frames() - scene[0].get_frames() > min_duration * vfps:
                    print('    Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
                        i + 1,
                        scene[0].get_timecode(), scene[0].get_frames(),
                        scene[1].get_timecode(), scene[1].get_frames(),))
                    scene_frames.append([scene[0].get_frames(), scene[1].get_frames()])

            # 默认选取场景里只有一个人的片段
            allow_person_num = kwargs.get("allow_person_num", [1])

            save_dir = kwargs.get("save_dir", os.path.dirname(vpath))
            os.makedirs(save_dir, exist_ok=True)

            # 先做一波筛选，如果这个片段小于2秒，则丢弃
            if len(scene_frames) == 0:
                print("no scene detect")
                return None

            scene_video_list = []
            j = 0
            scene_cur = scene_frames[j]
            frames_cur = []
            for i in range(vframe_num):
                ret, frame = vcap.read()
                if not ret:
                    break
                if i >= scene_cur[0] and i < scene_cur[1]:
                    frames_cur.append(frame)
                elif i == scene_cur[1]:
                    frames_cur.append(frame)
                    for f, img_bgr in enumerate(frames_cur):
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        h, w = img_rgb.shape[:2]
                        detect_bbox = self.models_dict["yolo_detect"].predict(img_rgb)
                        if detect_bbox is None or len(detect_bbox) not in allow_person_num:
                            frames_cur[f] = None
                        else:
                            detect_bbox = utils.get_player_box_by_max_area(detect_bbox, h, w)
                            if utils.iou(utils.xyxy2xywh(detect_bbox), [0, 0, w, h]) < 0.4:
                                frames_cur[f] = None

                    max_fnum = -1
                    max_f_inter = [-1, -1]
                    start_ind = -1
                    for k, fr in enumerate(frames_cur):
                        if fr is not None:
                            if start_ind == -1:
                                start_ind = k
                        else:
                            if start_ind != -1 and k - start_ind > max_fnum:
                                max_fnum = max(max_fnum, k - start_ind)
                                max_f_inter = [start_ind, k]
                            start_ind = -1
                    if start_ind != -1:
                        max_f_inter = [start_ind, len(frames_cur)]
                    if max_f_inter[1] - max_f_inter[0] > min_duration * vfps:
                        # save to video
                        max_f_inter[0] += 5
                        max_f_inter[1] -= 5
                        split_save_vpath = os.path.join(save_dir,
                                                        f"{os.path.splitext(vname)[0]}-"
                                                        f"{scene_cur[0] + max_f_inter[0]:06d}-"
                                                        f"{scene_cur[0] + max_f_inter[1]:06d}.mp4")
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        vcom = cv2.VideoWriter(split_save_vpath, fourcc, vfps, (vw, vh))
                        for img_bgr_ in frames_cur[max_f_inter[0]:max_f_inter[1]]:
                            vcom.write(img_bgr_)
                        vcom.release()
                        scene_video_list.append(split_save_vpath)

                    j += 1
                    frames_cur = []
                    if j >= len(scene_frames):
                        break
                    scene_cur = scene_frames[j]

            if len(frames_cur):
                for f, img_bgr in enumerate(frames_cur):
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    h, w = img_rgb.shape[:2]
                    detect_bbox = self.models_dict["yolo_detect"].predict(img_rgb)
                    if detect_bbox is None or len(detect_bbox) not in allow_person_num:
                        frames_cur[f] = None
                    else:
                        detect_bbox = utils.get_player_box_by_max_area(detect_bbox, h, w)
                        if utils.iou(utils.xyxy2xywh(detect_bbox), [0, 0, w, h]) < 0.4:
                            frames_cur[f] = None
                max_fnum = -1
                max_f_inter = [-1, -1]
                start_ind = -1
                for k, fr in enumerate(frames_cur):
                    if fr is not None:
                        if start_ind == -1:
                            start_ind = k
                    else:
                        if start_ind != -1 and k - start_ind > max_fnum:
                            max_fnum = max(max_fnum, k - start_ind)
                            max_f_inter = [start_ind, k]
                        start_ind = -1
                if start_ind != -1:
                    max_f_inter = [start_ind, len(frames_cur)]
                if max_f_inter[1] - max_f_inter[0] > min_duration * vfps:
                    # save to video
                    max_f_inter[0] += 5
                    max_f_inter[1] -= 5
                    split_save_vpath = os.path.join(save_dir,
                                                    f"{os.path.splitext(vname)[0]}-"
                                                    f"{scene_cur[0] + max_f_inter[0]:06d}-"
                                                    f"{scene_cur[0] + max_f_inter[1]:06d}.mp4")
                    # 如果已经存在就不要再重新写入了
                    if not os.path.exists(split_save_vpath):
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        vcom = cv2.VideoWriter(split_save_vpath, fourcc, vfps, (vw, vh))
                        for img_bgr_ in frames_cur[max_f_inter[0]:max_f_inter[1]]:
                            vcom.write(img_bgr_)
                        vcom.release()
                    scene_video_list.append(split_save_vpath)
            vcap.release()
            return scene_video_list
        except Exception as e:
            import traceback
            traceback.print_exc()
