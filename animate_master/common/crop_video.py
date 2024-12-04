# -*- coding: utf-8 -*-
# @Time    : 2024/8/18 21:28
# @Project : AnimateMaster
# @FileName: crop_video.py

import os
import pdb

import cv2
import shutil
import numpy as np
import tqdm
import av
from pathlib import Path
import subprocess


def crop_video_by_line(src_vpath, save_vpath=None, **kwargs):
    vcap = cv2.VideoCapture(src_vpath)
    vfps = int(vcap.get(cv2.CAP_PROP_FPS))
    vh = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vframe_num = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_step = kwargs.get("skip_step", vfps)

    crop_x1, crop_y1, crop_x2, crop_y2 = None, None, None, None
    vcom = None
    if save_vpath is None:
        save_vpath = src_vpath
    crop_save_vpath = save_vpath + "-crop.mp4"
    # 只对横屏进行切分
    if vh > vw:
        vcap.release()
    else:
        if os.path.exists(crop_save_vpath):
            vcap.release()
            return
        for i in tqdm.tqdm(range(vframe_num), total=vframe_num):
            ret, frame = vcap.read()
            if not ret:
                break
            if crop_x1 is None or crop_y1 is None or crop_x2 is None or crop_y2 is None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150, apertureSize=3)  # apertureSize是sobel算子窗口大小
                lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 180, threshold=100, lines=np.array([]),
                                        minLineLength=min(vw, vh) // 1.2, maxLineGap=80)
                if lines is not None:
                    for line in lines:
                        x1 = line[0][0]
                        y1 = line[0][1]
                        x2 = line[0][2]
                        y2 = line[0][3]

                        if kwargs.get("detect_vertical", True):
                            if abs(x1 - x2) <= 10 and min(y1, y2) < 10 and vh - max(y1, y2) < 10:
                                if x1 < vw // 2:
                                    crop_x1 = x1 if crop_x1 is None else max(crop_x1, x1)
                                    crop_x1 = min(max(0, crop_x1), vw)
                                else:
                                    crop_x2 = x1 if crop_x2 is None else min(crop_x2, x2)
                                    crop_x2 = min(max(0, crop_x2), vw)
                        else:
                            crop_x1 = 0
                            crop_x2 = vw

                        if kwargs.get("detect_horizon", True):
                            if abs(y1 - y2) <= 10 and min(x1, x2) < 10 and vh - min(x1, x2) < 10:
                                if y1 < vh // 2:
                                    crop_y1 = y1 if crop_y1 is None else max(crop_y1, y1)
                                    crop_y1 = min(max(0, crop_y1), vh)
                                else:
                                    crop_y2 = y1 if crop_y2 is None else min(crop_y2, y2)
                                    crop_y2 = min(max(0, crop_y2), vh)
                        else:
                            crop_y1 = 0
                            crop_y2 = vh
            if vcom is None and crop_x1 is not None and crop_y1 is not None and crop_x2 is not None and crop_y2 is not None:
                output_h = kwargs.get("output_h", None)
                output_w = kwargs.get("output_w", None)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                if output_h is None or output_w is None:
                    output_h = crop_y2 - crop_y1
                    output_w = crop_x2 - crop_x1
                vcom = cv2.VideoWriter(crop_save_vpath, fourcc, vfps, (output_w, output_h))
            if vcom is not None:
                frame_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                frame_crop = cv2.resize(frame_crop, (output_w, output_h))
                vcom.write(frame_crop)
            else:
                continue
        vcap.release()
        if vcom is not None:
            vcom.release()
            output_file = crop_save_vpath + "-crf18.mp4"
            command = [
                'ffmpeg',
                '-i', crop_save_vpath,
                '-c:v', 'h264_nvenc',  # 使用 CUDA 加速的编码器
                '-crf', '18',
                output_file,
                '-y'
            ]
            subprocess.call(command)
            os.remove(crop_save_vpath)
            if kwargs.get("delete_src_video", True):
                os.remove(src_vpath)
            if os.path.exists(output_file):
                os.rename(output_file, crop_save_vpath)
        else:
            # 如果到结束都没找到 分割线，说明不符合要求，直接删除
            print(f"delete video {src_vpath}")
            os.remove(src_vpath)
