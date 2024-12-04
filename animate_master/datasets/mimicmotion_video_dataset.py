# -*- coding: utf-8 -*-
# @Time    : 2024/8/30 23:00
# @Project : AnimateMaster
# @FileName: mimicmotion_video_dataset.py

import json
import os.path
import pdb
import pickle
import random
from typing import List

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from decord import VideoReader
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor


class MimicMotionVideoDataset(Dataset):
    def __init__(self,
                 img_size=(1024, 576),
                 img_scale=(1.0, 1.0),
                 img_ratio=(6 / 16.0, 12 / 16.0),
                 sample_rate=1,
                 n_sample_frames=32,
                 cond_type="openpose",
                 meta_paths=[],
                 cond_with_mask=["openpose"],
                 **kwargs):
        self.img_size = img_size
        self.img_scale = img_scale
        self.img_ratio = img_ratio
        self.sample_rate = sample_rate
        self.n_sample_frames = n_sample_frames
        self.cond_type = cond_type
        self.meta_paths = meta_paths
        self.cond_with_mask = cond_with_mask
        self.load_videos(**kwargs)

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.img_size,
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.img_size,
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.NEAREST,
                )
            ]
        )

    def load_videos(self, **kwargs):
        self.video_paths = []
        for root_dir in self.meta_paths:
            video_dir = os.path.join(root_dir, "videos")
            for vname in os.listdir(video_dir):
                if vname.lower().endswith(".mp4"):
                    vpath = os.path.join(video_dir, vname)
                    cond_vpath = os.path.join(root_dir, self.cond_type, vname)
                    if os.path.exists(vpath) and os.path.exists(cond_vpath):
                        self.video_paths.append((vpath, cond_vpath))

    def augmentation(self, images, transform, state=None):
        if isinstance(images, List):
            transformed_images = []
            for img in images:
                if state is not None:
                    torch.set_rng_state(state)
                transformed_images.append(transform(img))
            ret_tensor = torch.stack(transformed_images, dim=0)  # (f, c, h, w)
        else:
            if state is not None:
                torch.set_rng_state(state)
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor

    def create_mask_from_keypoints(self, keypoints, h, w, score_thred=0.4):
        mask = np.zeros((h, w), dtype=np.float32)

        # COCO WholeBody的关键点索引
        face_indices = list(range(23, 91))
        left_hand_indices = list(range(91, 112))
        right_hand_indices = list(range(112, 133))

        def get_bbox_from_keypoints(keypoints_subset, score_thred):
            """计算关键点子集的边界框，并进行一定的扩展"""
            if not (keypoints_subset[:, 2] > score_thred).all():
                return None
            valid_keypoints = keypoints_subset[keypoints_subset[:, 2] > score_thred]
            x_min = int(np.min(valid_keypoints[:, 0]))
            x_max = int(np.max(valid_keypoints[:, 0]))
            y_min = int(np.min(valid_keypoints[:, 1]))
            y_max = int(np.max(valid_keypoints[:, 1]))
            if abs((x_max - x_min) - (y_max - y_min)) > min(x_max - x_min, y_max - y_min):
                return None
            if (x_max - x_min) < w // 10 or (y_max - y_min) < h // 20:
                return None
            w_padding = (x_max - x_min) // 5
            h_padding = (y_max - y_min) // 5
            # 确保边界框不会超出图片边界
            x_min = max(0, x_min - w_padding)
            x_max = min(w, x_max + w_padding)
            y_min = max(0, y_min - h_padding)
            y_max = min(h, y_max + w_padding)
            return x_min, y_min, x_max, y_max

        # 获取各部分的包围框
        face_bbox = get_bbox_from_keypoints(keypoints[face_indices], score_thred)
        left_hand_bbox = get_bbox_from_keypoints(keypoints[left_hand_indices], score_thred)
        right_hand_bbox = get_bbox_from_keypoints(keypoints[right_hand_indices], score_thred)

        # 填充掩码
        if face_bbox:
            x_min, y_min, x_max, y_max = face_bbox
            mask[y_min:y_max, x_min:x_max] = 1
        if left_hand_bbox:
            x_min, y_min, x_max, y_max = left_hand_bbox
            mask[y_min:y_max, x_min:x_max] = 1
        if right_hand_bbox:
            x_min, y_min, x_max, y_max = right_hand_bbox
            mask[y_min:y_max, x_min:x_max] = 1

        return mask

    def drop_hand_area(self, cond_img, hand_keypoints, h, w, score_thred):
        def get_bbox_from_keypoints(keypoints_subset, score_thred):
            if (keypoints_subset[:, 2] > score_thred).all():
                return None
            valid_keypoints = keypoints_subset[keypoints_subset[:, 2] > score_thred]
            if valid_keypoints.shape[0] < 6:
                return None
            x_min = int(np.min(valid_keypoints[:, 0]))
            x_max = int(np.max(valid_keypoints[:, 0]))
            y_min = int(np.min(valid_keypoints[:, 1]))
            y_max = int(np.max(valid_keypoints[:, 1]))
            if abs((x_max - x_min) - (y_max - y_min)) > min(x_max - x_min, y_max - y_min):
                return None
            if (x_max - x_min) < w // 10 or (y_max - y_min) < h // 20:
                return None
            w_padding = (x_max - x_min) // 4
            h_padding = (y_max - y_min) // 4
            # 确保边界框不会超出图片边界
            x_min = max(0, x_min - w_padding)
            x_max = min(w, x_max + w_padding)
            y_min = max(0, y_min - h_padding)
            y_max = min(h, y_max + w_padding)
            return x_min, y_min, x_max, y_max

        # 获取各部分的包围框
        hand_bbox = get_bbox_from_keypoints(hand_keypoints, score_thred)

        # 填充掩码
        if hand_bbox:
            x_min, y_min, x_max, y_max = hand_bbox
            cond_img[y_min:y_max, x_min:x_max] = 0

        return cond_img

    def __getitem__(self, index):
        try:
            video_path, cond_vpath = self.video_paths[index]
            video_reader = VideoReader(video_path)
            cond_reader = VideoReader(cond_vpath)

            if self.cond_type in self.cond_with_mask:
                cond_pkl = cond_vpath[:-4] + ".pkl"
                with open(cond_pkl, "rb") as fin:
                    cond_data = pickle.load(fin)
            # assert len(video_reader) == len(
            #     cond_reader
            # ), f"{len(video_reader) = } != {len(cond_reader) = } in {video_path}"

            video_length = min(len(video_reader), len(cond_reader))
            video_fps = video_reader.get_avg_fps()
            if video_fps > 30:  # 30-60
                sample_rate = self.sample_rate * 2
            else:
                sample_rate = self.sample_rate

            sample_rate = sample_rate * random.uniform(0.5, 1.5)
            clip_length = min(
                video_length, int((self.n_sample_frames - 1) * sample_rate) + 1
            )
            start_idx = random.randint(0, video_length - clip_length)
            frame_indexs = np.linspace(
                start_idx, start_idx + clip_length - 1, self.n_sample_frames, dtype=int
            ).tolist()

            # read frames and kps
            vid_image_list = []
            cond_image_list = []
            cond_mask_list = []
            for index in frame_indexs:
                vid_img = video_reader[index].asnumpy()
                vid_image_list.append(Image.fromarray(vid_img))
                cond_img = cond_reader[index].asnumpy()
                if self.cond_type in self.cond_with_mask:
                    h, w = vid_img.shape[:2]
                    if self.cond_type in ["openpose"]:
                        keypoints_ = cond_data["keypoints"][index][0].copy()
                    else:
                        keypoints_ = cond_data["keypoints"][index].copy()
                    cond_mask = self.create_mask_from_keypoints(keypoints_, h, w, 0.5)
                    cond_mask_list.append(torch.from_numpy(cond_mask).float()[None])
                    # random drop hand area
                    left_hand_indices = list(range(91, 112))
                    right_hand_indices = list(range(112, 133))
                    # if random.random() < 0.2:
                    #     cond_img = self.drop_hand_area(cond_img, keypoints_[left_hand_indices], h, w, 0.5)
                    # if random.random() < 0.2:
                    #     cond_img = self.drop_hand_area(cond_img, keypoints_[right_hand_indices], h, w, 0.5)

                cond_image_list.append(Image.fromarray(cond_img))

            if random.random() < 0.05:
                swap_ind1 = random.randint(1, len(cond_image_list) - 2)
                swap_ind2 = random.randint(1, len(cond_image_list) - 2)
                cond_image_list[swap_ind1], cond_image_list[swap_ind2] = cond_image_list[swap_ind2], cond_image_list[
                    swap_ind1]
                if self.cond_type in self.cond_with_mask:
                    cond_mask_list[swap_ind1], cond_mask_list[swap_ind2] = cond_mask_list[swap_ind2], cond_mask_list[
                        swap_ind1]

            ref_img_idx = random.randint(0, video_length - 1)
            ref_img = Image.fromarray(video_reader[ref_img_idx].asnumpy())

            # transform
            state = torch.get_rng_state()
            pixel_values_vid = self.augmentation(
                vid_image_list, self.transform, state
            )
            pixel_values_cond = self.augmentation(
                cond_image_list, self.transform, state
            )
            if self.cond_type in self.cond_with_mask:
                mask_values_cond = self.augmentation(cond_mask_list, self.mask_transform, state)
            if random.random() < 0.2:
                state = torch.get_rng_state()
            pixel_values_ref = self.augmentation(ref_img, self.transform, state)

            sample = dict(
                pixel_values_vid=pixel_values_vid,
                pixel_values_cond=pixel_values_cond,
                pixel_values_ref=pixel_values_ref
            )
            if self.cond_type in self.cond_with_mask:
                sample["mask_values_cond"] = mask_values_cond

            return sample
        except Exception as e:
            # import traceback
            # traceback.print_exc()
            return self.__getitem__(random.randint(0, self.__len__() - 1))

    def __len__(self):
        return len(self.video_paths)
