# -*- coding: utf-8 -*-
# @Time    : 2024/8/18 15:49
# @Project : AnimateMaster
# @FileName: test_datasets.py
import os
import pdb
import random
import numpy as np
from PIL import Image


def test_mimicmotion_dataset():
    from animate_master.datasets.mimicmotion_video_dataset import MimicMotionVideoDataset
    dataset = MimicMotionVideoDataset(
        img_size=(1024, 576),
        img_scale=(0.8, 1.0),
        img_ratio=(6 / 16.0, 12 / 16.0),
        sample_rate=1,
        n_sample_frames=48,
        cond_type="openpose",
        meta_paths=["./data/cartoon_0831/"],
    )
    n = len(dataset)
    print(f"len of dataset: {len(dataset)}")

    ind = random.randint(0, n - 1)
    # ind = 14
    data = dataset[ind]
    save_dir = f"assets/test_examples/cartoon_{ind:03d}"
    os.makedirs(save_dir, exist_ok=True)

    # ref image
    ref_image = (data["pixel_values_ref"].permute(1, 2, 0) + 1) / 2
    ref_image = Image.fromarray((ref_image * 255).numpy().astype(np.uint8))
    ref_image.save(os.path.join(save_dir, "ref_image.png"))

    # pixel_values_vid
    vid_frame_dir = os.path.join(save_dir, "vid_frames")
    os.makedirs(vid_frame_dir, exist_ok=True)
    pixel_values_vid = data["pixel_values_vid"]
    for i in range(pixel_values_vid.shape[0]):
        image = (pixel_values_vid[i].permute(1, 2, 0) + 1) / 2
        image = Image.fromarray((image * 255).numpy().astype(np.uint8))
        image.save(os.path.join(vid_frame_dir, f"{i:03d}.jpg"))

    # pixel_values_cond
    cond_frame_dir = os.path.join(save_dir, "cond_frames")
    os.makedirs(cond_frame_dir, exist_ok=True)
    pixel_values_cond = data["pixel_values_cond"]
    for i in range(pixel_values_cond.shape[0]):
        image = (pixel_values_cond[i].permute(1, 2, 0) + 1) / 2
        image = Image.fromarray((image * 255).numpy().astype(np.uint8))
        image.save(os.path.join(cond_frame_dir, f"{i:03d}.jpg"))

    # mask_values_cond
    cond_mask_dir = os.path.join(save_dir, "cond_masks")
    os.makedirs(cond_mask_dir, exist_ok=True)
    mask_values_cond = data["mask_values_cond"]
    pdb.set_trace()
    for i in range(mask_values_cond.shape[0]):
        image = mask_values_cond[i][0]
        image = Image.fromarray((image * 255).numpy().astype(np.uint8))
        image.save(os.path.join(cond_mask_dir, f"{i:03d}.jpg"))


if __name__ == '__main__':
    test_mimicmotion_dataset()
