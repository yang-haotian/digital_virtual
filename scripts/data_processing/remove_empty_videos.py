# -*- coding: utf-8 -*-
# @Time    : 2024/8/23 21:16
# @Project : AnimateMaster
# @FileName: remove_empty_videos.py

import os
import argparse


def remove_empty_videos(folder_path):
    # 遍历文件夹中的所有文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            # 检查文件是否为视频文件（根据扩展名）
            if file.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv')):
                # 检查文件大小是否为 0
                if os.path.getsize(file_path) == 0:
                    print(f"Deleting empty video file: {file_path}")
                    os.remove(file_path)


if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="Remove empty video files from a specified folder.")
    parser.add_argument("video_dir", type=str, help="Path to the folder to search for empty video files.")
    args = parser.parse_args()

    # 删除空视频文件
    remove_empty_videos(args.video_dir)
