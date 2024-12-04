# -*- coding: utf-8 -*-
# @Time    : 2024/8/18 12:42
# @Author  : shaoguowen
# @Email   : wenshaoguo1026@gmail.com
# @Project : AnimateMaster
# @FileName: download_videos.py

"""
# 单个视频下载
python scripts/data_processing/download_videos.py --vurl https://www.youtube.com/shorts/OrfPHaqtjA4

# 多个视频下载
python scripts/data_processing/download_videos.py \
--vurl data/youtube_0821/urls
"""

import os
import multiprocessing as mp
import argparse
import pdb
import random
import time
import traceback
from pytubefix import YouTube
from pytubefix.cli import on_progress


def download_video(video_url, vind, save_dir):
    """
    下载视频
    :param video_url:
    :return:
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        print(f"download {vind}th {video_url}")
        """
        不行的话的就试以下不同的client：
        WEB, WEB_EMBED, WEB_MUSIC, WEB_CREATOR, WEB_SAFARI,
        ANDROID, ANDROID_MUSIC, ANDROID_CREATOR, ANDROID_VR, ANDROID_PRODUCER, ANDROID_TESTSUITE,
        IOS, IOS_MUSIC, IOS_CREATOR,
        MWEB, TV_EMBED, MEDIA_CONNECT
        """
        yt = YouTube(video_url, client='ANDROID_VR', on_progress_callback=on_progress, use_oauth=True,
                     allow_oauth_cache=True)
        vname = os.path.basename(video_url) + ".mp4" if "shorts" in video_url else video_url.split("v=")[-1] + ".mp4"
        yt.streams.filter(adaptive=True, file_extension='mp4', only_video=True).order_by(
            'resolution').desc().first().download(output_path=save_dir, filename=vname, max_retries=3)
        # time.sleep(0.2)
    except Exception as e:
        traceback.print_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vurl', required=True, type=str, help='video url')
    parser.add_argument('--save_dir', required=False, type=str, help='video save dir')
    args, unknown = parser.parse_known_args()
    if os.path.isdir(args.vurl):
        vurl_dir = args.vurl
        if args.save_dir is not None:
            root_save_dir = args.save_dir
        else:
            root_save_dir = os.path.join(vurl_dir, "..", "raw_videos")
        total_num = 0
        vurls_txt = [os.path.join(vurl_dir, txt_name) for txt_name in os.listdir(vurl_dir) if txt_name.endswith(".txt")]
        pool = mp.Pool(8)
        for vurl_txt_ in vurls_txt:
            save_dir = os.path.join(root_save_dir, os.path.splitext(os.path.basename(vurl_txt_))[0])
            vurls = []
            with open(vurl_txt_, "r", encoding="utf-8") as fin:
                vurls = fin.readlines()
                vurls = ["https://" + url.split("https://")[-1].strip().replace("\n", "") for url in vurls if
                         "https://" in url]
            print(f"{vurl_txt_} have total num of videos:{len(vurls)}")
            for i, vurl in enumerate(vurls):
                vname = os.path.basename(vurl) + ".mp4" if "shorts" in vurl else vurl.split("v=")[-1] + ".mp4"
                vpath = os.path.join(save_dir, vname)
                if not os.path.exists(vpath):
                    total_num += 1
                    pool.apply_async(download_video, args=(vurl, i, save_dir))

        pool.close()
        pool.join()
        print(f"total download {total_num} videos")
    else:
        vurl = args.vurl
        if args.save_dir is not None:
            root_save_dir = args.save_dir
        else:
            root_save_dir = "./data/tests/raw_videos"
        download_video(vurl, 0, root_save_dir)
