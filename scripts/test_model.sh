#!/bin/bash
folder_path="/root/scratch/Moore-AnimateAnyone/lmb/project_result/sample/20240909_1/09"
musepose_extrapose_path="/root/scratch/MusePose/pose_align.py"
# mimic_path="/root/scratch/backup/ws_projects/AnimateMaster/tests/test_pipelines_yang.py"

img_pose_path="/root/scratch/MusePose/assets/custom/img_pose/custom_img_pose.png"
video_pose="/root/scratch/MusePose/assets/custom/video_pose/custom_video_pose.mp4"
# save_path="/root/scratch/Moore-AnimateAnyone/lmb/project_result/sample/20240909_1/09/method6"
save_path="/root/scratch/Moore-AnimateAnyone/lmb/project_result/sample/20240909_1/09/method2"

# 进入文件夹
cd $folder_path

# 遍历文件夹中的所有文件
for file in *; do
    if [[ $file == *.mp4 ]]; then
        file_mp4=$folder_path/$file
        echo $file_mp4
    fi
    if [[ $file == *.jpg ]]; then
        file_img=$folder_path/$file
        echo $file_img
    fi
done
# 使用musepose的方法获取姿态对齐结果（存入video_pose）和图片（存入img_pose_path）
cd /root/scratch/MusePose
/root/miniconda3/envs/AA/bin/python $musepose_extrapose_path --imgfn_refer $file_img --vidfn $file_mp4



# # 使用mimic获取结果
# cd /root/scratch/backup/ws_projects/AnimateMaster
# /root/miniforge-pypy3/envs/animate/bin/python  $mimic_path --ref_img_path $file_img --ref_img_pose_path $img_pose_path --pose_video_path $video_pose --save_path $save_path