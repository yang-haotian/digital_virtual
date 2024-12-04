tmux -S farming
cd /project_data/ws_projects/AnimateMaster
conda activate animate
To re-attach
$ tmux -S farming attach

Alternatively, you can use the following single command to both create (if not exists already) and attach to a session:
$ tmux new-session -A -D -s farming

To delete farming session
$ tmux kill-session -t farming



train_mimicmotion_v2 shaoguo
train_mimicmotion_v2_test 用于直接调试时候使用
train_mimicmotion_v2_yang 昊天在v2基础上做实验
存在问题：
1. 当前直接把mimic的ref_image_latents进行输入，个人感觉不合理
2. 感觉不会跑通，因为架构不统一，一个是sd一个是svd







musepose带v2是用了跟mimic类似的loss
训练记录：
1030实验记录分享_1
输入：
选择脚本：train_musepose_stage2.py
选择数据：
      "./data/TikTok",
      "./data/UBC_fashion",
      "./data/youtube_0818",
      "./data/youtube_0821",
      "./data/youtube_man",
      "./data/youtube_man_0907",
      "./data/first_data_1015"
musepose阶段二
n_sample_frames: 64
denoising_unet_path: "/project_data/ws_projects/AnimateMaster/outputs/MusePose_stage1-20241028091309/denoising_unet-17000.pth"
reference_unet_path: "/project_data/ws_projects/AnimateMaster/outputs/MusePose_stage1-20241028091309/reference_unet-17000.pth"
pose_guider_path: "/project_data/ws_projects/AnimateMaster/outputs/MusePose_stage1-20241028091309/pose_guider-17000.pth"
motion_module_path: "/project_data/ws_projects/AnimateMaster/checkpoints/MusePose/motion_module.pth"

输出：（已删除，效果太差，没必要保存）
路径：/project_data/ws_projects/AnimateMaster/outputs/MusePose_stage2-20241029025941

实验效果，很差，考虑可以随时删除，背景训练到19000，还是会出现乱七八糟的生成。
![alt text](image.png)
![alt text](image-1.png)

todo:检查问题原因

1030实验记录分享_2（没写的参数一致，1030实验记录分享_1）
1.未解决上一步操作原因，n_sample_frames: 32，看一下结果是否还是非常不好，并检查下到底什么情况导致。[已解决]
换成32之后确实效果好了很多，不再出现之前的问题，后续可以尝试换成32-64之前的数据看下效果
2.验证成功，n_sample_frames: 32后，确实没在出现之前的问题，但是却一直没办法和开源一样取得衣服细致且自然的效果。（猜测1数据分布，猜测2学习率要么太高要么太低）
3.至少迭代5次（batch1 1000的数据量要迭代3000次）
4.第二阶段和第一阶段可以喂不同的数据做尝试。接下来去验证结果就好。
输出路径：(越训练效果感觉越差，删除模型产出)
/project_data/ws_projects/AnimateMaster/outputs/MusePose_stage2-20241030100907

1031实验记录分享_1（没写的参数一致，1030实验记录分享_1）
1.更换了数据集（只在stage2阶段进行更换，因为第一阶段已经很好了），更换为
"./data/first_data_1015",
"/project_data/subj_data"

2.一直有个疑问，感觉像素太低，并且不确定迭代上来后效果会提升很多吗,训练的次数多像素会高吗[已解决]
像素与配置有关，与迭代次数无关


3.更换了数据集结果会提升吗（因为从来没有尝试成功过1个epoch以上的，所以不好下结论）
输出路径：(越训练效果感觉越差，删除模型产出)
/project_data/ws_projects/AnimateMaster/outputs/MusePose_stage2-20241031020915

像素迭代1000次与开源视频的对比
![alt text](image-2.png)

左边1000 右边4000 像素没有差别
![alt text](image-3.png)

1031实验记录分享_2 实验目的：验证像素是否可以提高
1.参数更换（与1031实验记录分享_1比对，只更换了第二阶段如下参数）
  train_bs: 1
  train_width: 576
  train_height: 1024
2.配置调整，像素高了吗[已解决]
像素提升了
左一本次实验提升后的像素参数，左二上次实验的像素参数。右一开源的像素参数
![alt text](image-4.png)

3.像素越高，效果越好
感觉像素提升后效果也变好了
左边低像素，右边高像素
![alt text](image-5.png)

4.现在用的是mimic的骨骼，换成musepose看看会不会更好（目前手指很差劲）
训练数据换成 musepose 生成的方法

5.像素提升后好像不是很稳定，个别帧有问题
6.下一步做实验要找个第一阶段好的模型，不然生成的手指都有问题

以下等验证过6再试下面：
7.需要真正跑一次长时间的数据，看看到底能不能变化
（1）下大决心看看第一阶段训得就效果会好一点，生成的手指等是否会稳定一点
8.调整学习率试试，数据量大，学习率调正低一点试试效果。[已解决]
（1）经过和多人沟通和实验结果的loss观察，认为学习率并不高
9.多个数据训练一次 和少量数据迭代多次哪个更重要
10.可否不依赖别人，自己进行训练，因为我们的需求并不需要用sd1.5通用的 我们只需要真人的，所以直接换底层模型效果是否可能会更好
很想尝试下，可以等lora找到更好的底层模型后然后用这个进行尝试看看
11.mimic使用referencenet进行结合，看是否可以达到更好的效果

1101实验记录分享_1
最大的目的就是尝试训练久一点，看看能达到怎么样的效果。

1.更换第1阶段如下参数
  train_bs: 4
  train_width: 768
  train_height: 1024
  max_train_steps: 50000

  "./data/first_data_1015",
  "/project_data/subj_data"
  第一阶段脚本576改为768

2.输出如下
/project_data/ws_projects/AnimateMaster/outputs/MusePose_stage1-20241101023445
已经跑完5万次

1102实验记录分享_1
第一阶段
目的：
延续1101_1的实验结果，继续跑10万次试试，因为发现loss确实是一直在下降
输出：
/project_data/ws_projects/AnimateMaster/outputs/MusePose_stage1-20241102014252
第二阶段继续
打算跑一下迭代10万次的感觉，跑10万次epoch为5至少，但是跑了1000次就停了，然后继续
data:
  train_bs: 1
  train_width: 768
  train_height: 1024
  n_sample_frames: 24

第一次1000次/project_data/ws_projects/AnimateMaster/outputs/MusePose_stage2-20241102154520 1000次
第二次续上又2000次可以用 /project_data/ws_projects/AnimateMaster/outputs/MusePose_stage2-20241103005645 
第三次 /project_data/ws_projects/AnimateMaster/outputs/MusePose_stage2-20241103071809





注意：
（1）想验证一个问题：训练的时候低像素，推理的时候高像素可以吗
（2）拿好的模型做基础训练 一开始效果好可能是因为低模效果好，跑一会儿可能效果就不好了
（3）直接拿现成referencenet去改善mimic会有好的效果吗，明天整理下
（4）研究下一个模型多尝试几次生成 效果怎么样



经验总结沟通：
1.不用nohup,建议用tumx（不确定是否是以下原因，因为我在batchsize 选的比较大的时候好像也会发生如下问题）
否则会报错
W1030 17:53:58.738000 140422282622784 torch/distributed/elastic/multiprocessing/api.py:858 Sending process 2199936 closing signal SIGHUP
W1030 17:53:58.739000 140422282622784 torch/distributed/elastic/multiprocessing/api.py:858 Sending process 2199937 closing signal SIGHUP
torch.distributed.elastic.multiprocessing.api.SignalException: Process 2199872 got signal: 1


底层模版
第一阶段
data:
  train_bs: 8
  train_width: 576
  train_height: 1024
  # Margin of frame indexes between ref and tgt images
  sample_margin: 64

solver:
  gradient_accumulation_steps: 1
  mixed_precision: 'fp16'
  enable_xformers_memory_efficient_attention: True
  gradient_checkpointing: False
  max_train_steps: 50000
  max_grad_norm: 1.0
  # lr
  learning_rate: 5.0e-6
  scale_lr: False
  lr_warmup_steps: 100
  lr_scheduler: 'constant'

  # optimizer
  use_8bit_adam: False
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_weight_decay: 1.0e-2
  adam_epsilon: 1.0e-8

val:
  validation_steps: 1000


noise_scheduler_kwargs:
  num_train_timesteps: 1000
  beta_start: 0.00085
  beta_end: 0.012
  beta_schedule: "linear"
  steps_offset: 1
  clip_sample: false

pretrained_base_model_path: './checkpoints/sd-image-variations-diffusers'
pretrained_vae_path: './checkpoints/sd-vae-ft-mse'

# denoising_unet_path: "./outputs/MusePose_stage1-20241017072400/denoising_unet-2000.pth"
# reference_unet_path: "./outputs/MusePose_stage1-20241017072400/reference_unet-2000.pth"
# pose_guider_path: "./outputs/MusePose_stage1-20241017072400/pose_guider-2000.pth"
denoising_unet_path: "/project_data/ws_projects/AnimateMaster/checkpoints/MusePose/denoising_unet.pth"
reference_unet_path: "/project_data/ws_projects/AnimateMaster/checkpoints/MusePose/reference_unet.pth"
pose_guider_path: "/project_data/ws_projects/AnimateMaster/checkpoints/MusePose/pose_guider.pth"




weight_dtype: 'fp16'  # [fp16, fp32]
uncond_ratio: 0.1
noise_offset: 0.05
snr_gamma: 5.0
enable_zero_snr: True
pose_guider_pretrain: True

seed: 123
checkpointing_steps: 1000
exp_name: 'MusePose_stage1'
output_dir: './outputs'




第二阶段
data:
  train_bs: 1
  train_width: 448
  train_height: 768
  n_sample_frames: 64

solver:
  gradient_accumulation_steps: 1
  mixed_precision: 'fp16'
  enable_xformers_memory_efficient_attention: True
  gradient_checkpointing: True
  max_train_steps: 100000
  max_grad_norm: 1.0
  # lr
  learning_rate: 1e-5
  scale_lr: False
  lr_warmup_steps: 1
  lr_scheduler: 'constant'

  # optimizer
  use_8bit_adam: True
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_weight_decay: 1.0e-2
  adam_epsilon: 1.0e-8

val:
  validation_steps: 1000

unet_additional_kwargs:
  use_inflated_groupnorm: true
  unet_use_cross_frame_attention: false
  unet_use_temporal_attention: false
  use_motion_module: true
  motion_module_resolutions:
    - 1
    - 2
    - 4
    - 8
  motion_module_mid_block: true
  motion_module_decoder_only: false
  motion_module_type: Vanilla
  motion_module_kwargs:
    num_attention_heads: 8
    num_transformer_block: 1
    attention_block_types:
      - Temporal_Self
      - Temporal_Self
    temporal_position_encoding: true
    temporal_position_encoding_max_len: 128
    temporal_attention_dim_div: 1

noise_scheduler_kwargs:
  num_train_timesteps: 1000
  beta_start: 0.00085
  beta_end: 0.012
  beta_schedule: "linear"
  steps_offset: 1
  clip_sample: false

pretrained_base_model_path: './checkpoints/sd-image-variations-diffusers'
pretrained_vae_path: './checkpoints/sd-vae-ft-mse'

# denoising_unet_path: "./outputs/MusePose_stage1-20241017072400/denoising_unet-2000.pth"
# reference_unet_path: "./outputs/MusePose_stage1-20241017072400/reference_unet-2000.pth"
# pose_guider_path: "./outputs/MusePose_stage1-20241017072400/pose_guider-2000.pth"
# motion_module_path: "./outputs/MusePose_stage2-20241017093701/motion_module-1000.pth"

denoising_unet_path: "/project_data/ws_projects/AnimateMaster/outputs/MusePose_stage1-20241028091309/denoising_unet-17000.pth"
reference_unet_path: "/project_data/ws_projects/AnimateMaster/outputs/MusePose_stage1-20241028091309/reference_unet-17000.pth"
pose_guider_path: "/project_data/ws_projects/AnimateMaster/outputs/MusePose_stage1-20241028091309/pose_guider-17000.pth"
motion_module_path: "/project_data/ws_projects/AnimateMaster/checkpoints/MusePose/motion_module.pth"

weight_dtype: 'fp16'  # [fp16, fp32]
uncond_ratio: 0.1
noise_offset: 0.05
snr_gamma: 5.0
enable_zero_snr: True

seed: 42
checkpointing_steps: 1000
exp_name: 'MusePose_stage2'
output_dir: './outputs'


