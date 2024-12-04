# -*- coding: utf-8 -*-
# @Time    : 2024/8/16 21:51
# @Project : AnimateMaster
# @FileName: test_pipelines.py
import pdb
import random

import cv2
from torchvision.io import write_video


def test_crop_video_pipeline():
    from animate_master.common.crop_video import crop_video_by_line
    vpath = "data/tests/raw_videos/E8zMZKRdxsM.mp4"
    crop_video_by_line(vpath, None, detect_horizon=False, output_h=1280, output_w=720, delete_src_video=True)


def test_split_video_pipeline():
    from animate_master.common.split_video import VideoSceneDetectSplitPipeline
    pipe_kwargs = {
        "models": {
            "yolo_detect": {
                "name": "YoloHumanDetectModel",
                "predict_type": "trt",
                "model_path": "./checkpoints/AnimateMaster/yolov10x.trt"
            }
        }
    }
    pipe = VideoSceneDetectSplitPipeline(**pipe_kwargs)
    vpath = "data/tests/raw_videos/ffF9IESyN0U.mp4"
    save_dir = "data/tests/split_videos"
    scene_video_list = pipe.run(vpath, save_dir=save_dir, min_duration=1)
    print(scene_video_list)


def test_mimicmotion_referencenet_pipeline():
    import torch
    import torch.utils.checkpoint
    from diffusers.models import AutoencoderKLTemporalDecoder
    from diffusers.schedulers import EulerDiscreteScheduler
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    import numpy as np
    from diffusers import AutoencoderKL, DDIMScheduler
    import os
    import datetime
    from PIL import Image
    from animate_master.models.unet_2d_condition import UNet2DConditionModel
    from animate_master.models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
    from animate_master.models.pose_net import PoseNet
    from animate_master.pipelines.pipeline_mimicmotion_yang import MimicMotionPipeline_v2
    from animate_master.common import utils

    with torch.no_grad():
        device = torch.device("cuda")
        weight_dtype = torch.float16
        pretrained_base_model_path_sd = '/project_data/ws_projects/AnimateMaster/checkpoints/sd-image-variations-diffusers'
        # denoising_unet_path = "/project_data/ws_projects/AnimateMaster/outputs/MimicMotion-20241122093755/checkpoint-1000/denoising_unet.pth"
        # reference_unet_path = "/project_data/ws_projects/AnimateMaster/outputs/MimicMotion-20241122093755/checkpoint-1000/reference_unet.pth"
        denoising_unet_path = "/project_data/ws_projects/AnimateMaster/outputs/checkpoint-41000/denoising_unet.pth"
        reference_unet_path = "/project_data/ws_projects/AnimateMaster/outputs/checkpoint-41000/reference_unet.pth"
        base_model_path = "/project_data/ws_projects/AnimateMaster/checkpoints/stable-video-diffusion-img2vid-xt-1-1"
        # mimicmotion_model_path = "checkpoints/MimicMotion/MimicMotion_1-1.pth"
        # pretrained_vae_path = './checkpoints/sd-vae-ft-mse'
        pretrained_base_model_path = '/project_data/ws_projects/AnimateMaster/checkpoints/sd-image-variations-diffusers'
        pretrain_cond_net = '/project_data/ws_projects/AnimateMaster/outputs/checkpoint-41000/cond_net_openpose.pth'
        # pretrain_cond_net = '/project_data/ws_projects/AnimateMaster/outputs/MimicMotion-20241122093755/checkpoint-1000/cond_net_openpose.pth'
        
        reference_unet = UNet2DConditionModel.from_pretrained(
            pretrained_base_model_path_sd,
            subfolder="unet",
        ).to(dtype=weight_dtype, device=device)

        denoising_unet = UNetSpatioTemporalConditionModel.from_config(
            UNetSpatioTemporalConditionModel.load_config(base_model_path, subfolder="unet")).to(dtype=weight_dtype, device=device)
        
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            base_model_path, subfolder="vae").to(dtype=weight_dtype, device=device)
        # vae_2d = AutoencoderKL.from_pretrained(
        #     pretrained_vae_path,
        # ).to("cuda", dtype=weight_dtype)

        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            base_model_path, subfolder="image_encoder").to(dtype=weight_dtype, device=device)
        
        image_enc = CLIPVisionModelWithProjection.from_pretrained(
        pretrained_base_model_path,subfolder="image_encoder").to(dtype=weight_dtype, device=device)
        
        noise_scheduler = EulerDiscreteScheduler.from_pretrained(
            base_model_path, subfolder="scheduler")
        feature_extractor = CLIPImageProcessor.from_pretrained(
            base_model_path, subfolder="feature_extractor")
        # pose_net
        # pose_net = PoseNet(noise_latent_channels=unet.config.block_out_channels[0])
        cond_net = PoseNet(noise_latent_channels=denoising_unet.config.block_out_channels[0]).to(dtype=weight_dtype, device=device)
    

        # load pretrained weights

        reference_unet.load_state_dict(
            torch.load(reference_unet_path, map_location="cpu"),
        )
        denoising_unet.load_state_dict(
            torch.load(denoising_unet_path, map_location="cpu"),
            strict=False,
        )
        cond_net.load_state_dict(torch.load(pretrain_cond_net, map_location="cpu"),
                                strict=True)
                    
        # # create pipeline
        # if args.use_ema:
        #     # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
        #     ema_unet.store(unet.parameters())
        #     ema_unet.copy_to(unet.parameters())
        # The models need unwrapping because for compatibility in distributed training mode.
        # 以下自己写
        
        # # reference_unet = unet_.reference_unet
        # unet_ = accelerator.unwrap_model(unet)
        # unet = unet_.denoising_unet
        # pose_guider = ori_net.pose_guider
        # ori_net = accelerator.unwrap_model(unet)
        # reference_unet = ori_net.reference_unet
        # denoising_unet = ori_net.denoising_unet
        # pipeline = MimicMotionPipeline(
        #     vae=accelerator.unwrap_model(vae),
        #     image_encoder=accelerator.unwrap_model(image_encoder),
        #     unet=accelerator.unwrap_model(unet).eval(),
        #     scheduler=noise_scheduler,
        #     feature_extractor=feature_extractor,
        #     pose_net=accelerator.unwrap_model(cond_net).eval()
        # )

        
        # The models need unwrapping because for compatibility in distributed training mode.
        pipeline = MimicMotionPipeline_v2(
                            vae=vae,
                            image_encoder=image_encoder,
                            image_enc=image_enc,
                            reference_unet=reference_unet,
                            denoising_unet=denoising_unet,
                            scheduler=noise_scheduler,
                            feature_extractor=feature_extractor,
                            pose_net=cond_net
                        )
            # 以上自己写
        # pipeline = pipeline.to(accelerator.device)
        # pipeline.set_progress_bar_config(disable=True)

        # # run inference
        # val_save_dir = os.path.join(
        #     args.output_dir, "validation_images")

        # if not os.path.exists(val_save_dir):
        #     os.makedirs(val_save_dir)

        # with torch.autocast(
        #         str(accelerator.device).replace(":0", ""),
        #         enabled=accelerator.mixed_precision == "fp16"
        # ):
        # 以下自己
        ref_img_path = "/root/scratch/Moore-AnimateAnyone/lmb/project_result/sample/20240909_1/08/9.9.8.png"
        ref_image = Image.open(ref_img_path).convert("RGB")
        pose_video_path = "/root/scratch/Moore-AnimateAnyone/lmb/project_result/sample/20240909_1/08/9.9.6-pose.mp4"
        ref_pose_image = Image.open(ref_img_path[:-4] + "-pose.png")

        short_size = 576
        w, h = ref_image.size
        scale = short_size / min(w, h)
        ow = int(w * scale // 64 * 64)
        oh = int(h * scale // 64 * 64)
        ref_image = ref_image.resize((ow, oh))


        stride = 2
        pose_images = utils.read_frames(pose_video_path)
        pose_images = [ref_pose_image] + pose_images[::stride]
        pose_images = np.stack([np.array(img.resize((ow, oh))) for img in pose_images])
        pose_pixels = torch.from_numpy(pose_images.copy()) / 127.5 - 1
        pose_pixels = pose_pixels.permute(0, 3, 1, 2)

        tile_size = 24
        tile_overlap = 4
        seed = 42
        # fps = 30 // stride
        pose_vcap = cv2.VideoCapture(pose_video_path)
        fps = int(pose_vcap.get(cv2.CAP_PROP_FPS)) // stride
        pose_vcap.release()

        scale_latents = True

        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        video_frames = pipeline(
            ref_image,[ref_image], image_pose=pose_pixels, num_frames=pose_pixels.size(0),
            tile_size=tile_size, tile_overlap=tile_overlap,
            height=oh, width=ow, fps=7,
            noise_aug_strength=0, num_inference_steps=20,
            generator=generator, min_guidance_scale=2.5,
            max_guidance_scale=2.5, decode_chunk_size=8, output_type="pt",
            device=device, scale_latents=scale_latents
        ).frames.cpu()
        video_frames = (video_frames * 255.0).to(torch.uint8)[0, 1:].permute(
                    (0, 2, 3, 1))
        date_str = datetime.datetime.now().strftime("%m-%d-%H-%M")
        result_dir = "./results/{}-{}".format(MimicMotionPipeline_v2.__name__, date_str)
        os.makedirs(result_dir, exist_ok=True)
        save_vapth = os.path.join(result_dir, os.path.basename(pose_video_path))
        write_video(save_vapth, video_frames, 18)
        print(save_vapth)
            # 以上自己
                # ref_pose_image = Image.open(
                #     os.path.join(val_dir, val_infos[val_name][0][:-4] + "-pose.png"))

                # start_ind = val_infos[val_name][2]
                # cond_images = utils.read_frames(os.path.join(val_dir, val_infos[val_name][1]))
                # cond_images = cond_images[start_ind:start_ind + tile_size * 2 - tile_overlap - 1]
                # cond_images = [ref_pose_image] + cond_images
                # cond_images = np.stack([np.array(img.resize((ow, oh))) for img in cond_images])
                # cond_pixels = torch.from_numpy(cond_images.copy()) / 127.5 - 1
                # cond_pixels = cond_pixels.permute(0, 3, 1, 2)

                    
                    




def test_mimicmotion_pipeline():
    import torch
    import torch.utils.checkpoint
    from diffusers.models import AutoencoderKLTemporalDecoder
    from diffusers.schedulers import EulerDiscreteScheduler
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    import numpy as np
    import os
    import datetime
    from PIL import Image

    from animate_master.models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
    from animate_master.models.pose_net import PoseNet
    from animate_master.pipelines.pipeline_mimicmotion import MimicMotionPipeline
    from animate_master.common import utils
    with torch.no_grad():
        device = torch.device("cuda")
        dtype = torch.float16
        base_model_path = "checkpoints/stable-video-diffusion-img2vid-xt-1-1"
        mimicmotion_model_path = "checkpoints/MimicMotion/MimicMotion_1-1.pth"

        unet = UNetSpatioTemporalConditionModel.from_config(
            UNetSpatioTemporalConditionModel.load_config(base_model_path, subfolder="unet"))
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            base_model_path, subfolder="vae")
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            base_model_path, subfolder="image_encoder")
        noise_scheduler = EulerDiscreteScheduler.from_pretrained(
            base_model_path, subfolder="scheduler")
        feature_extractor = CLIPImageProcessor.from_pretrained(
            base_model_path, subfolder="feature_extractor")
        # pose_net
        pose_net = PoseNet(noise_latent_channels=unet.config.block_out_channels[0])

        mimic_state_dict = torch.load(mimicmotion_model_path, map_location=device)
        unet_state_dict = {key[5:]: val for key, val in mimic_state_dict.items() if key.startswith("unet.")}
        unet.load_state_dict(unet_state_dict, strict=True)
        unet.eval().to(device, dtype=dtype)
        pose_net_state_dict = {key[9:]: val for key, val in mimic_state_dict.items() if key.startswith("pose_net.")}
        pose_net.load_state_dict(pose_net_state_dict, strict=True)
        pose_net.eval().to(dtype=dtype, device=device)
        vae.eval().to(dtype=dtype, device=device)
        image_encoder.eval().to(dtype=dtype, device=device)

        pipe = MimicMotionPipeline(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=noise_scheduler,
            feature_extractor=feature_extractor,
            pose_net=pose_net
        )

        # ref_img_path = "data/matt_tests1/demo4/img.png"
        # ref_image = Image.open(ref_img_path).convert("RGB")
        # pose_video_path = "data/matt_tests1/demo4/001-mimic-pose.mp4"
        # ref_pose_image = Image.open(ref_img_path[:-4] + "-mimic-pose.png")


        ref_img_path = "/root/scratch/Moore-AnimateAnyone/lmb/project_result/sample/20240909_1/06/9.9.6.png"
        ref_image = Image.open(ref_img_path).convert("RGB")
        pose_video_path = "/root/scratch/Moore-AnimateAnyone/lmb/project_result/sample/20240909_1/06/9.9.6-mimic-pose.mp4"
        ref_pose_image = Image.open(ref_img_path[:-4] + "-mimic-pose.png")

        # org_video_path = "./data/tests/videos/4NZWPzWdGyU-002109-002229.mp4"
        # org_video_frames = utils.read_frames(org_video_path)
        # ref_image_path = ""
        # ref_image = org_video_frames[0]
        # pose_video_path = "./data/tests/openpose/4NZWPzWdGyU-002109-002229.mp4"

        short_size = 576
        w, h = ref_image.size
        scale = short_size / min(w, h)
        ow = int(w * scale // 64 * 64)
        oh = int(h * scale // 64 * 64)
        ref_image = ref_image.resize((ow, oh))

        stride = 2
        pose_images = utils.read_frames(pose_video_path)
        pose_images = [ref_pose_image] + pose_images[::stride]
        pose_images = np.stack([np.array(img.resize((ow, oh))) for img in pose_images])
        pose_pixels = torch.from_numpy(pose_images.copy()) / 127.5 - 1
        pose_pixels = pose_pixels.permute(0, 3, 1, 2)

        tile_size = 32
        tile_overlap = 6
        seed = 42
        # fps = 30 // stride
        pose_vcap = cv2.VideoCapture(pose_video_path)
        fps = int(pose_vcap.get(cv2.CAP_PROP_FPS)) // stride
        pose_vcap.release()


        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        frames = pipe(
            [ref_image], image_pose=pose_pixels, num_frames=pose_pixels.size(0),
            tile_size=tile_size, tile_overlap=tile_overlap,
            height=oh, width=ow, fps=7,
            noise_aug_strength=0, num_inference_steps=25,
            generator=generator, min_guidance_scale=2,
            max_guidance_scale=2, decode_chunk_size=8, output_type="pt", device=device
        ).frames.cpu()
        video_frames = (frames * 255.0).to(torch.uint8)[0, 1:].permute((0, 2, 3, 1))
        date_str = datetime.datetime.now().strftime("%m-%d-%H-%M")
        result_dir = "./results/{}-{}".format(MimicMotionPipeline.__name__, date_str)
        os.makedirs(result_dir, exist_ok=True)
        save_vapth = os.path.join(result_dir, os.path.basename(pose_video_path))
        options = {
            'crf': '18',  # 较低的 CRF 值表示更高的质量
            'preset': 'fast',  # 较慢的预设通常会产生更好的质量
            'video_bitrate': '10M'  # 设置目标比特率为 10 Mbps
        }
        write_video(save_vapth, video_frames, fps, options=options)
        print(save_vapth)


def test_animatemaster_pipeline():
    import torch
    import torch.utils.checkpoint
    from diffusers.models import AutoencoderKLTemporalDecoder
    from diffusers.schedulers import EulerDiscreteScheduler
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    import numpy as np
    import os
    import datetime
    from PIL import Image
    from animate_master.models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
    from animate_master.models.pose_net import PoseNet
    from animate_master.pipelines.pipeline_mimicmotion import MimicMotionPipeline
    from animate_master.common import utils
    with torch.no_grad():
        device = torch.device("cuda")
        dtype = torch.float16
        base_model_path = "checkpoints/stable-video-diffusion-img2vid-xt-1-1"
        unet_model_path = "outputs/MimicMotion-20240923170036/checkpoint-10000/unet.pth"
        cond_net_model_path = "outputs/MimicMotion-20240923170036/checkpoint-10000/cond_net_openpose.pth"

        unet = UNetSpatioTemporalConditionModel.from_config(
            base_model_path, subfolder="unet")
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            base_model_path, subfolder="vae")
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            base_model_path, subfolder="image_encoder")
        noise_scheduler = EulerDiscreteScheduler.from_pretrained(
            base_model_path, subfolder="scheduler")
        feature_extractor = CLIPImageProcessor.from_pretrained(
            base_model_path, subfolder="feature_extractor")
        # pose_net
        pose_net = PoseNet(noise_latent_channels=unet.config.block_out_channels[0])

        unet_state_dict = torch.load(unet_model_path)
        unet.load_state_dict(unet_state_dict, strict=False)
        unet.eval().to(device, dtype=dtype)
        pose_net_state_dict = torch.load(cond_net_model_path)
        pose_net.load_state_dict(pose_net_state_dict, strict=True)
        pose_net.eval().to(dtype=dtype, device=device)
        vae.eval().to(dtype=dtype, device=device)
        image_encoder.eval().to(dtype=dtype, device=device)

        pipe = MimicMotionPipeline(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=noise_scheduler,
            feature_extractor=feature_extractor,
            pose_net=pose_net
        )
        stride = 2
        ref_img_path = "data/matt_tests1/demo4/img.png"
        ref_image = Image.open(ref_img_path).convert("RGB")
        pose_video_path = "data/matt_tests1/demo4/001-pose.mp4"
        ref_pose_image = Image.open(ref_img_path[:-4] + "-pose.png").convert("RGB")

        short_size = 768
        w, h = ref_image.size
        scale = short_size / min(w, h)
        ow = int(w * scale // 64 * 64)
        oh = int(h * scale // 64 * 64)
        ref_image = ref_image.resize((ow, oh))

        pose_images = utils.read_frames(pose_video_path)
        pose_images = [ref_pose_image] + pose_images[::stride]
        pose_images = np.stack([np.array(img.resize((ow, oh))) for img in pose_images])
        pose_pixels = torch.from_numpy(pose_images.copy()) / 127.5 - 1
        pose_pixels = pose_pixels.permute(0, 3, 1, 2)

        tile_size = 32
        tile_overlap = 6
        seed = 42
        pose_vcap = cv2.VideoCapture(pose_video_path)
        fps = int(pose_vcap.get(cv2.CAP_PROP_FPS)) // stride
        pose_vcap.release()

        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        frames = pipe(
            [ref_image], image_pose=pose_pixels, num_frames=pose_pixels.size(0),
            tile_size=tile_size, tile_overlap=tile_overlap,
            height=oh, width=ow, fps=8,
            noise_aug_strength=0, num_inference_steps=25,
            generator=generator, min_guidance_scale=3,
            max_guidance_scale=3, decode_chunk_size=8, output_type="pt", device=device,
            scale_latents=True
        ).frames.cpu()
        video_frames = (frames * 255.0).to(torch.uint8)[0, 1:].permute((0, 2, 3, 1))
        date_str = datetime.datetime.now().strftime("%m-%d-%H-%M")
        result_dir = "./results/{}-{}-v1".format(MimicMotionPipeline.__name__, date_str)
        os.makedirs(result_dir, exist_ok=True)
        save_vapth = os.path.join(result_dir, os.path.basename(pose_video_path))
        write_video(save_vapth, video_frames, fps)
        options = {
            'crf': '18',  # 较低的 CRF 值表示更高的质量
            'preset': 'fast',  # 较慢的预设通常会产生更好的质量
            'video_bitrate': '10M'  # 设置目标比特率为 10 Mbps
        }
        write_video(save_vapth, video_frames, fps, options=options)
        print(save_vapth)


def test_animatemaster_v2_pipeline():
    import torch
    import torch.utils.checkpoint
    from diffusers.models import AutoencoderKLTemporalDecoder
    from diffusers.schedulers import EulerDiscreteScheduler
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    import numpy as np
    import os
    import datetime
    from PIL import Image
    from animate_master.models.unet_spatio_temporal_condition_v2 import UNetSpatioTemporalConditionModel
    from animate_master.models.pose_net import PoseNet
    from animate_master.pipelines.pipeline_mimicmotion import MimicMotionPipeline
    from animate_master.common import utils
    with torch.no_grad():
        device = torch.device("cuda")
        dtype = torch.float16
        base_model_path = "checkpoints/stable-video-diffusion-img2vid-xt-1-1"
        unet_model_path = "outputs/MimicMotion-20240923013507/checkpoint-14000/unet.pth"
        cond_net_model_path = "outputs/MimicMotion-20240923013507/checkpoint-14000/cond_net_openpose.pth"

        unet = UNetSpatioTemporalConditionModel.from_config(
            base_model_path, subfolder="unet")
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            base_model_path, subfolder="vae")
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            base_model_path, subfolder="image_encoder")
        noise_scheduler = EulerDiscreteScheduler.from_pretrained(
            base_model_path, subfolder="scheduler")
        feature_extractor = CLIPImageProcessor.from_pretrained(
            base_model_path, subfolder="feature_extractor")
        # pose_net
        pose_net = PoseNet(noise_latent_channels=unet.config.block_out_channels[0])

        unet_state_dict = torch.load(unet_model_path)
        unet.load_state_dict(unet_state_dict, strict=True)
        unet.eval().to(device, dtype=dtype)
        pose_net_state_dict = torch.load(cond_net_model_path)
        pose_net.load_state_dict(pose_net_state_dict, strict=True)
        pose_net.eval().to(dtype=dtype, device=device)
        vae.eval().to(dtype=dtype, device=device)
        image_encoder.eval().to(dtype=dtype, device=device)

        pipe = MimicMotionPipeline(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=noise_scheduler,
            feature_extractor=feature_extractor,
            pose_net=pose_net
        )

        # ref_img_path = "data/matt_tests1/demo1/img.png"
        # ref_image = Image.open(ref_img_path).convert("RGB")
        # pose_video_path = "data/matt_tests1/demo1/001-pose.mp4"
        # ref_pose_image = Image.open(ref_img_path[:-4] + "-pose.png").convert("RGB")

        ref_img_path = "/root/scratch/Moore-AnimateAnyone/lmb/project_result/sample/20240909_1/06/9.9.6.png"
        ref_image = Image.open(ref_img_path).convert("RGB")
        pose_video_path = "/root/scratch/Moore-AnimateAnyone/lmb/project_result/sample/20240909_1/06/9.9.6-pose.mp4"
        ref_pose_image = Image.open(ref_img_path[:-4] + "-pose.png")

        short_size = 768
        w, h = ref_image.size
        scale = short_size / min(w, h)
        ow = int(w * scale // 64 * 64)
        oh = int(h * scale // 64 * 64)
        ref_image = ref_image.resize((ow, oh))
        stride = 2

        pose_images = utils.read_frames(pose_video_path)
        pose_images = pose_images[::stride]
        # pose_images = pose_images[:32*2-6]
        # pose_images = [ref_pose_image] + pose_images[:]
        pose_images = np.stack([np.array(img.resize((ow, oh))) for img in pose_images])
        pose_pixels = torch.from_numpy(pose_images.copy()) / 127.5 - 1
        pose_pixels = pose_pixels.permute(0, 3, 1, 2)

        tile_size = 32
        tile_overlap = 6
        seed = 123456
        pose_vcap = cv2.VideoCapture(pose_video_path)
        fps = int(pose_vcap.get(cv2.CAP_PROP_FPS)) // stride
        pose_vcap.release()

        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        frames = pipe.forward_v2(
            [ref_image], image_pose=pose_pixels, num_frames=pose_pixels.size(0),
            tile_size=tile_size, tile_overlap=tile_overlap,
            height=oh, width=ow, fps=8,
            noise_aug_strength=0, num_inference_steps=25,
            generator=generator, min_guidance_scale=3,
            max_guidance_scale=3, decode_chunk_size=8, output_type="pt", device=device,
            scale_latents=True
        ).frames.cpu()
        video_frames = (frames * 255.0).to(torch.uint8)[0, :].permute((0, 2, 3, 1))
        date_str = datetime.datetime.now().strftime("%m-%d-%H-%M")
        result_dir = "./results/{}-{}-v2".format(MimicMotionPipeline.__name__, date_str)
        os.makedirs(result_dir, exist_ok=True)
        save_vapth = os.path.join(result_dir, os.path.basename(pose_video_path))
        options = {
            'crf': '18',  # 较低的 CRF 值表示更高的质量
            'preset': 'fast',  # 较慢的预设通常会产生更好的质量
            'video_bitrate': '10M'  # 设置目标比特率为 10 Mbps
        }
        write_video(save_vapth, video_frames, fps, options=options)
        print(save_vapth)


def test_animatemaster_v3_pipeline():
    import torch
    import torch.utils.checkpoint
    from diffusers.models import AutoencoderKLTemporalDecoder
    from diffusers.schedulers import EulerDiscreteScheduler
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    import numpy as np
    import os
    import datetime
    from PIL import Image
    from animate_master.models.unet_spatio_temporal_condition_v3 import UNetSpatioTemporalConditionModel
    from animate_master.models.pose_net import PoseNet
    from animate_master.pipelines.pipeline_mimicmotion import MimicMotionPipeline
    from animate_master.common import utils
    with torch.no_grad():
        device = torch.device("cuda")
        dtype = torch.float16
        base_model_path = "checkpoints/stable-video-diffusion-img2vid-xt-1-1"
        unet_model_path = "outputs/MimicMotion-20241016120657/checkpoint-2000/unet.pth"
        cond_net_model_path = "outputs/MimicMotion-20241016120657/checkpoint-2000/cond_net_openpose.pth"

        unet = UNetSpatioTemporalConditionModel.from_config(
            base_model_path, subfolder="unet")
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            base_model_path, subfolder="vae")
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            base_model_path, subfolder="image_encoder")
        noise_scheduler = EulerDiscreteScheduler.from_pretrained(
            base_model_path, subfolder="scheduler")
        feature_extractor = CLIPImageProcessor.from_pretrained(
            base_model_path, subfolder="feature_extractor")
        # pose_net
        pose_net = PoseNet(noise_latent_channels=unet.config.block_out_channels[0])

        unet_state_dict = torch.load(unet_model_path)
        unet.load_state_dict(unet_state_dict, strict=True)
        unet.eval().to(device, dtype=dtype)
        pose_net_state_dict = torch.load(cond_net_model_path)
        pose_net.load_state_dict(pose_net_state_dict, strict=True)
        pose_net.eval().to(dtype=dtype, device=device)
        vae.eval().to(dtype=dtype, device=device)
        image_encoder.eval().to(dtype=dtype, device=device)

        pipe = MimicMotionPipeline(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=noise_scheduler,
            feature_extractor=feature_extractor,
            pose_net=pose_net
        )

        ref_img_path = "data/tuokouxiu/demo6/img.png"
        ref_image = Image.open(ref_img_path).convert("RGB")
        pose_video_path = "data/tuokouxiu/demo6/001-pose.mp4"
        ref_pose_image = Image.open(ref_img_path[:-4] + "-pose.png").convert("RGB")

        short_size = 576
        w, h = ref_image.size
        scale = short_size / min(w, h)
        ow = int(w * scale // 64 * 64)
        oh = int(h * scale // 64 * 64)
        ref_image = ref_image.resize((ow, oh))
        stride = 2

        pose_images = utils.read_frames(pose_video_path)
        pose_images = pose_images[::stride]
        # pose_images = pose_images[:32*2-6]
        pose_images = [ref_pose_image] + pose_images[:]
        pose_images = np.stack([np.array(img.resize((ow, oh))) for img in pose_images])
        pose_pixels = torch.from_numpy(pose_images.copy()) / 127.5 - 1
        pose_pixels = pose_pixels.permute(0, 3, 1, 2)

        tile_size = 32
        tile_overlap = 6
        seed = 1234
        pose_vcap = cv2.VideoCapture(pose_video_path)
        fps = int(pose_vcap.get(cv2.CAP_PROP_FPS)) // stride
        pose_vcap.release()

        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        frames = pipe.forward_v2(
            [ref_image], image_pose=pose_pixels, num_frames=pose_pixels.size(0),
            tile_size=tile_size, tile_overlap=tile_overlap,
            height=oh, width=ow, fps=8,
            noise_aug_strength=0, num_inference_steps=25,
            generator=generator, min_guidance_scale=3,
            max_guidance_scale=3, decode_chunk_size=8, output_type="pt", device=device,
            scale_latents=True
        ).frames.cpu()
        video_frames = (frames * 255.0).to(torch.uint8)[0, 1:].permute((0, 2, 3, 1))
        date_str = datetime.datetime.now().strftime("%m-%d-%H-%M")
        result_dir = "./results/{}-{}-v3".format(MimicMotionPipeline.__name__, date_str)
        os.makedirs(result_dir, exist_ok=True)
        save_vapth = os.path.join(result_dir, os.path.basename(pose_video_path))
        options = {
            'crf': '18',  # 较低的 CRF 值表示更高的质量
            'preset': 'fast',  # 较慢的预设通常会产生更好的质量
            'video_bitrate': '10M'  # 设置目标比特率为 10 Mbps
        }
        write_video(save_vapth, video_frames, fps, options=options)
        print(save_vapth)


def test_musepose_image_pipeline():
    import torch
    from diffusers import AutoencoderKL, DDIMScheduler
    import os
    from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
    from einops import repeat
    from omegaconf import OmegaConf
    from PIL import Image
    from torchvision import transforms
    from transformers import CLIPVisionModelWithProjection
    import numpy as np
    from datetime import datetime
    from animate_master.pipelines.pipeline_musepose_image import MusePoseImagePipeline
    from animate_master.models.unet_2d_condition import UNet2DConditionModel
    from animate_master.models.unet_3d_condition import UNet3DConditionModel
    from animate_master.models.pose_guider import PoseGuider
    from animate_master.common import utils

    device = torch.device("cuda")
    weight_dtype = torch.float16

    pretrained_base_model_path = './checkpoints/sd-image-variations-diffusers'
    pretrained_vae_path = './checkpoints/sd-vae-ft-mse'

    denoising_unet_path = "./checkpoints/MusePose/denoising_unet.pth"
    reference_unet_path = "./checkpoints/MusePose/reference_unet.pth"
    pose_guider_path = "./checkpoints/MusePose/pose_guider.pth"

    vae = AutoencoderKL.from_pretrained(
        pretrained_vae_path,
    ).to("cuda", dtype=weight_dtype)

    reference_unet = UNet2DConditionModel.from_pretrained(
        pretrained_base_model_path,
        subfolder="unet",
    ).to(dtype=weight_dtype, device=device)

    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        pretrained_base_model_path,
        motion_module_path="",
        subfolder="unet",
        unet_additional_kwargs={
            "use_motion_module": False,
            "unet_use_temporal_attention": False,
        },
    ).to(dtype=weight_dtype, device=device)

    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
        dtype=weight_dtype, device=device
    )

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        pretrained_base_model_path,
        subfolder="image_encoder",
    ).to(dtype=weight_dtype, device=device)

    infer_config = OmegaConf.load("configs/inference/musepose_infer.yaml")
    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    width, height = 768, 1024

    # load pretrained weights
    denoising_unet.load_state_dict(
        torch.load(denoising_unet_path, map_location="cpu"),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(reference_unet_path, map_location="cpu"),
    )
    pose_guider.load_state_dict(
        torch.load(pose_guider_path, map_location="cpu"),
    )

    pipe = MusePoseImagePipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )

    pipe = pipe.to(device, dtype=weight_dtype)

    seed = 1234
    cfg = 3.5
    steps = 20

    generator = torch.manual_seed(seed)
    ref_img_path = "data/musepose_tests/demo1/001.png"
    ref_name = os.path.basename(ref_img_path)
    ref_image_pil = Image.open(ref_img_path).convert("RGB")
    pose_video_path = "data/musepose_tests/demo1/001-pose.mp4"
    pose_images = utils.read_frames(pose_video_path)
    pose_name = os.path.basename(pose_video_path)
    pose_image = random.choice(pose_images)

    original_width, original_height = pose_image.size

    image = pipe(
        ref_image_pil,
        pose_image,
        width,
        height,
        steps,
        cfg,
        generator=generator,
    ).images

    image = image.squeeze(2).squeeze(0)  # (c, h, w)
    image = image.transpose(0, 1).transpose(1, 2)  # (h w c)

    image = (image * 255).numpy().astype(np.uint8)
    image = Image.fromarray(image, 'RGB')
    # image.save(os.path.join(save_dir, f"{ref_name}_{pose_name}_{args.H}x{args.W}_{int(args.cfg)}_{time_str}.png"))

    image_grid = Image.new('RGB', (original_width * 3, original_height))
    imgs = [ref_image_pil, pose_image, image]
    x_offset = 0
    date_str = datetime.now().strftime("%m-%d-%H-%M")
    save_dir = "./results/MusePoseImagePipeline-{}".format(date_str)
    os.makedirs(save_dir, exist_ok=True)
    for i, img in enumerate(imgs):
        img = img.resize((original_width * 2, original_height * 2))
        img.save(os.path.join(save_dir, f"res{i}_{ref_name}_{pose_name}_{cfg}_{seed}.jpg"))
        img = img.resize((original_width, original_height))
        image_grid.paste(img, (x_offset, 0))
        x_offset += img.size[0]
    img_save_path = os.path.join(save_dir, f"grid_{ref_name}_{pose_name}_{cfg}_{seed}.jpg")
    image_grid.save(img_save_path)
    print(img_save_path)


def test_musepose_video_pipeline():
    import torch
    from diffusers import AutoencoderKL, DDIMScheduler
    import os
    from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
    from einops import repeat
    from omegaconf import OmegaConf
    from PIL import Image
    from torchvision import transforms
    from transformers import CLIPVisionModelWithProjection
    import numpy as np
    import torch.nn.functional as F

    from datetime import datetime
    from animate_master.pipelines.pipeline_musepose_video import MusePoseVideoPipeline
    from animate_master.models.unet_2d_condition import UNet2DConditionModel
    from animate_master.models.unet_3d_condition import UNet3DConditionModel
    from animate_master.models.pose_guider import PoseGuider
    from animate_master.common import utils

    device = torch.device("cuda")
    weight_dtype = torch.float16

    pretrained_base_model_path = './checkpoints/sd-image-variations-diffusers'
    pretrained_vae_path = './checkpoints/sd-vae-ft-mse'

    denoising_unet_path = "./checkpoints/MusePose/denoising_unet.pth"
    reference_unet_path = "./checkpoints/MusePose/reference_unet.pth"
    pose_guider_path = "./checkpoints/MusePose/pose_guider.pth"
    motion_module_path = "./checkpoints/MusePose/motion_module.pth"
    infer_config = OmegaConf.load("configs/inference/musepose_infer.yaml")

    vae = AutoencoderKL.from_pretrained(
        pretrained_vae_path,
    ).to("cuda", dtype=weight_dtype)

    reference_unet = UNet2DConditionModel.from_pretrained(
        pretrained_base_model_path,
        subfolder="unet",
    ).to(dtype=weight_dtype, device=device)

    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        pretrained_base_model_path,
        motion_module_path=motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=weight_dtype, device=device)

    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
        dtype=weight_dtype, device=device
    )

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        pretrained_base_model_path,
        subfolder="image_encoder",
    ).to(dtype=weight_dtype, device=device)

    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    width, height = 512, 768

    # load pretrained weights
    denoising_unet.load_state_dict(
        torch.load(denoising_unet_path, map_location="cpu"),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(reference_unet_path, map_location="cpu"),
    )
    pose_guider.load_state_dict(
        torch.load(pose_guider_path, map_location="cpu"),
    )

    pipe = MusePoseVideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )

    pipe = pipe.to(device, dtype=weight_dtype)

    seed = 1234
    cfg = 3.5
    steps = 20

    generator = torch.manual_seed(seed)
    ref_img_path = "data/musepose_tests/demo1/001.png"
    ref_image_pil = Image.open(ref_img_path).convert("RGB")
    pose_video_path = "data/musepose_tests/demo1/001-pose.mp4"
    pose_images = utils.read_frames(pose_video_path)

    stride = 1
    S = 48
    O = 4
    pose_list = pose_images[::stride]
    pose_list = pose_list[:2 * S - O]
    L = len(pose_list)

    pose_vcap = cv2.VideoCapture(pose_video_path)
    src_fps = int(pose_vcap.get(cv2.CAP_PROP_FPS)) // stride
    pose_vcap.release()
    print("fps", src_fps)

    # repeart the last segment
    last_segment_frame_num = (L - S) % (S - O)
    repeart_frame_num = (S - O - last_segment_frame_num) % (S - O)
    for i in range(repeart_frame_num):
        pose_list.append(pose_list[-1])
    print("processing length:", len(pose_list))
    original_width, original_height = pose_list[0].size

    frames = pipe.forward_long(
        ref_image_pil,
        pose_list,
        width,
        height,
        len(pose_list),
        steps,
        cfg,
        generator=generator,
        context_frames=S,
        context_stride=1,
        context_overlap=O,
        output_type="tensor"
    ).videos
    date_str = datetime.now().strftime("%m-%d-%H-%M")
    save_dir = "./results/MusePoseVideoPipeline-{}".format(date_str)
    os.makedirs(save_dir, exist_ok=True)

    def scale_video(video, width, height):
        video_reshaped = video.view(-1, *video.shape[2:])  # [batch*frames, channels, height, width]
        scaled_video = F.interpolate(video_reshaped, size=(height, width), mode='bilinear', align_corners=False)
        scaled_video = scaled_video.view(*video.shape[:2], scaled_video.shape[1], height,
                                         width)  # [batch, frames, channels, height, width]

        return scaled_video

    frames = scale_video(frames, original_width, original_height)
    video_frames = (frames * 255.0).to(torch.uint8)[0].permute((1, 2, 3, 0))
    save_vapth = os.path.join(save_dir, os.path.basename(pose_video_path))
    options = {
        'crf': '18',  # 较低的 CRF 值表示更高的质量
        'preset': 'fast',  # 较慢的预设通常会产生更好的质量
        'video_bitrate': '10M'  # 设置目标比特率为 10 Mbps
    }
    write_video(save_vapth, video_frames, src_fps, options=options)
    print(save_vapth)


def test_musepose_video_v1_1_pipeline():
    import torch
    from diffusers import AutoencoderKL, DDIMScheduler
    import os
    from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
    from einops import repeat
    from omegaconf import OmegaConf
    from PIL import Image
    from torchvision import transforms
    from transformers import CLIPVisionModelWithProjection
    import numpy as np
    import torch.nn.functional as F

    from datetime import datetime
    from animate_master.pipelines.pipeline_musepose_video import MusePoseVideoPipeline
    from animate_master.models.unet_2d_condition import UNet2DConditionModel
    from animate_master.models.unet_3d_condition import UNet3DConditionModel
    from animate_master.models.pose_guider import PoseGuider
    from animate_master.common import utils

    device = torch.device("cuda")
    weight_dtype = torch.float16

    pretrained_base_model_path = './checkpoints/sd-image-variations-diffusers'
    pretrained_vae_path = './checkpoints/sd-vae-ft-mse'

    # 清洗数据训练版本
    denoising_unet_path = "./outputs/MusePose_stage1-20241020121326/denoising_unet-7000.pth"
    reference_unet_path = "./outputs/MusePose_stage1-20241020121326/reference_unet-7000.pth"
    pose_guider_path = "./outputs/MusePose_stage1-20241020121326/pose_guider-7000.pth"
    motion_module_path = "./outputs/MusePose_stage2-20241020155915/motion_module-15000.pth"
    infer_config = OmegaConf.load("configs/inference/musepose_infer.yaml")

    vae = AutoencoderKL.from_pretrained(
        pretrained_vae_path,
    ).to("cuda", dtype=weight_dtype)

    reference_unet = UNet2DConditionModel.from_pretrained(
        pretrained_base_model_path,
        subfolder="unet",
    ).to(dtype=weight_dtype, device=device)

    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        pretrained_base_model_path,
        motion_module_path=motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=weight_dtype, device=device)

    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
        dtype=weight_dtype, device=device
    )

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        pretrained_base_model_path,
        subfolder="image_encoder",
    ).to(dtype=weight_dtype, device=device)

    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    width, height = 576, 1024

    # load pretrained weights
    denoising_unet.load_state_dict(
        torch.load(denoising_unet_path, map_location="cpu"),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(reference_unet_path, map_location="cpu"),
    )
    pose_guider.load_state_dict(
        torch.load(pose_guider_path, map_location="cpu"),
    )

    pipe = MusePoseVideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )

    pipe = pipe.to(device, dtype=weight_dtype)

    seed = 1234
    cfg = 3.5
    steps = 20

    generator = torch.manual_seed(seed)
    ref_img_path = "data/yht_tests/demo2/player1v2.jpg"
    ref_image_pil = Image.open(ref_img_path).convert("RGB")
    ref_image_pil = ref_image_pil.resize((width, height))
    pose_video_path = "data/yht_tests/demo2/IMG_7388-pose.mp4"
    pose_images = utils.read_frames(pose_video_path)

    stride = 2
    S = 32
    O = 6
    pose_list = pose_images[::stride]
    # pose_list = pose_list[:2 * S - O]
    original_width, original_height = pose_list[0].size
    pose_list = [img.resize((width, height)) for img in pose_list]
    L = len(pose_list)

    pose_vcap = cv2.VideoCapture(pose_video_path)
    src_fps = int(pose_vcap.get(cv2.CAP_PROP_FPS)) // stride
    pose_vcap.release()
    print("fps", src_fps)

    # repeart the last segment
    last_segment_frame_num = (L - S) % (S - O)
    repeart_frame_num = (S - O - last_segment_frame_num) % (S - O)
    for i in range(repeart_frame_num):
        pose_list.append(pose_list[-1])
    print("processing length:", len(pose_list))

    scale = min(width, height) / min(original_width, original_height)
    original_width = int(original_width * scale // 64 * 64)
    original_height = int(original_height * scale // 64 * 64)

    frames = pipe.forward_long(
        ref_image_pil,
        pose_list,
        width,
        height,
        len(pose_list),
        steps,
        cfg,
        generator=generator,
        context_frames=S,
        context_stride=1,
        context_overlap=O,
        output_type="tensor"
    ).videos
    date_str = datetime.now().strftime("%m-%d-%H-%M")
    save_dir = "./results/MusePoseVideoPipeline-{}".format(date_str)
    os.makedirs(save_dir, exist_ok=True)

    def scale_video(video, width, height):
        video_reshaped = video.view(-1, *video.shape[2:])  # [batch*frames, channels, height, width]
        scaled_video = F.interpolate(video_reshaped, size=(height, width), mode='bilinear', align_corners=False)
        scaled_video = scaled_video.view(*video.shape[:2], scaled_video.shape[1], height,
                                         width)  # [batch, frames, channels, height, width]

        return scaled_video

    frames = scale_video(frames, original_width, original_height)
    video_frames = (frames * 255.0).to(torch.uint8)[0].permute((1, 2, 3, 0))
    save_vapth = os.path.join(save_dir, os.path.basename(pose_video_path))
    options = {
        'crf': '18',  # 较低的 CRF 值表示更高的质量
        'preset': 'fast',  # 较慢的预设通常会产生更好的质量
        'video_bitrate': '10M'  # 设置目标比特率为 10 Mbps
    }
    write_video(save_vapth, video_frames, src_fps, options=options)
    print(save_vapth)


def test_musepose_video_v1_2_pipeline():
    import torch
    from diffusers import AutoencoderKL, DDIMScheduler
    import os
    from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
    from einops import repeat
    from omegaconf import OmegaConf
    from PIL import Image
    from torchvision import transforms
    from transformers import CLIPVisionModelWithProjection
    import numpy as np
    import torch.nn.functional as F

    from datetime import datetime
    from animate_master.pipelines.pipeline_musepose_video import MusePoseVideoPipeline
    from animate_master.models.unet_2d_condition import UNet2DConditionModel
    from animate_master.models.unet_3d_condition import UNet3DConditionModel
    from animate_master.models.pose_guider import PoseGuider
    from animate_master.common import utils

    device = torch.device("cuda")
    weight_dtype = torch.float16

    pretrained_base_model_path = './checkpoints/sd-image-variations-diffusers'
    pretrained_vae_path = './checkpoints/sd-vae-ft-mse'

    # 清洗数据训练版本
    denoising_unet_path = "./outputs/MusePose_stage1-20241020121326/denoising_unet-7000.pth"
    reference_unet_path = "./outputs/MusePose_stage1-20241020121326/reference_unet-7000.pth"
    pose_guider_path = "./outputs/MusePose_stage1-20241020121326/pose_guider-7000.pth"
    motion_module_path = "./outputs/MusePose_stage2-20241020155915/motion_module-15000.pth"
    infer_config = OmegaConf.load("configs/inference/musepose_infer.yaml")

    vae = AutoencoderKL.from_pretrained(
        pretrained_vae_path,
    ).to("cuda", dtype=weight_dtype)

    reference_unet = UNet2DConditionModel.from_pretrained(
        pretrained_base_model_path,
        subfolder="unet",
    ).to(dtype=weight_dtype, device=device)

    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        pretrained_base_model_path,
        motion_module_path=motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=weight_dtype, device=device)

    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
        dtype=weight_dtype, device=device
    )

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        pretrained_base_model_path,
        subfolder="image_encoder",
    ).to(dtype=weight_dtype, device=device)

    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    width, height = 576, 1024

    # load pretrained weights
    denoising_unet.load_state_dict(
        torch.load(denoising_unet_path, map_location="cpu"),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(reference_unet_path, map_location="cpu"),
    )
    pose_guider.load_state_dict(
        torch.load(pose_guider_path, map_location="cpu"),
    )

    pipe = MusePoseVideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )

    pipe = pipe.to(device, dtype=weight_dtype)

    seed = 4224
    cfg = 3.5
    steps = 20

    generator = torch.manual_seed(seed)
    ref_img_path = "data/yht_tests/demo3/player2v3.jpg"
    ref_image_pil = Image.open(ref_img_path).convert("RGB")
    ref_image_pil = ref_image_pil.resize((width, height))
    ref_pose_image = Image.open(ref_img_path[:-4] + "-pose.png").convert("RGB")
    ref_pose_image = ref_pose_image.resize((width, height))
    pose_video_path = "data/yht_tests/demo3/IMG_7387-pose.mp4"
    pose_images = utils.read_frames(pose_video_path)

    stride = 2
    S = 32
    O = 6
    pose_list = pose_images[::stride]
    # pose_list = pose_list[:2 * S - O]
    original_width, original_height = pose_list[0].size
    pose_list = [img.resize((width, height)) for img in pose_list]
    pose_list = [ref_pose_image] + pose_list
    L = len(pose_list)

    pose_vcap = cv2.VideoCapture(pose_video_path)
    src_fps = int(pose_vcap.get(cv2.CAP_PROP_FPS)) // stride
    pose_vcap.release()
    print("fps", src_fps)

    # repeart the last segment
    last_segment_frame_num = (L - S) % (S - O)
    repeart_frame_num = (S - O - last_segment_frame_num) % (S - O)
    for i in range(repeart_frame_num):
        pose_list.append(pose_list[-1])
    print("processing length:", len(pose_list))

    scale = min(width, height) / min(original_width, original_height)
    original_width = int(original_width * scale // 64 * 64)
    original_height = int(original_height * scale // 64 * 64)

    frames = pipe.forward_long_v2(
        ref_image_pil,
        pose_list,
        width,
        height,
        len(pose_list),
        steps,
        cfg,
        generator=generator,
        context_frames=S,
        context_stride=1,
        context_overlap=O,
        output_type="tensor"
    ).videos
    date_str = datetime.now().strftime("%m-%d-%H-%M")
    save_dir = "./results/MusePoseVideoPipeline-{}".format(date_str)
    os.makedirs(save_dir, exist_ok=True)

    def scale_video(video, width, height):
        video_reshaped = video.view(-1, *video.shape[2:])  # [batch*frames, channels, height, width]
        scaled_video = F.interpolate(video_reshaped, size=(height, width), mode='bilinear', align_corners=False)
        scaled_video = scaled_video.view(*video.shape[:2], scaled_video.shape[1], height,
                                         width)  # [batch, frames, channels, height, width]

        return scaled_video

    frames = frames[:, :, 1:]
    frames = scale_video(frames, original_width, original_height)
    video_frames = (frames * 255.0).to(torch.uint8)[0].permute((1, 2, 3, 0))
    save_vapth = os.path.join(save_dir, os.path.basename(pose_video_path))
    options = {
        'crf': '18',  # 较低的 CRF 值表示更高的质量
        'preset': 'fast',  # 较慢的预设通常会产生更好的质量
        'video_bitrate': '10M'  # 设置目标比特率为 10 Mbps
    }
    write_video(save_vapth, video_frames, src_fps, options=options)
    print(save_vapth)


def test_musepose_video_v2_1_pipeline():
    import torch
    from diffusers import AutoencoderKL, DDIMScheduler
    import os
    from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
    from einops import repeat
    from omegaconf import OmegaConf
    from PIL import Image
    from torchvision import transforms
    from transformers import CLIPVisionModelWithProjection
    import numpy as np
    import torch.nn.functional as F

    from datetime import datetime
    from animate_master.pipelines.pipeline_musepose_video_v2 import MusePoseVideoPipeline
    from animate_master.models.unet_2d_condition import UNet2DConditionModel
    from animate_master.models.unet_3d_condition import UNet3DConditionModel
    from animate_master.models.pose_guider import PoseGuider
    from animate_master.common import utils

    device = torch.device("cuda")
    weight_dtype = torch.float16

    pretrained_base_model_path = './checkpoints/sd-image-variations-diffusers'
    pretrained_vae_path = './checkpoints/sd-vae-ft-mse'

    denoising_unet_path = "./outputs/MusePose_stage1-20241015101418/denoising_unet-5000.pth"
    reference_unet_path = "./outputs/MusePose_stage1-20241015101418/reference_unet-5000.pth"
    pose_guider_path = "./outputs/MusePose_stage1-20241015101418/pose_guider-5000.pth"
    motion_module_path = "./outputs/MusePose_stage2-20241017040330/motion_module-12000.pth"
    infer_config = OmegaConf.load("configs/inference/musepose_infer.yaml")

    vae = AutoencoderKL.from_pretrained(
        pretrained_vae_path,
    ).to(device, dtype=weight_dtype)

    reference_unet = UNet2DConditionModel.from_pretrained(
        pretrained_base_model_path,
        subfolder="unet",
    ).to(dtype=weight_dtype, device=device)
    print(motion_module_path)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        pretrained_base_model_path,
        motion_module_path=motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=weight_dtype, device=device)

    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
        dtype=weight_dtype, device=device
    )

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        pretrained_base_model_path,
        subfolder="image_encoder",
    ).to(dtype=weight_dtype, device=device)

    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    width, height = 576, 1024

    # load pretrained weights
    print(denoising_unet_path)
    denoising_unet.load_state_dict(
        torch.load(denoising_unet_path, map_location="cpu"),
        strict=False,
    )
    print(reference_unet_path)
    reference_unet.load_state_dict(
        torch.load(reference_unet_path, map_location="cpu"),
    )
    print(pose_guider_path)
    pose_guider.load_state_dict(
        torch.load(pose_guider_path, map_location="cpu"),
    )

    pipe = MusePoseVideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )

    pipe = pipe.to(device, dtype=weight_dtype)

    seed = 1234
    cfg = 3.5
    steps = 20

    generator = torch.manual_seed(seed)
    ref_img_path = "data/yht_tests_1011/demo3/9.9.5.jpg"
    ref_image_pil = Image.open(ref_img_path).convert("RGB")
    ref_image_pil = ref_image_pil.resize((width, height))
    ref_pose_image = Image.open(ref_img_path[:-4] + "-pose.png").convert("RGB")
    ref_pose_image = ref_pose_image.resize((width, height))
    pose_video_path = "data/yht_tests_1011/demo3/9.9.4-pose.mp4"
    pose_images = utils.read_frames(pose_video_path)

    stride = 2
    S = 32
    O = 6
    pose_list = pose_images[::stride]
    original_width, original_height = pose_list[0].size
    pose_list = [img.resize((width, height)) for img in pose_list]
    # pose_list = pose_list[:2 * S - O]
    L = len(pose_list)

    pose_vcap = cv2.VideoCapture(pose_video_path)
    src_fps = int(pose_vcap.get(cv2.CAP_PROP_FPS)) // stride
    pose_vcap.release()
    print("fps", src_fps)

    # repeart the last segment
    last_segment_frame_num = (L - S) % (S - O)
    repeart_frame_num = (S - O - last_segment_frame_num) % (S - O)
    for i in range(repeart_frame_num):
        pose_list.append(pose_list[-1])
    print("processing length:", len(pose_list))
    scale = min(width, height) / min(original_width, original_height)
    original_width = int(original_width * scale // 64 * 64)
    original_height = int(original_height * scale // 64 * 64)

    frames = pipe.forward_long(
        ref_image_pil,
        pose_list,
        width,
        height,
        len(pose_list),
        steps,
        cfg,
        generator=generator,
        context_frames=S,
        context_stride=1,
        context_overlap=O,
        output_type="tensor"
    ).videos
    date_str = datetime.now().strftime("%m-%d-%H-%M")
    save_dir = "./results/MusePoseVideoPipeline-{}".format(date_str)
    os.makedirs(save_dir, exist_ok=True)

    def scale_video(video, width, height):
        video_reshaped = video.view(-1, *video.shape[2:])  # [batch*frames, channels, height, width]
        scaled_video = F.interpolate(video_reshaped, size=(height, width), mode='bilinear', align_corners=False)
        scaled_video = scaled_video.view(*video.shape[:2], scaled_video.shape[1], height,
                                         width)  # [batch, frames, channels, height, width]

        return scaled_video

    frames = scale_video(frames, original_width, original_height)
    video_frames = (frames * 255.0).to(torch.uint8)[0].permute((1, 2, 3, 0))
    save_vapth = os.path.join(save_dir, os.path.basename(pose_video_path))
    options = {
        'crf': '18',  # 较低的 CRF 值表示更高的质量
        'preset': 'fast',  # 较慢的预设通常会产生更好的质量
        'video_bitrate': '10M'  # 设置目标比特率为 10 Mbps
    }
    write_video(save_vapth, video_frames, src_fps, options=options)
    print(save_vapth)


def test_musepose_video_v2_2_pipeline():
    import torch
    from diffusers import AutoencoderKL, DDIMScheduler
    import os
    from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
    from einops import repeat
    from omegaconf import OmegaConf
    from PIL import Image
    from torchvision import transforms
    from transformers import CLIPVisionModelWithProjection
    import numpy as np
    import torch.nn.functional as F

    from datetime import datetime
    from animate_master.pipelines.pipeline_musepose_video_v2 import MusePoseVideoPipeline
    from animate_master.models.unet_2d_condition import UNet2DConditionModel
    from animate_master.models.unet_3d_condition import UNet3DConditionModel
    from animate_master.models.pose_guider import PoseGuider
    from animate_master.common import utils

    device = torch.device("cuda")
    weight_dtype = torch.float16

    pretrained_base_model_path = './checkpoints/sd-image-variations-diffusers'
    pretrained_vae_path = './checkpoints/sd-vae-ft-mse'

    # v1 版本: 使用所有的数据
    # denoising_unet_path = "./outputs/MusePose_stage1-20240929112134/denoising_unet-8000.pth"
    # reference_unet_path = "./outputs/MusePose_stage1-20240929112134/reference_unet-8000.pth"
    # pose_guider_path = "./outputs/MusePose_stage1-20240929112134/pose_guider-8000.pth"
    # motion_module_path = "./outputs/MusePose_stage2-20240930003905/motion_module-66000.pth"
    # infer_config = OmegaConf.load("configs/inference/musepose_infer.yaml")

    # v2 版本: 仅使用清洗的数据
    # denoising_unet_path = "./outputs/MusePose_stage1-20241015101418/denoising_unet-5000.pth"
    # reference_unet_path = "./outputs/MusePose_stage1-20241015101418/reference_unet-5000.pth"
    # pose_guider_path = "./outputs/MusePose_stage1-20241015101418/pose_guider-5000.pth"
    # motion_module_path = "./outputs/MusePose_stage2-20241015125637/motion_module-12000.pth"
    # infer_config = OmegaConf.load("configs/inference/musepose_infer.yaml")

    vae = AutoencoderKL.from_pretrained(
        pretrained_vae_path,
    ).to(device, dtype=weight_dtype)

    reference_unet = UNet2DConditionModel.from_pretrained(
        pretrained_base_model_path,
        subfolder="unet",
    ).to(dtype=weight_dtype, device=device)
    print(motion_module_path)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        pretrained_base_model_path,
        motion_module_path=motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=weight_dtype, device=device)

    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
        dtype=weight_dtype, device=device
    )

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        pretrained_base_model_path,
        subfolder="image_encoder",
    ).to(dtype=weight_dtype, device=device)

    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    width, height = 768, 1024

    # load pretrained weights
    print(denoising_unet_path)
    denoising_unet.load_state_dict(
        torch.load(denoising_unet_path, map_location="cpu"),
        strict=False,
    )
    print(reference_unet_path)
    reference_unet.load_state_dict(
        torch.load(reference_unet_path, map_location="cpu"),
    )
    print(pose_guider_path)
    pose_guider.load_state_dict(
        torch.load(pose_guider_path, map_location="cpu"),
    )

    pipe = MusePoseVideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )

    pipe = pipe.to(device, dtype=weight_dtype)

    seed = 1234
    cfg = 3.5
    steps = 20

    generator = torch.manual_seed(seed)
    ref_img_path = "data/yht_tests/demo3/player2v3.jpg"
    ref_image_pil = Image.open(ref_img_path).convert("RGB")
    ref_image_pil = ref_image_pil.resize((width, height))
    ref_pose_image = Image.open(ref_img_path[:-4] + "-pose.png").convert("RGB")
    ref_pose_image = ref_pose_image.resize((width, height))
    pose_video_path = "data/yht_tests/demo3/IMG_7387-pose.mp4"
    pose_images = utils.read_frames(pose_video_path)

    stride = 2
    S = 32
    O = 6
    pose_list = pose_images[::stride]
    original_width, original_height = pose_list[0].size
    pose_list = [img.resize((width, height)) for img in pose_list]
    pose_list = [ref_pose_image] + pose_list
    # pose_list = pose_list[:2 * S - O]
    L = len(pose_list)

    pose_vcap = cv2.VideoCapture(pose_video_path)
    src_fps = int(pose_vcap.get(cv2.CAP_PROP_FPS)) // stride
    pose_vcap.release()
    print("fps", src_fps)

    # repeart the last segment
    last_segment_frame_num = (L - S) % (S - O)
    repeart_frame_num = (S - O - last_segment_frame_num) % (S - O)
    for i in range(repeart_frame_num):
        pose_list.append(pose_list[-1])
    print("processing length:", len(pose_list))
    scale = min(width, height) / min(original_width, original_height)
    original_width = int(original_width * scale // 64 * 64)
    original_height = int(original_height * scale // 64 * 64)

    frames = pipe.forward_long_v2(
        ref_image_pil,
        pose_list,
        width,
        height,
        len(pose_list),
        steps,
        cfg,
        generator=generator,
        context_frames=S,
        context_stride=1,
        context_overlap=O,
        output_type="tensor"
    ).videos
    date_str = datetime.now().strftime("%m-%d-%H-%M")
    save_dir = "./results/MusePoseVideoPipeline-{}".format(date_str)
    os.makedirs(save_dir, exist_ok=True)

    def scale_video(video, width, height):
        video_reshaped = video.view(-1, *video.shape[2:])  # [batch*frames, channels, height, width]
        scaled_video = F.interpolate(video_reshaped, size=(height, width), mode='bilinear', align_corners=False)
        scaled_video = scaled_video.view(*video.shape[:2], scaled_video.shape[1], height,
                                         width)  # [batch, frames, channels, height, width]

        return scaled_video

    frames = frames[:, :, 1:]
    frames = scale_video(frames, original_width, original_height)
    video_frames = (frames * 255.0).to(torch.uint8)[0].permute((1, 2, 3, 0))
    save_vapth = os.path.join(save_dir, os.path.basename(pose_video_path))
    options = {
        'crf': '18',  # 较低的 CRF 值表示更高的质量
        'preset': 'fast',  # 较慢的预设通常会产生更好的质量
        'video_bitrate': '10M'  # 设置目标比特率为 10 Mbps
    }
    write_video(save_vapth, video_frames, src_fps, options=options)
    print(save_vapth)


if __name__ == '__main__':
    # test_crop_video_pipeline()
    # test_split_video_pipeline()
    # test_mimicmotion_pipeline()
    # test_animatemaster_pipeline()
    # test_animatemaster_v2_pipeline()
    # test_animatemaster_v3_pipeline()
    # test_musepose_image_pipeline()
    # test_musepose_video_pipeline()
    # test_musepose_video_v1_1_pipeline()
    # test_musepose_video_v1_2_pipeline()
    # test_musepose_video_v2_1_pipeline()
    # test_musepose_video_v2_2_pipeline()

    # mimicmotion_referencenet实验
    test_mimicmotion_referencenet_pipeline()
