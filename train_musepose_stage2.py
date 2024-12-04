"""
CUDA_VISIBLE_DEVICES=1 accelerate launch train_musepose_stage2.py \
--config configs/train/musepose_train_stage2.yaml
"""
import os
import os.path as osp
import argparse
import logging
import math
import pdb
import random
import warnings
import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import cv2
from torchvision.io import write_video
from animate_master.common import utils
from einops import rearrange
import diffusers
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

from animate_master.datasets.musepose_video_dataset import MusePoseVideoDataset
from animate_master.models.mutual_self_attention import ReferenceAttentionControl
from animate_master.models.pose_guider import PoseGuider
from animate_master.models.unet_2d_condition import UNet2DConditionModel
from animate_master.models.unet_3d_condition import UNet3DConditionModel
from animate_master.pipelines.pipeline_musepose_video import MusePoseVideoPipeline
from animate_master.common.utils import delete_additional_ckpt, seed_everything

warnings.filterwarnings("ignore")

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")


class Net(nn.Module):
    def __init__(
            self,
            reference_unet: UNet2DConditionModel,
            denoising_unet: UNet3DConditionModel,
            pose_guider: PoseGuider,
            reference_control_writer,
            reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    def forward(
            self,
            noisy_latents,
            timesteps,
            ref_image_latents,
            clip_image_embeds,
            pose_img,
            uncond_fwd: bool = False,
    ):
        pose_cond_tensor = pose_img.to(device="cuda")
        pose_fea = self.pose_guider(pose_cond_tensor)

        if not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            self.reference_unet(
                ref_image_latents,
                ref_timesteps,
                encoder_hidden_states=clip_image_embeds,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)

        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            pose_cond_fea=pose_fea,
            encoder_hidden_states=clip_image_embeds,
        ).sample

        return model_pred


def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod ** 0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device
    )[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


@torch.no_grad()
def log_validation(
        vae,
        image_enc,
        net,
        scheduler,
        accelerator,
        seed,
        global_step,
        width,
        height,
        save_dir
):
    logger.info("Running validation... ")
    os.makedirs(save_dir, exist_ok=True)

    ori_net = accelerator.unwrap_model(net)
    reference_unet = ori_net.reference_unet
    denoising_unet = ori_net.denoising_unet
    pose_guider = ori_net.pose_guider

    generator = torch.Generator(accelerator.device).manual_seed(seed)

    pipe = MusePoseVideoPipeline(
        vae=accelerator.unwrap_model(vae).eval(),
        image_encoder=accelerator.unwrap_model(image_enc).eval(),
        reference_unet=accelerator.unwrap_model(reference_unet).eval(),
        denoising_unet=accelerator.unwrap_model(denoising_unet).eval(),
        pose_guider=accelerator.unwrap_model(pose_guider).eval(),
        scheduler=scheduler,
    )
    pipe = pipe.to(accelerator.device)

    def scale_video(video, width, height):
        video_reshaped = video.view(-1, *video.shape[2:])  # [batch*frames, channels, height, width]
        scaled_video = F.interpolate(video_reshaped, size=(height, width), mode='bilinear', align_corners=False)
        scaled_video = scaled_video.view(*video.shape[:2], scaled_video.shape[1], height,
                                         width)  # [batch, frames, channels, height, width]

        return scaled_video

    with torch.autocast(
            str(accelerator.device).replace(":0", ""),
            enabled=accelerator.mixed_precision == "fp16"
    ):
        val_infos = {
            # "data/yht_tests/demo1": ["move1.JPG", "IMG_8812-pose.mp4", 40],
            "data/yht_tests/demo2": ["player1v2.jpg", "IMG_7388-pose.mp4", 30],
            "data/yht_tests/demo3": ["player2v3.jpg", "IMG_7387-pose.mp4", 30],
            # "data/yht_tests/demo4": ["img.png", "verse3725-pose.mp4", 100],
            # "data/yht_tests/demo8": ["img.png", "0725-pose.mp4", 70],
            # "data/yht_tests/demo9": ["onana.JPG", "IMG_5494-pose.mp4", 70],
            # "data/yht_tests/demo10": ["onna1.png", "IMG_8876-pose.mp4", 30],
            "data/yht_tests/demo11": ["onanav2.png", "throw-pose.mp4", 30],
            "data/yht_tests/demo13": ["picture-9.png", "videos_3-pose.mp4", 30],
        }
        for val_dir in val_infos:
            val_name = os.path.basename(val_dir)
            ref_img_path = os.path.join(val_dir, val_infos[val_dir][0])
            ref_image_pil = Image.open(ref_img_path).convert("RGB")
            pose_video_path = os.path.join(val_dir, val_infos[val_dir][1])
            pose_images = utils.read_frames(pose_video_path)

            short_size = min(height, width)
            w, h = ref_image_pil.size
            scale = short_size / min(w, h)
            # ow = int(w * scale // 64 * 64)
            # oh = int(h * scale // 64 * 64)
            ow = width
            oh = height
            ref_image_pil = ref_image_pil.resize((ow, oh))

            start_ind = val_infos[val_dir][2]
            stride = 1
            S = 32
            O = 6
            pose_list = pose_images[::stride]
            pose_list = pose_list[start_ind:start_ind + 2 * S - O]
            original_width, original_height = pose_list[0].size
            scale = min(width, height) / min(original_width, original_height)
            original_width = int(original_width * scale // 2 * 2)
            original_height = int(original_height * scale // 2 * 2)

            pose_list = [img.resize((ow, oh)) for img in pose_list]
            L = len(pose_list)

            pose_vcap = cv2.VideoCapture(pose_video_path)
            src_fps = int(pose_vcap.get(cv2.CAP_PROP_FPS)) // stride
            pose_vcap.release()
            print("fps", src_fps)

            frames = pipe.forward_long(
                ref_image_pil,
                pose_list,
                ow,
                oh,
                len(pose_list),
                20,
                3.5,
                generator=generator,
                context_frames=S,
                context_stride=1,
                context_overlap=O,
                output_type="tensor"
            ).videos
            frames = scale_video(frames, original_width, original_height)
            video_frames = (frames * 255.0).to(torch.uint8)[0].permute((1, 2, 3, 0))
            save_vapth = os.path.join(save_dir, f"{val_name}_{seed}_{global_step}.mp4")
            options = {
                'crf': '18',  # 较低的 CRF 值表示更高的质量
                'preset': 'fast',  # 较慢的预设通常会产生更好的质量
                'video_bitrate': '8M'  # 设置目标比特率为 10 Mbps
            }
            write_video(save_vapth, video_frames, src_fps, options=options)
            print(save_vapth)

    ReferenceAttentionControl(
        reference_unet,
        mode="write",
        fusion_blocks="full",
        do_classifier_free_guidance=False
    )
    ReferenceAttentionControl(
        denoising_unet,
        mode="read",
        fusion_blocks="full",
        do_classifier_free_guidance=False
    )


def main(cfg):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        mixed_precision=cfg.solver.mixed_precision,
        log_with="mlflow",
        project_dir="./mlruns",
        kwargs_handlers=[kwargs],
    )

    if accelerator.is_local_main_process:
        from torch.utils.tensorboard import SummaryWriter
        tensorboard_writer = SummaryWriter(log_dir=f"{cfg.output_dir}/{cfg.exp_name}/loss")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    date_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    exp_name = cfg.exp_name
    save_dir = f"{cfg.output_dir}/{exp_name}-{date_str}"
    if accelerator.is_main_process and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training"
        )

    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    train_noise_scheduler = DDIMScheduler(**sched_kwargs)

    clip_image_processor = CLIPImageProcessor()
    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        cfg.pretrained_base_model_path,
        subfolder="image_encoder",
    ).to(dtype=weight_dtype, device=accelerator.device)

    vae = AutoencoderKL.from_pretrained(cfg.pretrained_vae_path).to(
        accelerator.device, dtype=weight_dtype
    )
    reference_unet = UNet2DConditionModel.from_pretrained(
        cfg.pretrained_base_model_path,
        subfolder="unet",
    ).to(accelerator.device, dtype=weight_dtype)

    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        cfg.pretrained_base_model_path,
        cfg.motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=OmegaConf.to_container(
            cfg.unet_additional_kwargs
        ),
    ).to(accelerator.device)

    pose_guider = PoseGuider(
        conditioning_embedding_channels=320, block_out_channels=(16, 32, 96, 256)
    ).to(accelerator.device, dtype=weight_dtype)

    # load pretrained weights
    if cfg.denoising_unet_path and os.path.exists(cfg.denoising_unet_path):
        logger.info(f"loading pretrained denoising_unet_path: {cfg.denoising_unet_path}")
        denoising_unet.load_state_dict(
            torch.load(cfg.denoising_unet_path, map_location="cpu"),
            strict=False,
        )
    if cfg.reference_unet_path and os.path.exists(cfg.reference_unet_path):
        logger.info(f"loading pretrained reference_unet_path: {cfg.reference_unet_path}")
        reference_unet.load_state_dict(
            torch.load(cfg.reference_unet_path, map_location="cpu"),
        )
    if cfg.pose_guider_path and os.path.exists(cfg.pose_guider_path):
        logger.info(f"loading pretrained pose_guider_path: {cfg.pose_guider_path}")
        pose_guider.load_state_dict(
            torch.load(cfg.pose_guider_path, map_location="cpu"),
        )

    # Freeze
    vae.requires_grad_(False)
    image_enc.requires_grad_(False)
    reference_unet.requires_grad_(False)
    denoising_unet.requires_grad_(False)
    pose_guider.requires_grad_(False)

    # Set motion module learnable
    for name, module in denoising_unet.named_modules():
        if "motion_modules" in name:
            for params in module.parameters():
                params.requires_grad = True

    reference_control_writer = ReferenceAttentionControl(
        reference_unet,
        mode="write",
        fusion_blocks="full",
    )
    reference_control_reader = ReferenceAttentionControl(
        denoising_unet,
        mode="read",
        fusion_blocks="full",
    )

    net = Net(
        reference_unet,
        denoising_unet,
        pose_guider,
        reference_control_writer,
        reference_control_reader,
    )

    if cfg.solver.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            reference_unet.enable_xformers_memory_efficient_attention()
            denoising_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if cfg.solver.gradient_checkpointing:
        reference_unet.enable_gradient_checkpointing()
        denoising_unet.enable_gradient_checkpointing()

    if cfg.solver.scale_lr:
        learning_rate = (
                cfg.solver.learning_rate
                * cfg.solver.gradient_accumulation_steps
                * cfg.data.train_bs
                * accelerator.num_processes
        )
    else:
        learning_rate = cfg.solver.learning_rate

    # Initialize the optimizer
    if cfg.solver.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = list(filter(lambda p: p.requires_grad, net.parameters()))
    logger.info(f"Total trainable params {len(trainable_params)}")
    optimizer = optimizer_cls(
        trainable_params,
        lr=learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )

    # Scheduler
    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_warmup_steps
                         * cfg.solver.gradient_accumulation_steps,
        num_training_steps=cfg.solver.max_train_steps
                           * cfg.solver.gradient_accumulation_steps,
    )

    train_dataset = MusePoseVideoDataset(
        img_size=(cfg.data.train_height, cfg.data.train_width),
        img_scale=(0.8, 1.0),
        img_ratio=(0.5, 0.6),
        n_sample_frames=cfg.data.n_sample_frames,
        cond_with_mask=[],
        meta_paths=[
            # "./data/TikTok",
            # "./data/UBC_fashion",
            # "./data/cartoon_0830",
            # "./data/cartoon_0831",
            # "./data/youtube_0818",
            # "./data/youtube_0821",
            # "./data/youtube_man",
            # "./data/yht",
            # "./data/youtube_man_0907",
            "./data/first_data_1015"
        ],
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.data.train_bs, shuffle=True, num_workers=cfg.data.train_bs, drop_last=True
    )

    # Prepare everything with our `accelerator`.
    (
        net,
        optimizer,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        net,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.solver.gradient_accumulation_steps
    )
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(
        cfg.solver.max_train_steps / num_update_steps_per_epoch
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        accelerator.init_trackers(
            exp_name,
            init_kwargs={"mlflow": {"run_name": run_time}},
        )
        # dump config file
        mlflow.log_dict(OmegaConf.to_container(cfg), "config.yaml")

    # Train!
    total_batch_size = (
            cfg.data.train_bs
            * accelerator.num_processes
            * cfg.solver.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.data.train_bs}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {cfg.solver.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {cfg.solver.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            net.train()
            with accelerator.accumulate(net):
                # Convert videos to latent space
                pixel_values_tgt = batch["pixel_values_tgt"].to(weight_dtype).to(
                    accelerator.device, non_blocking=True
                )
                pixel_values_cond = batch["pixel_values_cond"].to(weight_dtype).to(
                    accelerator.device, non_blocking=True
                )
                pixel_values_ref = batch["pixel_values_ref"].to(weight_dtype).to(
                    accelerator.device, non_blocking=True
                )
                if "mask_values_cond" in batch:
                    mask_values_cond = batch["mask_values_cond"].to(weight_dtype).to(
                        accelerator.device, non_blocking=True
                    )

                with torch.no_grad():
                    org_B = pixel_values_tgt.shape[0]
                    org_H, org_W = pixel_values_tgt.shape[-2:]
                    # 随机选择新的宽度和高度
                    # if random.random() < 0.5:
                    #     new_width = org_W
                    #     new_height = random.choice(range(768, org_H + 1, 64))
                    #     pixel_values_tgt = rearrange(pixel_values_tgt, "b f c h w -> (b f) c h w")
                    #     pixel_values_tgt = F.interpolate(pixel_values_tgt, size=(new_height, new_width),
                    #                                      mode='bilinear', align_corners=True)
                    #     pixel_values_tgt = rearrange(pixel_values_tgt, "(b f) c h w -> b f c h w", b=org_B)
                    #
                    #     pixel_values_cond = rearrange(pixel_values_cond, "b f c h w -> (b f) c h w")
                    #     pixel_values_cond = F.interpolate(pixel_values_cond, size=(new_height, new_width),
                    #                                       mode='bilinear', align_corners=True)
                    #     pixel_values_cond = rearrange(pixel_values_cond, "(b f) c h w -> b f c h w", b=org_B)
                    #
                    #     pixel_values_ref = F.interpolate(pixel_values_ref, size=(new_height, new_width),
                    #                                      mode='bilinear',
                    #                                      align_corners=True)
                    #     if "mask_values_cond" in batch:
                    #         mask_values_cond = rearrange(mask_values_cond, "b f c h w -> (b f) c h w")
                    #         mask_values_cond = F.interpolate(mask_values_cond, size=(new_height, new_width),
                    #                                          mode='nearest')
                    #         mask_values_cond = rearrange(mask_values_cond, "(b f) c h w -> b f c h w", b=org_B)

                with torch.no_grad():
                    video_length = pixel_values_tgt.shape[1]
                    pixel_values_tgt = rearrange(
                        pixel_values_tgt, "b f c h w -> (b f) c h w"
                    )
                    latents = vae.encode(pixel_values_tgt).latent_dist.sample()
                    latents = rearrange(
                        latents, "(b f) c h w -> b c f h w", f=video_length
                    )
                    latents = latents * 0.18215

                noise = torch.randn_like(latents)
                if cfg.noise_offset > 0:
                    noise += cfg.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1, 1),
                        device=latents.device,
                    )
                bsz = latents.shape[0]
                # Sample a random timestep for each video
                timesteps = torch.randint(
                    0,
                    train_noise_scheduler.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                pixel_values_cond = pixel_values_cond.transpose(
                    1, 2
                )  # (bs, c, f, H, W)
                uncond_fwd = random.random() < cfg.uncond_ratio
                with torch.no_grad():
                    ref_image_latents = vae.encode(
                        pixel_values_ref
                    ).latent_dist.sample()  # (bs, d, 64, 64)
                    ref_image_latents = ref_image_latents * 0.18215

                    pixel_values_clip = F.interpolate(pixel_values_ref, (224, 224), mode="bilinear", align_corners=True)
                    pixel_values_clip = (pixel_values_clip + 1) / 2.0
                    pixel_values_clip = clip_image_processor(
                        images=pixel_values_clip,
                        do_normalize=True,
                        do_center_crop=False,
                        do_resize=False,
                        do_rescale=False,
                        return_tensors="pt",
                    ).pixel_values
                    clip_image_embeds = image_enc(
                        pixel_values_clip.to(accelerator.device, dtype=weight_dtype)
                    ).image_embeds
                    image_prompt_embeds = clip_image_embeds.unsqueeze(1)  # (bs, 1, d)
                    if uncond_fwd:
                        image_prompt_embeds = torch.zeros_like(image_prompt_embeds)
                        if "mask_values_cond" in batch:
                            mask_values_cond = torch.zeros_like(mask_values_cond)

                # add noise
                noisy_latents = train_noise_scheduler.add_noise(
                    latents, noise, timesteps
                )

                # Get the target for loss depending on the prediction type
                if train_noise_scheduler.prediction_type == "epsilon":
                    target = noise
                elif train_noise_scheduler.prediction_type == "v_prediction":
                    target = train_noise_scheduler.get_velocity(
                        latents, noise, timesteps
                    )
                else:
                    raise ValueError(
                        f"Unknown prediction type {train_noise_scheduler.prediction_type}"
                    )

                # ---- Forward!!! -----
                model_pred = net(
                    noisy_latents,
                    timesteps,
                    ref_image_latents,
                    image_prompt_embeds,
                    pixel_values_cond,
                    uncond_fwd
                )

                if "mask_values_cond" in batch:
                    mask_values_cond = rearrange(mask_values_cond, "b f c h w -> (b f) c h w")
                    mask_values_cond = F.interpolate(mask_values_cond, size=model_pred.shape[-2:], mode='nearest')
                    mask_values_cond = rearrange(mask_values_cond, "(b f) c h w -> b c f h w", b=org_B)
                    mask_scale = mask_values_cond.float() * 0.1 + 1.0
                else:
                    mask_scale = torch.ones_like(model_pred)

                if cfg.snr_gamma == 0:
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    ) * mask_scale
                    loss = loss.mean()
                else:
                    snr = compute_snr(train_noise_scheduler, timesteps)
                    if train_noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective requires that we add one to SNR values before we divide by them.
                        snr = snr + 1
                    mse_loss_weights = (
                            torch.stack(
                                [snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1
                            ).min(dim=1)[0]
                            / snr
                    )
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    ) * mask_scale
                    loss = (
                            loss.mean(dim=list(range(1, len(loss.shape))))
                            * mse_loss_weights
                    )
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(cfg.data.train_bs)).mean()
                train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                # if accelerator.sync_gradients:
                #     accelerator.clip_grad_norm_(
                #         trainable_params,
                #         cfg.solver.max_grad_norm,
                #     )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                reference_control_reader.clear()
                reference_control_writer.clear()
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                if accelerator.is_main_process:
                    tensorboard_writer.add_scalar('train_loss', train_loss, global_step)
                train_loss = 0.0

                if global_step % cfg.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        unwrap_net = accelerator.unwrap_model(net)
                        save_checkpoint(
                            unwrap_net.denoising_unet,
                            save_dir,
                            "motion_module",
                            global_step,
                            total_limit=3,
                        )

                if global_step == 0 or \
                        global_step % cfg.val.validation_steps == 0:
                    if accelerator.is_main_process:
                        log_validation(
                            vae=vae,
                            image_enc=image_enc,
                            net=net,
                            scheduler=val_noise_scheduler,
                            accelerator=accelerator,
                            seed=cfg.seed,
                            global_step=global_step,
                            width=cfg.data.train_width,
                            height=cfg.data.train_height,
                            save_dir=os.path.join(save_dir, "validation")
                        )

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0]
            }
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.solver.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


def save_checkpoint(model, save_dir, prefix, ckpt_num, total_limit=None):
    save_path = osp.join(save_dir, f"{prefix}-{ckpt_num}.pth")

    if total_limit is not None:
        checkpoints = os.listdir(save_dir)
        checkpoints = [d for d in checkpoints if d.startswith(prefix)]
        checkpoints = sorted(
            checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
        )

        if len(checkpoints) >= total_limit:
            num_to_remove = len(checkpoints) - total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(save_dir, removing_checkpoint)
                os.remove(removing_checkpoint)

    state_dict = model.state_dict()
    from collections import OrderedDict
    mm_state_dict = OrderedDict()
    for key in state_dict:
        if "motion_module" in key:
            mm_state_dict[key] = state_dict[key]
    torch.save(mm_state_dict, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train_stage_2.yaml")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    main(config)
