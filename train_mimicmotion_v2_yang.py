# -*- coding: utf-8 -*-
"""
CUDA_VISIBLE_DEVICES=0 accelerate launch train_mimicmotion_v2_yang.py \
    --pretrained_model_name_or_path=/project_data/ws_projects/AnimateMaster/checkpoints/stable-video-diffusion-img2vid-xt-1-1 \
    --pretrain_unet /project_data/ws_projects/AnimateMaster/outputs/MimicMotion-20240923013507/checkpoint-14000/unet.pth \
    --pretrain_cond_net /project_data/ws_projects/AnimateMaster/outputs/MimicMotion-20240923013507/checkpoint-14000/cond_net_openpose.pth \
    --pretrained_vae_path=/project_data/ws_projects/AnimateMaster/checkpoints/sd-vae-ft-mse \
    --pretrained_base_model_path=/project_data/ws_projects/AnimateMaster/checkpoints/sd-image-variations-diffusers \
    --reference_unet_path=/project_data/ws_projects/AnimateMaster/outputs/MusePose_stage1-20241021112227/reference_unet-4000.pth \
    --per_gpu_batch_size=1 \
    --gradient_accumulation_steps=1  \
    --max_train_steps=50000 \
    --width=576 \
    --height=1024 \
    --num_frames 10 \
    --conditioning_dropout_prob 0.01 \
    --checkpointing_steps=1000 \
    --checkpoints_total_limit=3 \
    --learning_rate=1e-5 \
    --lr_warmup_steps=100 \
    --seed=1234 \
    --mixed_precision="fp16" \
    --validation_steps=500 \
    --gradient_checkpointing \
    --enable_xformers_memory_efficient_attention \
    --use_8bit_adam \
    --seed 1234
"""

import argparse
import datetime
import pdb
import random
import logging
import math
import os
import cv2
import shutil
from pathlib import Path
from urllib.parse import urlparse
from torchvision.io import write_video
import accelerate
import numpy as np
import PIL
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import RandomSampler
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from einops import rearrange

import diffusers
from diffusers import StableVideoDiffusionPipeline
from diffusers.models.lora import LoRALinearLayer
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, load_image
from diffusers.utils.import_utils import is_xformers_available

from animate_master.models.unet_spatio_temporal_condition_v2 import UNetSpatioTemporalConditionModel
from animate_master.models.pose_net import PoseNet
from animate_master.pipelines.pipeline_mimicmotion_yang import MimicMotionPipeline_v2
from animate_master.datasets.mimicmotion_video_dataset import MimicMotionVideoDataset
from animate_master.common import utils

# 以下自己加
import torch.nn as nn
from animate_master.models.unet_2d_condition import UNet2DConditionModel
from animate_master.models.mutual_self_attention import ReferenceAttentionControl
from diffusers import AutoencoderKL, DDIMScheduler
# 以上自己加

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")

logger = get_logger(__name__, log_level="INFO")

# 以下为自己写
class Net(nn.Module):
    def __init__(
            self,
            reference_unet: UNet2DConditionModel,
            denoising_unet: UNetSpatioTemporalConditionModel,
            reference_control_writer,
            reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    def forward(
            self,
            ref_image_latents,
            image_prompt_embeds,
            inp_noisy_latents,
            timesteps,
            ref_latents,
            added_time_ids,
            encoder_hidden_states,
            pose_latents
    ):
        # pose_cond_tensor = pose_img.to(device="cuda")
        # pose_fea = self.pose_guider(pose_cond_tensor)
        ref_timesteps = torch.zeros_like(timesteps)
        # 
        self.reference_unet(
            ref_image_latents,
            ref_timesteps,
            encoder_hidden_states=image_prompt_embeds,
            return_dict=False,
        )
        self.reference_control_reader.update(self.reference_control_writer)

        model_pred = self.denoising_unet(
            inp_noisy_latents,
            timesteps,
            encoder_hidden_states,
            added_time_ids=added_time_ids,
            pose_latents=pose_latents.flatten(0, 1)
        ).sample

        return model_pred
# 以上为自己写


# copy from https://github.com/crowsonkb/k-diffusion.git
def rand_log_normal(shape, loc=0., scale=1., device='cpu', dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()


# resizing utils
# TODO: clean up later
def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(
        input, size=size, mode=interpolation, align_corners=align_corners)
    return output


def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(
        device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(
        input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device,
                      dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out


def export_to_video(video_frames, output_video_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(
        output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)


def export_to_gif(frames, output_gif_path, fps):
    """
    Export a list of frames to a GIF.

    Args:
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - output_gif_path (str): Path to save the output GIF.
    - duration_ms (int): Duration of each frame in milliseconds.

    """
    # Convert numpy arrays to PIL Images if needed
    pil_frames = [Image.fromarray(frame) if isinstance(
        frame, np.ndarray) else frame for frame in frames]

    pil_frames[0].save(output_gif_path.replace('.mp4', '.gif'),
                       format='GIF',
                       append_images=pil_frames[1:],
                       save_all=True,
                       duration=500,
                       loop=0)


def tensor_to_vae_latent(t, vae, scale=True):
    video_length = t.shape[1]

    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
    if scale:
        latents = latents * vae.config.scaling_factor

    return latents


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train Stable Video Diffusion."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        # default='checkpoints/stable-video-diffusion-img2vid-xt-1-1',
        default='/project_data/ws_projects/AnimateMaster/checkpoints/stable-video-diffusion-img2vid-xt-1-1',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=576,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=576,
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=5,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the text/image prompt"
            " multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=1235, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--per_gpu_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=50000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=0.01,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--use_ema", action="store_true", help="Whether to use EMA model."
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=2,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )

    parser.add_argument(
        "--pretrain_unet",
        type=str,
        # default='outputs/MimicMotion-20240911160402/checkpoint-12000/unet.pth',
        default='/project_data/ws_projects/AnimateMaster/outputs/MimicMotion-20240923013507/checkpoint-14000/unet.pth',
        help="use weight for unet block",
    )

    # 以下自己
    # pretrained_base_model_path


    # pretrained_vae_path
    parser.add_argument(
            "--pretrained_vae_path",
            type=str,
            default='/project_data/ws_projects/AnimateMaster/checkpoints/sd-vae-ft-mse',

        )

    parser.add_argument(
        "--pretrained_base_model_path",
        type=str,
        default='/project_data/ws_projects/AnimateMaster/checkpoints/sd-image-variations-diffusers',

    )
    # reference_unet_path: "./outputs/MusePose_stage1-20241021112227/reference_unet-4000.pth"
    parser.add_argument(
        "--reference_unet_path",
        type=str,
        default='/project_data/ws_projects/AnimateMaster/outputs/MusePose_stage1-20241021112227/reference_unet-4000.pth',

    )
    # 以上自己

    parser.add_argument(
        "--pretrain_cond_net",
        type=str,
        # default='outputs/MimicMotion-20240911160402/checkpoint-12000/cond_net_openpose.pth',
        default='/project_data/ws_projects/AnimateMaster/outputs/MimicMotion-20240923013507/checkpoint-14000/cond_net_openpose.pth',
        help="use weight for cond net block",
    )

    parser.add_argument(
        "--cond_type",
        type=str,
        default='openpose',
        help="conditional type",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


# copy from https://github.com/crowsonkb/k-diffusion.git
def stratified_uniform(shape, group=0, groups=1, dtype=None, device=None):
    """Draws stratified samples from a uniform distribution."""
    if groups <= 0:
        raise ValueError(f"groups must be positive, got {groups}")
    if group < 0 or group >= groups:
        raise ValueError(f"group must be in [0, {groups})")
    n = shape[-1] * groups
    offsets = torch.arange(group, n, groups, dtype=dtype, device=device)
    u = torch.rand(shape, dtype=dtype, device=device)
    return (offsets + u) / n


def rand_cosine_interpolated(shape, image_d, noise_d_low, noise_d_high, sigma_data=1., min_value=1e-3, max_value=1e3,
                             device='cpu', dtype=torch.float32):
    """Draws samples from an interpolated cosine timestep distribution (from simple diffusion)."""

    def logsnr_schedule_cosine(t, logsnr_min, logsnr_max):
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))
        return -2 * torch.log(torch.tan(t_min + t * (t_max - t_min)))

    def logsnr_schedule_cosine_shifted(t, image_d, noise_d, logsnr_min, logsnr_max):
        shift = 2 * math.log(noise_d / image_d)
        return logsnr_schedule_cosine(t, logsnr_min - shift, logsnr_max - shift) + shift

    def logsnr_schedule_cosine_interpolated(t, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max):
        logsnr_low = logsnr_schedule_cosine_shifted(
            t, image_d, noise_d_low, logsnr_min, logsnr_max)
        logsnr_high = logsnr_schedule_cosine_shifted(
            t, image_d, noise_d_high, logsnr_min, logsnr_max)
        return torch.lerp(logsnr_low, logsnr_high, t)

    logsnr_min = -2 * math.log(min_value / sigma_data)
    logsnr_max = -2 * math.log(max_value / sigma_data)
    u = stratified_uniform(
        shape, group=0, groups=1, dtype=dtype, device=device
    )
    logsnr = logsnr_schedule_cosine_interpolated(
        u, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max)
    return torch.exp(-logsnr / 2) * sigma_data


def main():
    args = parse_args()

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    date_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    args.output_dir = os.path.join(args.output_dir, f"MimicMotion-{date_str}")
    os.makedirs(args.output_dir, exist_ok=True)
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir)
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        # kwargs_handlers=[ddp_kwargs]
    )

    generator = torch.Generator(
        device=accelerator.device).manual_seed(args.seed)

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training.")
        import wandb

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
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load img encoder, tokenizer and models.
    feature_extractor = CLIPImageProcessor.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="feature_extractor", revision=args.revision
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    




    # 以下自己加
    # 将之前的位置移动到了这里
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    # 
    vae_2d = AutoencoderKL.from_pretrained(args.pretrained_vae_path).to(
        accelerator.device, dtype=weight_dtype
    )

    clip_image_processor = CLIPImageProcessor()
    image_enc = CLIPVisionModelWithProjection.from_pretrained(
    args.pretrained_base_model_path,
    subfolder="image_encoder",
    ).to(dtype=weight_dtype, device=accelerator.device)

    reference_unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_base_model_path,
        subfolder="unet",
    ).to(accelerator.device)


    if args.reference_unet_path and os.path.exists(args.reference_unet_path):
        logger.info(f"loading pretrained reference_unet_path: {args.reference_unet_path}")
        reference_unet.load_state_dict(
            torch.load(args.reference_unet_path, map_location="cpu"),
        )
    # 以上自己加



    noise_scheduler = EulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")
    # 以下自己加
    # unet = UNetSpatioTemporalConditionModel.from_config(
    #     args.pretrained_model_name_or_path, subfolder="unet"
    # )
    denoising_unet = UNetSpatioTemporalConditionModel.from_config(
        args.pretrained_model_name_or_path, subfolder="unet"
    )
    # if args.pretrain_unet:
    #     unet.load_state_dict(torch.load(args.pretrain_unet, weights_only=True), strict=False)
    if args.pretrain_unet:
        denoising_unet.load_state_dict(torch.load(args.pretrain_unet, weights_only=True), strict=False)
    # 以上自己加

    # pose_net
    if args.cond_type in ['openpose']:
        # 以下自己加
        # cond_net = PoseNet(noise_latent_channels=unet.config.block_out_channels[0])
        cond_net = PoseNet(noise_latent_channels=denoising_unet.config.block_out_channels[0])
        # 以上自己加
    else:
        raise NotImplementedError
    if args.pretrain_cond_net:
        cond_net.load_state_dict(torch.load(args.pretrain_cond_net, weights_only=True), strict=True)

    # Freeze vae and image_encoder
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    cond_net.requires_grad_(False)
    # 以下自己加
    vae_2d.requires_grad_(False)
    image_enc.requires_grad_(False)
    # unet.requires_grad_(False)
    denoising_unet.requires_grad_(True)
    reference_unet.requires_grad_(True)

    #  Some top layer parames of reference_unet don't need grad
    for name, param in reference_unet.named_parameters():
        if "up_blocks.3" in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)

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
    unet = Net(
        reference_unet,
        denoising_unet,
        reference_control_writer,
        reference_control_reader,
        )

    # 以上自己加



    # Move image_encoder and vae to gpu and cast to weight_dtype
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    # 其余在加载模型的时候都自己加过了
    # cond_net.to(accelerator.device, dtype=weight_dtype)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(
        ), model_cls=UNetSpatioTemporalConditionModel, model_config=unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            # 以下自己加
            # unet.enable_xformers_memory_efficient_attention()
            reference_unet.enable_xformers_memory_efficient_attention()
            denoising_unet.enable_xformers_memory_efficient_attention()
            # 以上自己加
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(
                    input_dir, "unet_ema"), UNetSpatioTemporalConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNetSpatioTemporalConditionModel.from_pretrained(
                    input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)
    # 以下自己加
    if args.gradient_checkpointing:
        # unet.enable_gradient_checkpointing()
        reference_unet.enable_gradient_checkpointing()
        denoising_unet.enable_gradient_checkpointing()
    # 以上自己加
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps *
                args.per_gpu_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    parameters_list = []
    # Customize the parameters that need to be trained; if necessary, you can uncomment them yourself.
    # unet
    # 以下自己改
    # 遍历模型的参数 denoising_unet
    # for name, param in unet.named_parameters():
    for name, param in denoising_unet.named_parameters():
        # if 'temporal_transformer_block' in name or "pose_warp_attn" in name:
        #     param.requires_grad = True
        #     parameters_list.append(param)
        # else:
        #     param.requires_grad = False
        param.requires_grad = True
        # parameters_list.append(param)
    # 以上自己改
    # cond net
    for name, param in cond_net.named_parameters():
        param.requires_grad = True
        # parameters_list.append(param)

    print(f"trainable params len:{len(parameters_list)}")
    # 以下自己改
    trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
    optimizer = optimizer_cls(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    # 以上自己改
    # check parameters
    if accelerator.is_main_process:
        rec_txt1 = open('params_freeze.txt', 'w')
        rec_txt2 = open('params_train.txt', 'w')
        for name, para in unet.named_parameters():
            if para.requires_grad is False:
                rec_txt1.write(f'{name}\n')
            else:
                rec_txt2.write(f'{name}\n')
        rec_txt1.close()
        rec_txt2.close()

    # DataLoaders creation:
    args.global_batch_size = args.per_gpu_batch_size * accelerator.num_processes
    train_dataset = MimicMotionVideoDataset(
        img_size=(args.height, args.width),
        img_scale=(0.8, 1.0),
        img_ratio=(0.5, 0.6),
        sample_rate=1,
        n_sample_frames=args.num_frames,
        cond_type=args.cond_type,
        meta_paths=[
            "/project_data/ws_projects/AnimateMaster/data/TikTok",
            "/project_data/ws_projects/AnimateMaster/data/UBC_fashion",
            "/project_data/ws_projects/AnimateMaster/data/cartoon_0830",
            "/project_data/ws_projects/AnimateMaster/data/cartoon_0831",
            "/project_data/ws_projects/AnimateMaster/data/youtube_man",
            "/project_data/ws_projects/AnimateMaster/data/yht",
            "/project_data/ws_projects/AnimateMaster/data/youtube_0818",
            "/project_data/ws_projects/AnimateMaster/data/youtube_0821",
            "/project_data/ws_projects/AnimateMaster/data/youtube_man_0907",
           #  "/project_data/ws_projects/AnimateMaster/data/TikTok"
        ],
    )
    print(f"total num of videos >>>>>>>>>: {len(train_dataset)}")
    sampler = RandomSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.per_gpu_batch_size,
        num_workers=args.num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    cond_net.dtype = torch.float32
    # Prepare everything with our `accelerator`.
    unet, cond_net, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        unet, cond_net, optimizer, lr_scheduler, train_dataloader
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # attribute handling for models using DDP
    if isinstance(unet, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        unet = unet.module

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("SVDXtend", config=vars(args))

    # Train!
    total_batch_size = args.per_gpu_batch_size * \
                       accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_gpu_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    def encode_image(pixel_values):
        # pixel: [-1, 1]
        pixel_values = _resize_with_antialiasing(pixel_values, (224, 224))
        # We unnormalize it after resizing.
        pixel_values = (pixel_values + 1.0) / 2.0

        # Normalize the image with for CLIP input
        pixel_values = feature_extractor(
            images=pixel_values,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        ).pixel_values

        pixel_values = pixel_values.to(
            device=accelerator.device, dtype=weight_dtype)
        image_embeddings = image_encoder(pixel_values).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)
        return image_embeddings

    def _get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            dtype,
            batch_size,
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]
        # 以下自己写 denoising_unet
        # passed_add_embed_dim = unet.config.addition_time_embed_dim * \
        #                        len(add_time_ids)
        # expected_add_embed_dim = unet.add_embedding.linear_1.in_features

        passed_add_embed_dim = denoising_unet.config.addition_time_embed_dim * \
                               len(add_time_ids)
        expected_add_embed_dim = denoising_unet.add_embedding.linear_1.in_features
        # 以上自己写 
        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size, 1)
        return add_time_ids

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                    num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    scale_latents = True

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            unet.train()
            # cond_net.train()
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet, cond_net):
                pixel_values_vid = batch["pixel_values_vid"].to(weight_dtype).to(
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

                if random.random() > 0.:
                    with torch.no_grad():
                        org_B, org_F, org_C, org_H, org_W = pixel_values_vid.shape
                        # 随机选择新的宽度和高度
                        new_width = org_W
                        # new_width = random.choice(range(320, org_W + 1, 64))  # 320到576之间的随机宽度，步长为64
                        new_height = random.choice(range(576, org_H + 1, 64))  # 576到1024之间的随机高度，步长为64
                        # 计算缩放比例
                        scale_factor_w = new_width / org_W
                        scale_factor_h = new_height / org_H
                        # 缩放因子影响到F的维度，面积缩放比例为w*h，F的缩放因子为sqrt(w*h)
                        scale_factor_f = (scale_factor_w * scale_factor_h) ** 0.5
                        scale_factor_f = int(scale_factor_f * 10) / 10.0
                        # new_F = min(int(16 / scale_factor_f), org_F)
                        new_F = org_F
                        start_ind = random.randint(0, org_F - new_F)
                        pixel_values_vid = rearrange(pixel_values_vid, "b f c h w -> (b f) c h w")
                        pixel_values_vid = F.interpolate(pixel_values_vid, size=(new_height, new_width),
                                                         mode='bilinear', align_corners=True)
                        pixel_values_vid = rearrange(pixel_values_vid, "(b f) c h w -> b f c h w", b=org_B)
                        pixel_values_vid = pixel_values_vid[:, start_ind:start_ind + new_F].contiguous()

                        pixel_values_cond = rearrange(pixel_values_cond, "b f c h w -> (b f) c h w")
                        pixel_values_cond = F.interpolate(pixel_values_cond, size=(new_height, new_width),
                                                          mode='bilinear', align_corners=True)
                        pixel_values_cond = rearrange(pixel_values_cond, "(b f) c h w -> b f c h w", b=org_B)
                        pixel_values_cond = pixel_values_cond[:, start_ind:start_ind + new_F].contiguous()

                        # 以下自己
                        ref_image_latents = vae_2d.encode(pixel_values_ref).latent_dist.sample()  
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
                        clip_image_embeds =  image_enc(
                        pixel_values_clip.to(accelerator.device, dtype=weight_dtype)
                        ).image_embeds
                        image_prompt_embeds = clip_image_embeds.unsqueeze(1) 
                        # 以上自己

                        pixel_values_ref = F.interpolate(pixel_values_ref, size=(new_height, new_width), mode='bilinear',
                                                         align_corners=True)
                        if "mask_values_cond" in batch:
                            mask_values_cond = rearrange(mask_values_cond, "b f c h w -> (b f) c h w")
                            mask_values_cond = F.interpolate(mask_values_cond, size=(new_height, new_width), mode='nearest')
                            mask_values_cond = rearrange(mask_values_cond, "(b f) c h w -> b f c h w", b=org_B)
                            mask_values_cond = mask_values_cond[:, start_ind:start_ind + new_F].contiguous()

                # first, convert images to latent space.
                latents = tensor_to_vae_latent(pixel_values_vid, vae)
                bsz = latents.shape[0]

                if "mask_values_cond" in batch:
                    mask_values_cond = rearrange(mask_values_cond, "b f c h w -> (b f) c h w")
                    mask_values_cond = F.interpolate(mask_values_cond, size=latents.shape[-2:], mode='nearest')
                    mask_values_cond = rearrange(mask_values_cond, "(b f) c h w -> b f c h w", b=bsz)

                # Get the text embedding for conditioning.
                encoder_hidden_states = encode_image(pixel_values_ref)

                train_noise_aug = 0.0
                pixel_values_ref = pixel_values_ref + train_noise_aug * torch.randn_like(pixel_values_ref)
                ref_latents = tensor_to_vae_latent(pixel_values_ref[:, None], vae, scale=scale_latents)[:, 0]

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                # sigmas = rand_cosine_interpolated(shape=[bsz, ], image_d=image_d, noise_d_low=noise_d_low,
                #                                   noise_d_high=noise_d_high, sigma_data=sigma_data, min_value=min_value,
                #                                   max_value=max_value).to(latents.device, dtype=weight_dtype)

                # sigmas = rand_log_normal(shape=[bsz,], loc=0.7, scale=1.6).to(latents)
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                # sigmas_reshaped = sigmas.clone()
                # while len(sigmas_reshaped.shape) < len(latents.shape):
                #     sigmas_reshaped = sigmas_reshaped.unsqueeze(-1)
                # noisy_latents = latents + noise * sigmas_reshaped
                # timesteps = torch.Tensor([0.25 * sigma.log() for sigma in sigmas]).to(latents.device,
                #                                                                       dtype=weight_dtype)
                #
                # inp_noisy_latents = noisy_latents / ((sigmas_reshaped ** 2 + 1) ** 0.5)

                sigmas = rand_log_normal(shape=[bsz, ], loc=0.7, scale=1.6).to(latents.device)
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                sigmas = sigmas[:, None, None, None, None]
                noisy_latents = latents + noise * sigmas
                timesteps = torch.Tensor(
                    [0.25 * sigma.log() for sigma in sigmas]).to(accelerator.device)
                inp_noisy_latents = noisy_latents / ((sigmas ** 2 + 1) ** 0.5)

                # Here I input a fixed numerical value for 'motion_bucket_id', which is not reasonable.
                # However, I am unable to fully align with the calculation method of the motion score,
                # so I adopted this approach. The same applies to the 'fps' (frames per second).
                added_time_ids = _get_add_time_ids(
                    7,  # fixed
                    127,  # motion_bucket_id = 127, fixed
                    train_noise_aug,  # noise_aug_strength == cond_sigmas
                    encoder_hidden_states.dtype,
                    bsz,
                )
                added_time_ids = added_time_ids.to(latents.device)

                # cond net
                pixel_values_cond = rearrange(pixel_values_cond, "b f c h w -> (b f) c h w")
                pose_latents = cond_net(pixel_values_cond)
                pose_latents = rearrange(pose_latents, "(b f) c h w -> b f c h w", b=bsz)

                # Conditioning dropout to support classifier-free guidance during inference. For more details
                # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
                if args.conditioning_dropout_prob is not None:
                    random_p = torch.rand(
                        bsz, device=latents.device, generator=generator)
                    # Sample masks for the edit prompts.
                    prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                    # Final text conditioning.
                    null_conditioning = torch.zeros_like(encoder_hidden_states)
                    encoder_hidden_states = torch.where(
                        prompt_mask, null_conditioning, encoder_hidden_states)
                    # Sample masks for the original images.
                    image_mask_dtype = ref_latents.dtype
                    image_mask = 1 - (
                            (random_p >= args.conditioning_dropout_prob).to(
                                image_mask_dtype)
                            * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
                    )
                    image_mask = image_mask.reshape(bsz, 1, 1, 1)
                    # Final image conditioning.
                    ref_latents = image_mask * ref_latents
                    pose_latents = image_mask[:, None] * pose_latents

                # Concatenate the `conditional_latents` with the `noisy_latents`.
                ref_latents = ref_latents.unsqueeze(1).repeat(1, noisy_latents.shape[1], 1, 1, 1)
                inp_noisy_latents = torch.cat(
                    [inp_noisy_latents, ref_latents], dim=2)

                # check https://arxiv.org/abs/2206.00364(the EDM-framework) for more details.
                target = latents
                # 以下自己
                model_pred = unet(ref_image_latents,
                                  image_prompt_embeds,
                                    inp_noisy_latents,
                                    timesteps,
                                    ref_latents,
                                    added_time_ids,
                                    encoder_hidden_states,
                                    pose_latents
                                )
                # 以上自己
                # Denoise the latents
                c_out = -sigmas / ((sigmas ** 2 + 1) ** 0.5)
                c_skip = 1 / (sigmas ** 2 + 1)
                denoised_latents = model_pred * c_out + c_skip * noisy_latents
                weighing = (1 + sigmas ** 2) * (sigmas ** -2.0)

                # MSE loss
                if "mask_values_cond" in batch:
                    mask_scale = mask_values_cond.float() * 0.2 + 1.0
                else:
                    mask_scale = torch.ones_like(denoised_latents)
                loss = torch.mean(
                    (weighing.float() * (denoised_latents.float() -
                                         target.float()) ** 2 * mask_scale).reshape(target.shape[0], -1),
                    dim=1,
                ).mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(
                    loss.repeat(args.per_gpu_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                # if accelerator.sync_gradients:
                #     accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                # 以下自己
                reference_control_reader.clear()
                reference_control_writer.clear()
                # 以上自己
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process:
                    # save checkpoints!
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(
                                    checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}")
                        # accelerator.save_state(save_path)
                        os.makedirs(save_path, exist_ok=True)
                        unet_ = accelerator.unwrap_model(unet)
                        if args.use_ema:
                            ema_unet.copy_to(unet_.parameters())
                        cond_net_ = accelerator.unwrap_model(cond_net)
                        # 以下自己
                        # torch.save(unet_.state_dict(), os.path.join(save_path, "unet.pth"))
                        torch.save(unet_.reference_unet.state_dict(), os.path.join(save_path, "reference_unet.pth"))
                        torch.save(unet_.denoising_unet.state_dict(), os.path.join(save_path, "denoising_unet.pth"))
                        # 以上自己
                        torch.save(cond_net_.state_dict(), os.path.join(save_path, f"cond_net_{args.cond_type}.pth"))
                        logger.info(f"Saved state to {save_path}")

                    # sample images!
                    if (
                            (global_step % args.validation_steps == 0)
                            # or (global_step == 1)
                    ):
                        with torch.no_grad():
                            logger.info(
                                f"Running validation... \n"
                            )
                            # create pipeline
                            if args.use_ema:
                                # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                                ema_unet.store(unet.parameters())
                                ema_unet.copy_to(unet.parameters())
                            # The models need unwrapping because for compatibility in distributed training mode.
                            # 以下自己写
                            
                            # # reference_unet = unet_.reference_unet
                            # unet_ = accelerator.unwrap_model(unet)
                            # unet = unet_.denoising_unet
                            # pose_guider = ori_net.pose_guider
                            ori_net = accelerator.unwrap_model(unet)
                            reference_unet = ori_net.reference_unet
                            denoising_unet = ori_net.denoising_unet
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
                                vae=accelerator.unwrap_model(vae),
                                image_encoder=accelerator.unwrap_model(image_encoder),
                                image_enc=accelerator.unwrap_model(image_enc),
                                # unet=accelerator.unwrap_model(unet),
                                reference_unet=accelerator.unwrap_model(reference_unet),
                                denoising_unet=accelerator.unwrap_model(denoising_unet),
                                scheduler=noise_scheduler,
                                feature_extractor=feature_extractor,
                                pose_net=accelerator.unwrap_model(cond_net)
                            )
                             # 以上自己写
                            pipeline = pipeline.to(accelerator.device)
                            # pipeline.set_progress_bar_config(disable=True)

                            # run inference
                            val_save_dir = os.path.join(
                                args.output_dir, "validation_images")

                            if not os.path.exists(val_save_dir):
                                os.makedirs(val_save_dir)

                            with torch.autocast(
                                    str(accelerator.device).replace(":0", ""),
                                    enabled=accelerator.mixed_precision == "fp16"
                            ):
                                val_infos = {
                                    "demo1": ["move1.JPG", "IMG_8812-pose.mp4", 10],
                                    "demo3": ["player2v3.jpg", "IMG_7387-pose.mp4", 30],
                                    "demo4": ["img.png", "verse3725-pose.mp4", 90],
                                    "demo8": ["img.png", "0725-pose.mp4", 40],
                                    "demo9": ["onana.JPG", "IMG_5494-pose.mp4", 40],
                                    "demo10": ["onna1.png", "IMG_8876-pose.mp4", 30]
                                }
                                tile_size = args.num_frames
                                tile_overlap = 6
                                for val_name in val_infos:
                                    val_dir = os.path.join("data/yht_tests", val_name)
                                    ref_image = Image.open(os.path.join(val_dir, val_infos[val_name][0]))
                                    short_size = min(args.height, args.width)
                                    w, h = ref_image.size
                                    scale = short_size / min(w, h)
                                    ow = int(w * scale // 64 * 64)
                                    oh = int(h * scale // 64 * 64)
                                    ref_image = ref_image.resize((ow, oh))

                                    ref_pose_image = Image.open(
                                        os.path.join(val_dir, val_infos[val_name][0][:-4] + "-pose.png"))

                                    start_ind = val_infos[val_name][2]
                                    cond_images = utils.read_frames(os.path.join(val_dir, val_infos[val_name][1]))
                                    cond_images = cond_images[start_ind:start_ind + tile_size * 2 - tile_overlap - 1]
                                    cond_images = [ref_pose_image] + cond_images
                                    cond_images = np.stack([np.array(img.resize((ow, oh))) for img in cond_images])
                                    cond_pixels = torch.from_numpy(cond_images.copy()) / 127.5 - 1
                                    cond_pixels = cond_pixels.permute(0, 3, 1, 2)
                                    with torch.no_grad():
                                        video_frames = pipeline(
                                            ref_image,
                                            [ref_image], image_pose=cond_pixels, num_frames=cond_pixels.size(0),
                                            tile_size=tile_size, tile_overlap=tile_overlap,
                                            height=cond_pixels.size(-2), width=cond_pixels.size(-1), fps=7,
                                            noise_aug_strength=train_noise_aug, num_inference_steps=20,
                                            generator=generator, min_guidance_scale=2.5,
                                            max_guidance_scale=2.5, decode_chunk_size=8, output_type="pt",
                                            device=accelerator.device, scale_latents=scale_latents
                                        ).frames.cpu()
                                        video_frames = (video_frames * 255.0).to(torch.uint8)[0, 1:].permute(
                                            (0, 2, 3, 1))
                                        save_vapth = os.path.join(
                                            val_save_dir,
                                            f"step_{global_step}_val_img_{val_name}.mp4",
                                        )
                                        write_video(save_vapth, video_frames, 8)
                                        print(save_vapth)

                            if args.use_ema:
                                # Switch back to the original UNet parameters.
                                ema_unet.restore(unet.parameters())

            accelerator.wait_for_everyone()
            logs = {"step_loss": loss.detach().item(
            ), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet_ = accelerator.unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet_.parameters())
        cond_net_ = accelerator.unwrap_model(cond_net)
        save_path = os.path.join(
            args.output_dir, f"checkpoint-final")
        # accelerator.save_state(save_path)
        os.makedirs(save_path, exist_ok=True)
        torch.save(unet_.denoising_unet.state_dict(), os.path.join(save_path, "unet.pth"))
        torch.save(cond_net_.state_dict(), os.path.join(save_path, f"cond_net_{args.cond_type}.pth"))
    accelerator.end_training()


if __name__ == "__main__":
    main()
