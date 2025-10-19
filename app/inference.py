from __future__ import annotations

from typing import List

import numpy as np
import torch
from diffusers import StableVideoDiffusionPipeline
from PIL import Image

from .config import settings


_PIPELINE: StableVideoDiffusionPipeline | None = None
_DEVICE: str | None = None
_DTYPE = torch.float16 if settings.torch_dtype == "float16" else (
    torch.bfloat16 if settings.torch_dtype == "bfloat16" else torch.float32
)


def _load_pipeline() -> StableVideoDiffusionPipeline:
    global _PIPELINE, _DEVICE
    if _PIPELINE is not None:
        return _PIPELINE
    _DEVICE = settings.resolve_device()
    auth_token = settings.hf_token

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        settings.model_id,
        torch_dtype=_DTYPE,
        use_safetensors=True,
        variant="fp16" if _DTYPE == torch.float16 else None,
        token=auth_token,
    )

    pipe.enable_xformers_memory_efficient_attention() if hasattr(pipe, "enable_xformers_memory_efficient_attention") else None

    if _DEVICE == "cuda":
        pipe.enable_model_cpu_offload() if hasattr(pipe, "enable_model_cpu_offload") else pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")

    _PIPELINE = pipe
    return _PIPELINE


def _resize_long_edge(image: Image.Image, max_long_edge: int) -> Image.Image:
    w, h = image.size
    long_edge = max(w, h)
    if long_edge <= max_long_edge:
        return image
    scale = max_long_edge / long_edge
    new_size = (int(round(w * scale)), int(round(h * scale)))
    return image.resize(new_size, Image.BICUBIC)


def generate_video_frames(
    init_image: Image.Image,
    num_frames: int | None = None,
    num_inference_steps: int | None = None,
    motion_bucket_id: int | None = None,
    noise_aug_strength: float | None = None,
    fps: int | None = None,
) -> List[Image.Image]:
    pipe = _load_pipeline()

    num_frames = num_frames or settings.default_num_frames
    num_inference_steps = num_inference_steps or settings.default_num_inference_steps
    motion_bucket_id = motion_bucket_id or settings.default_motion_bucket_id
    noise_aug_strength = noise_aug_strength or settings.default_noise_aug_strength
    fps = fps or settings.default_fps

    num_frames = min(num_frames, settings.max_frames)

    image = init_image.convert("RGB")
    image = _resize_long_edge(image, settings.max_long_edge)

    # SVD expects numpy array BxCxHxW in [0,1] floats
    image_np = np.array(image).astype(np.float32) / 255.0

    result = pipe(
        image=image_np,
        decode_chunk_size=8,
        output_type="pil",
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        motion_bucket_id=motion_bucket_id,
        noise_aug_strength=noise_aug_strength,
        fps=fps,
    )

    frames: List[Image.Image] = result.frames[0]
    return frames
