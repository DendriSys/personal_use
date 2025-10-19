from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import os


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ITV_", env_file=".env", extra="ignore")

    # Core model configuration
    model_id: str = Field(
        default="stabilityai/stable-video-diffusion-img2vid-xt-1-1",
        description="Hugging Face model repo for Stable Video Diffusion image-to-video",
    )
    hf_token: str | None = Field(default=None, description="Optional HF token for gated models")

    # Generation defaults
    default_num_frames: int = Field(default=25)
    default_num_inference_steps: int = Field(default=25)
    default_motion_bucket_id: int = Field(default=127)
    default_noise_aug_strength: float = Field(default=0.02)
    default_fps: int = Field(default=24)
    max_frames: int = Field(default=61)
    max_long_edge: int = Field(default=1024, description="Resize longest image edge to this many pixels")

    # Compute
    torch_dtype: str = Field(default="float16", description="float16 | bfloat16 | float32")
    device_preference: str = Field(default="cuda", description="cuda | cpu | auto")

    # S3 I/O
    input_bucket: str | None = Field(default=None)
    output_bucket: str | None = Field(default=None)
    output_prefix: str = Field(default="outputs/")
    aws_region: str | None = Field(default=os.getenv("AWS_REGION"))

    # Moderation and consent
    require_consent: bool = Field(default=True)
    enable_rekognition: bool = Field(default=False)
    reject_on_moderation: bool = Field(default=False)
    moderation_min_confidence: int = Field(default=80)

    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8080)

    def resolve_device(self) -> str:
        if self.device_preference == "auto":
            try:
                import torch  # noqa: WPS433

                return "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                return "cpu"
        return self.device_preference


settings = Settings()