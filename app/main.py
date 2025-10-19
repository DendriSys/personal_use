from __future__ import annotations

import io
import os
import tempfile
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi import Body
from PIL import Image

from .config import settings
from .inference import generate_video_frames
from .video_utils import frames_to_mp4
from .s3_utils import (
    download_s3_to_tempfile,
    upload_file_to_s3,
    derive_output_key,
)
from .moderation import analyze_with_rekognition

app = FastAPI(title="Image-to-Video Service", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ping")
def ping():
    # SageMaker expects 200 OK from /ping
    return {"status": "ok"}


@app.post("/generate")
async def generate(
    image_file: Optional[UploadFile] = File(default=None),
    image_s3_uri: Optional[str] = Form(default=None),
    num_frames: Optional[int] = Form(default=None),
    num_inference_steps: Optional[int] = Form(default=None),
    motion_bucket_id: Optional[int] = Form(default=None),
    noise_aug_strength: Optional[float] = Form(default=None),
    fps: Optional[int] = Form(default=None),
    consent: Optional[bool] = Form(default=None),
    s3_output_bucket: Optional[str] = Form(default=None),
    s3_output_prefix: Optional[str] = Form(default=None),
):
    if settings.require_consent and not consent:
        raise HTTPException(status_code=400, detail="Consent required to proceed")

    # Input image source
    tmp_input_path = None
    try:
        if image_file is not None:
            contents = await image_file.read()
            if settings.enable_rekognition:
                mod = analyze_with_rekognition(contents, settings.moderation_min_confidence)
                if mod.flagged and settings.reject_on_moderation:
                    raise HTTPException(status_code=400, detail=f"Moderation failed: {','.join(mod.reasons)}")
            tmp_input_path = tempfile.mkstemp()[1]
            with open(tmp_input_path, "wb") as f:
                f.write(contents)
        elif image_s3_uri is not None:
            tmp_input_path = download_s3_to_tempfile(image_s3_uri, settings.aws_region)
        else:
            raise HTTPException(status_code=400, detail="Provide either image_file or image_s3_uri")

        with Image.open(tmp_input_path) as img:
            frames = generate_video_frames(
                init_image=img,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                motion_bucket_id=motion_bucket_id,
                noise_aug_strength=noise_aug_strength,
                fps=fps,
            )

        # Encode to mp4
        fd, tmp_out_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        frames_to_mp4(frames, tmp_out_path, fps=fps or settings.default_fps)

        # Upload to S3 if configured
        bucket = s3_output_bucket or settings.output_bucket
        prefix = s3_output_prefix or settings.output_prefix
        if bucket:
            run_id, filename = derive_output_key(prefix, extension=".mp4")
            s3_uri = upload_file_to_s3(tmp_out_path, bucket, prefix, filename, region_name=settings.aws_region)
            return JSONResponse({
                "run_id": run_id,
                "s3_uri": s3_uri,
                "num_frames": len(frames),
                "fps": fps or settings.default_fps,
            })

        # Fallback: return temp path (not ideal for prod)
        return JSONResponse({
            "local_path": tmp_out_path,
            "num_frames": len(frames),
            "fps": fps or settings.default_fps,
        })
    finally:
        if tmp_input_path and os.path.exists(tmp_input_path):
            try:
                os.remove(tmp_input_path)
            except Exception:
                pass


@app.post("/invocations")
async def invocations(
    payload: dict = Body(...),
):
    """SageMaker-compatible inference endpoint.

    Expected JSON payload keys:
    - image_s3_uri (preferred) or raw multipart not supported here
    - consent: bool
    - num_frames, num_inference_steps, motion_bucket_id, noise_aug_strength, fps
    - s3_output_bucket, s3_output_prefix (optional overrides)
    """
    consent = payload.get("consent")
    if settings.require_consent and not consent:
        raise HTTPException(status_code=400, detail="Consent required to proceed")

    image_s3_uri = payload.get("image_s3_uri")
    if not image_s3_uri:
        raise HTTPException(status_code=400, detail="image_s3_uri is required for /invocations")

    num_frames = payload.get("num_frames")
    num_inference_steps = payload.get("num_inference_steps")
    motion_bucket_id = payload.get("motion_bucket_id")
    noise_aug_strength = payload.get("noise_aug_strength")
    fps = payload.get("fps")
    s3_output_bucket = payload.get("s3_output_bucket") or settings.output_bucket
    s3_output_prefix = payload.get("s3_output_prefix") or settings.output_prefix

    tmp_input_path = None
    try:
        tmp_input_path = download_s3_to_tempfile(image_s3_uri, settings.aws_region)
        with Image.open(tmp_input_path) as img:
            frames = generate_video_frames(
                init_image=img,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                motion_bucket_id=motion_bucket_id,
                noise_aug_strength=noise_aug_strength,
                fps=fps,
            )

        fd, tmp_out_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        frames_to_mp4(frames, tmp_out_path, fps=fps or settings.default_fps)

        if not s3_output_bucket:
            # In SageMaker we should always upload to S3
            raise HTTPException(status_code=500, detail="Output S3 bucket not configured")

        run_id, filename = derive_output_key(s3_output_prefix, extension=".mp4")
        s3_uri = upload_file_to_s3(
            tmp_out_path,
            s3_output_bucket,
            s3_output_prefix,
            filename,
            region_name=settings.aws_region,
        )
        return JSONResponse({
            "run_id": run_id,
            "s3_uri": s3_uri,
            "num_frames": len(frames),
            "fps": fps or settings.default_fps,
        })
    finally:
        if tmp_input_path and os.path.exists(tmp_input_path):
            try:
                os.remove(tmp_input_path)
            except Exception:
                pass
