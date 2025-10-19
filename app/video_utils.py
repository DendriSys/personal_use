from __future__ import annotations

from typing import List

import imageio
import numpy as np
from PIL import Image


def frames_to_mp4(frames: List[Image.Image], output_path: str, fps: int = 24) -> None:
    # Ensure frames are numpy arrays in RGB
    encoded_frames = []
    for frame in frames:
        if not isinstance(frame, Image.Image):
            frame = Image.fromarray(frame)
        if frame.mode != "RGB":
            frame = frame.convert("RGB")
        encoded_frames.append(np.array(frame))

    writer = imageio.get_writer(
        output_path,
        format="ffmpeg",
        mode="I",
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
        pixelformat="yuv420p",
    )
    try:
        for f in encoded_frames:
            writer.append_data(f)
    finally:
        writer.close()
