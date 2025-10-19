# CUDA base with Python
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Prevent pip cache bloat
ENV PIP_NO_CACHE_DIR=1

# Copy and install deps first
COPY requirements.txt ./
RUN python3 -m pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install xformers==0.0.28.post3 --extra-index-url https://download.pytorch.org/whl/cu121 && \
    pip install torch==2.4.1 torchvision==0.19.1 --extra-index-url https://download.pytorch.org/whl/cu121

# Copy app
COPY app ./app

ENV PYTHONPATH=/app

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
