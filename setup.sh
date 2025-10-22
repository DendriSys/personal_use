#!/bin/bash

# Setup script for Multi-Face LoRA Video Swapper

echo "=========================================="
echo "Multi-Face LoRA Video Swapper - Setup"
echo "=========================================="
echo ""

# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name --format=csv,noheader
else
    echo "⚠ Warning: No NVIDIA GPU detected. CPU mode will be used (slower)."
fi

echo ""
echo "Installing Python dependencies..."

# Install PyTorch (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt

echo ""
echo "Creating directory structure..."

# Create directories
mkdir -p models/lora
mkdir -p models/swap
mkdir -p data/faces
mkdir -p output
mkdir -p .cache

echo ""
echo "Downloading required models..."

# Download InSwapper model
echo "Downloading InSwapper face swap model..."
if [ ! -f "models/inswapper_128.onnx" ]; then
    wget -O models/inswapper_128.onnx \
        "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx"
    echo "✓ InSwapper model downloaded"
else
    echo "✓ InSwapper model already exists"
fi

# Download GFPGAN model
echo "Downloading GFPGAN face enhancement model..."
if [ ! -f "models/GFPGANv1.4.pth" ]; then
    wget -O models/GFPGANv1.4.pth \
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
    echo "✓ GFPGAN model downloaded"
else
    echo "✓ GFPGAN model already exists"
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Prepare your face datasets in data/faces/"
echo "2. Update config.yaml with your face configurations"
echo "3. Train LoRA models: python train_lora.py"
echo "4. Run face swapping: python main.py --input video.mp4 --output output.mp4"
echo ""
echo "See README.md for detailed instructions"
echo ""
