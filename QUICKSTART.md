# Quick Start Guide

Get up and running with face swapping in 5 minutes!

## Prerequisites

- Python 3.8+ installed
- NVIDIA GPU with CUDA (recommended)
- 10GB free disk space

## Step 1: Install Dependencies (2 minutes)

```bash
# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt
```

## Step 2: Download Models (3 minutes)

```bash
# Create models directory
mkdir -p models

# Download InSwapper (face swap model)
wget -O models/inswapper_128.onnx \
  "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx"

# Download GFPGAN (face enhancement)
wget -O models/GFPGANv1.4.pth \
  "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
```

Or just run:
```bash
chmod +x setup.sh
./setup.sh
```

## Step 3: Prepare Your Face Images (5 minutes)

1. Create a directory with 10-20 photos of the person whose face you want to use:

```bash
mkdir -p my_faces
# Copy 10-20 clear face photos to my_faces/
```

2. Extract and organize faces:

```bash
python prepare_dataset.py --mode extract --input my_faces/ --output faces_extracted/
python prepare_dataset.py --mode organize --input faces_extracted/ --name myself --output data/faces
```

## Step 4: Update Configuration (1 minute)

Edit `config.yaml`:

```yaml
faces:
  me:
    name: "MySelf"
    reference_images: "data/faces/myself/"
    trigger_word: "myself"
```

## Step 5: Run Face Swap! (Varies)

### Swap faces in a video:

```bash
python main.py --input input_video.mp4 --output output_video.mp4
```

### Swap faces in an image:

```bash
python main.py --input photo.jpg --output result.jpg --mode image
```

## Example Output

The system will:
1. ✅ Detect all faces in your input video/image
2. ✅ Recognize which faces match your reference images
3. ✅ Swap matched faces with your source face
4. ✅ Enhance face quality for seamless results
5. ✅ Save the output with original audio (for videos)

## Processing Time

On RTX 3090:
- 1 minute video (1080p) ≈ 30-60 seconds with enhancement
- 1 minute video (1080p) ≈ 15-30 seconds without enhancement

Use `--no-enhance` for faster processing:
```bash
python main.py --input video.mp4 --output output.mp4 --no-enhance
```

## Troubleshooting

### "CUDA out of memory"
```bash
python main.py --input video.mp4 --output output.mp4 --batch-size 4
```

### "No faces detected"
Lower the detection threshold in `config.yaml`:
```yaml
detection:
  detection_threshold: 0.3  # Default is 0.5
```

### "Wrong faces being swapped"
Increase similarity threshold in `config.yaml`:
```yaml
detection:
  similarity_threshold: 0.7  # Default is 0.6
```

## Next Steps

- **Better Quality**: Train LoRA models with `python train_lora.py`
- **Multiple Faces**: Add more people to `config.yaml`
- **Custom Mappings**: Use `--swap-mapping` to control which faces swap
- **Read Full Docs**: See `README.md` for advanced features

## Need Help?

- Check `README.md` for detailed documentation
- Review configuration options in `config.yaml`
- Test with a short video clip first

Happy face swapping! 🎭
