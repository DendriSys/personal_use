# Multi-Face LoRA Video Swapper

A comprehensive face-swapping system that supports training personalized LoRA face models and swapping multiple faces in videos simultaneously. Perfect for creating deepfakes, face replacements, and AI-generated content.

## 🌟 Features

- **LoRA Fine-tuning**: Train personalized face models using LoRA (Low-Rank Adaptation) on Stable Diffusion
- **Multi-Face Recognition**: Automatically detect and identify multiple faces in videos
- **Simultaneous Face Swapping**: Swap multiple different faces in a single video
- **Face Enhancement**: Built-in GFPGAN support for high-quality face restoration
- **Keyword/Name Mapping**: Associate faces with names or keywords for easy management
- **Batch Processing**: Efficient video processing with GPU acceleration
- **High Quality**: Professional-grade face swapping with seamless blending

## 📋 Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support (recommended, CPU mode available but slower)
- 16GB+ RAM (32GB recommended for video processing)
- 10GB+ free disk space

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd multi-face-lora-swapper

# Run setup script
chmod +x setup.sh
./setup.sh

# Or manual installation:
pip install -r requirements.txt
```

### 2. Prepare Face Datasets

Organize your reference face images:

```bash
# Extract faces from raw images
python prepare_dataset.py --mode extract \
    --input raw_images/john/ \
    --output extracted_faces/john/

# Organize for training
python prepare_dataset.py --mode organize \
    --input extracted_faces/john/ \
    --name john \
    --output data/faces

# Verify dataset
python prepare_dataset.py --mode verify \
    --input data/faces/john/
```

Your dataset structure should look like:
```
data/faces/
├── john/
│   ├── john_000.jpg
│   ├── john_001.jpg
│   └── ...
├── jane/
│   ├── jane_000.jpg
│   ├── jane_001.jpg
│   └── ...
```

### 3. Configure Faces

Edit `config.yaml` to define your faces:

```yaml
faces:
  person1:
    name: "John"
    lora_path: "models/lora/john_face.safetensors"
    reference_images: "data/faces/john/"
    trigger_word: "j0hn"
  
  person2:
    name: "Jane"
    lora_path: "models/lora/jane_face.safetensors"
    reference_images: "data/faces/jane/"
    trigger_word: "j4ne"
```

### 4. Train LoRA Models (Optional)

Train personalized face LoRAs for higher quality:

```bash
# Train all faces in config
python train_lora.py --config config.yaml

# Train specific face
python train_lora.py --config config.yaml --face person1
```

Training parameters can be adjusted in `config.yaml` under the `training` section.

### 5. Swap Faces in Video

```bash
# Basic usage - swap all configured faces
python main.py \
    --input input_video.mp4 \
    --output output_video.mp4 \
    --mode video

# Swap specific faces only
python main.py \
    --input input_video.mp4 \
    --output output_video.mp4 \
    --mode video \
    --swap-mapping John Jane

# Process single image
python main.py \
    --input photo.jpg \
    --output result.jpg \
    --mode image
```

## 📖 Detailed Usage

### Training LoRA Models

LoRA training creates personalized face models that can be used with Stable Diffusion for higher quality face generation.

**Training Configuration** (`config.yaml`):

```yaml
training:
  base_model: "stabilityai/stable-diffusion-xl-base-1.0"
  resolution: 512
  train_batch_size: 1
  max_train_steps: 1000
  learning_rate: 1e-4
  rank: 32  # LoRA rank
  lora_alpha: 32
```

**Tips for Better Results**:
- Use 15-30 high-quality reference images per person
- Include variety: different angles, expressions, lighting
- Minimum 512x512 resolution recommended
- Clear, frontal face photos work best
- Avoid heavy makeup, glasses, or occlusions

### Face Detection & Recognition

The system uses InsightFace for robust face detection and recognition:

**Configuration** (`config.yaml`):

```yaml
detection:
  detection_model: "retinaface"
  detection_threshold: 0.5
  recognition_model: "arcface"
  similarity_threshold: 0.6  # Adjust for stricter/looser matching
  min_face_size: 80
```

**Adjusting Similarity Threshold**:
- Higher (0.7-0.9): Stricter matching, fewer false positives
- Lower (0.4-0.6): More lenient matching, may swap similar faces

### Face Swapping Options

**Enhancement**:

```bash
# With face enhancement (default)
python main.py --input video.mp4 --output output.mp4

# Without enhancement (faster)
python main.py --input video.mp4 --output output.mp4 --no-enhance
```

**Custom Face Mappings**:

Swap detected faces with different source faces:

```bash
# Swap John's face with Bob's appearance
python main.py \
    --input video.mp4 \
    --output output.mp4 \
    --swap-mapping John:Bob Jane:Alice
```

**Using Pre-built Face Database**:

```bash
# Save face database for reuse
python main.py \
    --input video.mp4 \
    --output output.mp4 \
    --save-face-db face_db.pkl

# Load saved database (faster startup)
python main.py \
    --input video.mp4 \
    --output output.mp4 \
    --face-db face_db.pkl
```

### Video Processing Settings

Configure video processing in `config.yaml`:

```yaml
video:
  output_format: "mp4"
  output_fps: null  # null = use input fps
  output_quality: 23  # CRF value (lower = better, 18-28 recommended)
  batch_size: 8  # Frames to process in parallel
  temp_frame_dir: "temp_frames/"
  keep_temp_frames: false
```

**Performance Tips**:
- Increase `batch_size` if you have more GPU memory
- Lower `output_quality` (higher CRF) for smaller file sizes
- Use `--no-enhance` for 2-3x faster processing

## 🏗️ Architecture

### Project Structure

```
.
├── config.yaml              # Main configuration file
├── requirements.txt         # Python dependencies
├── setup.sh                 # Setup script
│
├── train_lora.py           # LoRA training script
├── face_detector.py        # Face detection & recognition module
├── face_swapper.py         # Face swapping engine
├── main.py                 # Main application
├── prepare_dataset.py      # Dataset preparation utilities
│
├── models/
│   ├── inswapper_128.onnx  # Face swap model
│   ├── GFPGANv1.4.pth     # Face enhancement model
│   └── lora/               # Trained LoRA models
│
├── data/
│   └── faces/              # Reference face datasets
│       ├── person1/
│       └── person2/
│
└── output/                 # Processed videos/images
```

### Processing Pipeline

1. **Face Detection**: RetinaFace detects all faces in each frame
2. **Face Recognition**: ArcFace compares detected faces to database
3. **Face Matching**: Identifies which faces to swap based on similarity
4. **Face Swapping**: InSwapper performs the face replacement
5. **Enhancement**: GFPGAN enhances swapped faces for quality
6. **Blending**: Seamless blending into original frame
7. **Video Encoding**: FFmpeg combines frames with audio

## 🎯 Use Cases

- **Content Creation**: Create personalized video content
- **Film Production**: Face replacement in movies/series
- **Research**: Face recognition and synthesis studies
- **Entertainment**: Fun face swaps with friends/family
- **Privacy**: Anonymize faces in videos

## ⚙️ Advanced Configuration

### Custom Swap Models

You can use different face swap models:

```yaml
swapping:
  swap_model: "inswapper"  # or "simswap", "faceswap"
  blend_method: "seamless"  # or "poisson", "gaussian"
  face_enhancer: "gfpgan"   # or "codeformer", "none"
  enhancement_strength: 0.5  # 0.0-1.0
  color_correction: true
```

### Multi-GPU Support

For multi-GPU setups, use environment variables:

```bash
CUDA_VISIBLE_DEVICES=0,1 python main.py --input video.mp4 --output output.mp4
```

### Memory Optimization

For large videos or limited GPU memory:

```yaml
training:
  gradient_checkpointing: true
  use_8bit_adam: true
  mixed_precision: "fp16"
```

## 🐛 Troubleshooting

### Common Issues

**1. Out of Memory Errors**

```bash
# Reduce batch size
python main.py --batch-size 4 ...

# Or disable enhancement
python main.py --no-enhance ...
```

**2. No Faces Detected**

- Check `detection_threshold` in config (try lowering to 0.3)
- Ensure faces are visible and not too small
- Verify input video quality

**3. Wrong Faces Swapped**

- Adjust `similarity_threshold` in config (try increasing to 0.7)
- Add more reference images for better recognition
- Use face database inspection

**4. Poor Quality Results**

- Enable face enhancement: remove `--no-enhance`
- Increase `enhancement_strength` in config
- Use higher quality input videos
- Train LoRA models for better face generation

**5. Slow Processing**

- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Increase batch size if GPU memory allows
- Disable enhancement with `--no-enhance`
- Use CPU mode only as last resort

### Model Downloads

If models fail to download automatically:

**InSwapper**: https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx
**GFPGAN**: https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth

Place in `models/` directory.

## 📊 Performance Benchmarks

Approximate processing times (RTX 3090, 1080p video):

| Configuration | Speed | Quality |
|--------------|-------|---------|
| No enhancement | 2-3x realtime | Good |
| With enhancement | 1-1.5x realtime | Excellent |
| + LoRA faces | 1x realtime | Outstanding |

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## ⚠️ Ethical Considerations

This tool is powerful and should be used responsibly:

- **Consent**: Always obtain consent before using someone's likeness
- **Disclosure**: Clearly label AI-generated/modified content
- **No Harm**: Do not create harmful, deceptive, or illegal content
- **Respect Privacy**: Respect individuals' rights to their image
- **Legal Compliance**: Follow all applicable laws and regulations

## 📄 License

This project is for educational and research purposes. Please review the licenses of all dependencies:

- InsightFace: https://github.com/deepinsight/insightface
- GFPGAN: https://github.com/TencentARC/GFPGAN
- Diffusers: https://github.com/huggingface/diffusers

## 🙏 Acknowledgments

Built with:
- [InsightFace](https://github.com/deepinsight/insightface) - Face detection and recognition
- [GFPGAN](https://github.com/TencentARC/GFPGAN) - Face enhancement
- [Diffusers](https://github.com/huggingface/diffusers) - Stable Diffusion and LoRA
- [InSwapper](https://github.com/haofanwang/inswapper) - Face swapping model

## 📞 Support

For issues, questions, or feature requests:
- Open an issue on GitHub
- Check existing issues for solutions
- Review documentation carefully

---

**Note**: This is a powerful AI tool. Use responsibly and ethically. Always respect privacy, obtain proper consent, and follow applicable laws.
