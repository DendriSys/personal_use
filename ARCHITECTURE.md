# System Architecture

## Overview

The Multi-Face LoRA Video Swapper is built with a modular architecture that separates concerns into distinct components:

```
┌─────────────────────────────────────────────────────────────┐
│                     Main Application                         │
│                      (main.py)                               │
└──────────────────┬──────────────────────────────────────────┘
                   │
        ┌──────────┴──────────┬──────────────┬────────────┐
        │                     │              │            │
┌───────▼────────┐  ┌─────────▼────────┐  ┌─▼──────────┐ │
│ Face Detector  │  │  Face Swapper    │  │   Config   │ │
│ (detector.py)  │  │  (swapper.py)    │  │ (config.yaml)│
└───────┬────────┘  └─────────┬────────┘  └────────────┘ │
        │                     │                           │
        │           ┌─────────┴──────────┐                │
        │           │                    │                │
┌───────▼───────────▼─────┐   ┌──────────▼────────┐      │
│   InsightFace Models    │   │  InSwapper Model  │      │
│ - RetinaFace (detect)   │   │  - Face Swapping  │      │
│ - ArcFace (recognize)   │   │  - ONNX Runtime   │      │
└─────────────────────────┘   └───────────────────┘      │
                                                          │
                              ┌───────────────────────────▼──┐
                              │  LoRA Training Pipeline      │
                              │  (train_lora.py)            │
                              │  - Stable Diffusion Base     │
                              │  - PEFT/LoRA                │
                              │  - Custom Face Datasets      │
                              └──────────────────────────────┘
```

## Component Breakdown

### 1. Face Detector (`face_detector.py`)

**Purpose**: Detect and recognize faces in images/video frames

**Key Features**:
- Face detection using InsightFace/RetinaFace
- Face embedding extraction using ArcFace
- Face database management (register, identify)
- Support for multiple identities

**Models Used**:
- RetinaFace: Face detection
- ArcFace: Face recognition/embedding
- ONNX Runtime: Inference optimization

**Main Classes**:
```python
class FaceDetector:
    - detect_faces(image) -> List[Dict]
    - extract_face_embedding(image, bbox) -> np.ndarray
    - register_face(name, image_path) -> bool
    - identify_face(embedding) -> Tuple[str, float]
    - save_database(path) / load_database(path)
```

### 2. Face Swapper (`face_swapper.py`)

**Purpose**: Perform face swapping on images and videos

**Key Features**:
- Single and multi-face swapping
- Face enhancement with GFPGAN
- Video processing with audio preservation
- Batch processing for efficiency

**Models Used**:
- InSwapper: Face swap ONNX model
- GFPGAN: Face enhancement
- FFmpeg: Video encoding/decoding

**Main Classes**:
```python
class FaceSwapper:
    - swap_face(source_embedding, target_image, target_face) -> np.ndarray
    - process_frame(frame, face_mappings) -> Tuple[np.ndarray, int]
    - swap_video(input_video, output_video, face_mappings)
    - swap_image(input_image, output_image, face_mappings)
```

### 3. LoRA Training (`train_lora.py`)

**Purpose**: Train personalized face models using LoRA

**Key Features**:
- Fine-tune Stable Diffusion on face datasets
- LoRA (Low-Rank Adaptation) for efficient training
- Support for multiple face identities
- Checkpointing and validation

**Models Used**:
- Stable Diffusion (base model)
- PEFT library (LoRA implementation)
- CLIP (text encoding)

**Main Functions**:
```python
class FaceDataset:
    - Custom dataset for face images
    - Caption generation with trigger words

def train_lora_model():
    - Setup base model and LoRA config
    - Training loop with gradient accumulation
    - Checkpoint saving
```

### 4. Dataset Preparation (`prepare_dataset.py`)

**Purpose**: Prepare face datasets for training

**Key Features**:
- Extract faces from raw images
- Organize datasets for training
- Verify dataset quality

**Main Functions**:
```python
- extract_faces_from_images(): Crop and align faces
- organize_dataset_for_training(): Structure for LoRA
- verify_dataset(): Quality checks
```

### 5. Configuration (`config.yaml`)

**Purpose**: Centralized configuration management

**Key Sections**:
```yaml
faces:           # Face identity definitions
training:        # LoRA training parameters
detection:       # Face detection settings
swapping:        # Face swap settings
video:          # Video processing settings
paths:          # Directory paths
```

## Data Flow

### Training Flow

```
Raw Images
    ↓
[Face Extraction] (prepare_dataset.py)
    ↓
Face Dataset
    ↓
[LoRA Training] (train_lora.py)
    ↓
Trained LoRA Models
    ↓
[Face Database Registration] (face_detector.py)
    ↓
Face Embeddings Database
```

### Inference Flow (Video)

```
Input Video
    ↓
[Frame Extraction] (cv2.VideoCapture)
    ↓
For Each Frame:
    ↓
[Face Detection] (FaceDetector.detect_faces)
    ↓
[Face Recognition] (FaceDetector.identify_face)
    ↓
[Face Swapping] (FaceSwapper.swap_face)
    ↓
[Face Enhancement] (GFPGAN)
    ↓
Processed Frame
    ↓
[Video Encoding] (FFmpeg)
    ↓
Output Video
```

## Model Requirements

### Required Models

1. **InSwapper** (`inswapper_128.onnx`)
   - Size: ~500MB
   - Purpose: Face swapping
   - Source: https://github.com/facefusion/facefusion-assets

2. **GFPGAN** (`GFPGANv1.4.pth`)
   - Size: ~350MB
   - Purpose: Face enhancement
   - Source: https://github.com/TencentARC/GFPGAN

3. **InsightFace Models** (Auto-downloaded)
   - buffalo_l model pack
   - Includes RetinaFace and ArcFace

### Optional Models

4. **Stable Diffusion Base** (For LoRA training)
   - Size: 4-7GB
   - Options:
     - stabilityai/stable-diffusion-xl-base-1.0
     - runwayml/stable-diffusion-v1-5

## Processing Pipeline Details

### Face Detection Pipeline

1. **Input**: RGB/BGR image
2. **Detection**: RetinaFace finds all faces
3. **Alignment**: 5-point landmark detection
4. **Embedding**: ArcFace extracts 512-dim embedding
5. **Recognition**: Cosine similarity matching
6. **Output**: Face bounding boxes, landmarks, embeddings, identities

### Face Swapping Pipeline

1. **Input**: Source embedding + Target image + Target face
2. **Alignment**: Align target face to canonical pose
3. **Swap**: InSwapper performs face replacement
4. **Blending**: Seamless/Poisson blending
5. **Enhancement**: GFPGAN improves quality
6. **Output**: Image with swapped face

### Video Processing Pipeline

1. **Extract**: Decode video to frames
2. **Process**: Apply face swapping to each frame
3. **Enhance**: Improve face quality
4. **Audio**: Extract audio from original
5. **Encode**: Combine frames + audio to video
6. **Cleanup**: Remove temporary files

## Performance Optimization

### GPU Acceleration

- CUDA support for all models
- Mixed precision training (FP16)
- Gradient checkpointing for memory efficiency
- Batch processing for video frames

### Memory Management

- Lazy model loading
- Frame batching
- Gradient accumulation
- 8-bit Adam optimizer (optional)

### Parallel Processing

- Multi-threaded frame processing
- Concurrent face detection/swapping
- GPU asynchronous operations

## Error Handling

- Graceful degradation (CPU fallback)
- Missing model detection
- Invalid input handling
- Memory overflow protection
- Progress tracking and recovery

## Security Considerations

- No network requests during inference
- Local model storage
- No data collection
- User consent required for face data
- Watermarking support (future)

## Future Enhancements

1. **Real-time Processing**: Webcam support
2. **Multi-GPU**: Distributed processing
3. **Better Models**: Latest face swap models
4. **UI Interface**: Web-based GUI
5. **Mobile Support**: ONNX/TFLite export
6. **Cloud Deployment**: API service
