#!/bin/bash

# Example usage scripts for multi-face video swapper

echo "Multi-Face Video Swapper - Example Commands"
echo "==========================================="
echo ""

# Example 1: Basic video face swap
echo "Example 1: Basic video face swap"
echo "python main.py --input examples/video.mp4 --output output/result.mp4"
echo ""

# Example 2: Image face swap
echo "Example 2: Single image face swap"
echo "python main.py --input examples/photo.jpg --output output/result.jpg --mode image"
echo ""

# Example 3: Fast processing without enhancement
echo "Example 3: Fast processing (no enhancement)"
echo "python main.py --input video.mp4 --output output.mp4 --no-enhance"
echo ""

# Example 4: Specific face mappings
echo "Example 4: Swap specific faces only"
echo "python main.py --input video.mp4 --output output.mp4 --swap-mapping John Jane"
echo ""

# Example 5: Using saved face database
echo "Example 5: Save and reuse face database"
echo "python main.py --input video.mp4 --output output1.mp4 --save-face-db face_db.pkl"
echo "python main.py --input video2.mp4 --output output2.mp4 --face-db face_db.pkl"
echo ""

# Example 6: CPU mode
echo "Example 6: CPU mode (slower, no GPU required)"
echo "python main.py --input video.mp4 --output output.mp4 --device cpu"
echo ""

# Example 7: Low memory mode
echo "Example 7: Low memory mode"
echo "python main.py --input video.mp4 --output output.mp4 --batch-size 2 --no-enhance"
echo ""

# Dataset preparation examples
echo ""
echo "Dataset Preparation Examples:"
echo "==========================================="
echo ""

echo "Extract faces from images:"
echo "python prepare_dataset.py --mode extract --input raw_photos/ --output extracted_faces/"
echo ""

echo "Organize dataset for training:"
echo "python prepare_dataset.py --mode organize --input extracted_faces/ --name john --output data/faces"
echo ""

echo "Verify dataset is ready:"
echo "python prepare_dataset.py --mode verify --input data/faces/john/"
echo ""

# LoRA training examples
echo ""
echo "LoRA Training Examples:"
echo "==========================================="
echo ""

echo "Train all faces in config:"
echo "python train_lora.py --config config.yaml"
echo ""

echo "Train specific person:"
echo "python train_lora.py --config config.yaml --face person1"
echo ""

# Advanced examples
echo ""
echo "Advanced Examples:"
echo "==========================================="
echo ""

echo "Custom swap mapping (swap John with Bob's appearance):"
echo "python main.py --input video.mp4 --output output.mp4 --swap-mapping John:Bob"
echo ""

echo "Multi-GPU processing:"
echo "CUDA_VISIBLE_DEVICES=0,1 python main.py --input video.mp4 --output output.mp4"
echo ""

echo "Batch process multiple videos:"
cat << 'EOF'
for video in videos/*.mp4; do
    output="output/$(basename "$video")"
    python main.py --input "$video" --output "$output" --face-db face_db.pkl
done
EOF
echo ""

echo "==========================================="
echo "See README.md for more detailed information"
echo "==========================================="
