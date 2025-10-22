"""
Dataset preparation utilities for LoRA training
Helps organize and prepare face images for training
"""

import argparse
import shutil
from pathlib import Path
import cv2
from face_detector import FaceDetector
from tqdm import tqdm


def extract_faces_from_images(
    input_dir: str,
    output_dir: str,
    min_size: int = 256,
    padding: float = 0.3,
):
    """
    Extract and crop faces from images
    
    Args:
        input_dir: Directory containing source images
        output_dir: Directory to save cropped faces
        min_size: Minimum face size to keep
        padding: Padding around face (fraction of face size)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize detector
    print("Initializing face detector...")
    detector = FaceDetector()
    
    # Process all images
    image_files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png')) + list(input_path.glob('*.jpeg'))
    
    print(f"\nProcessing {len(image_files)} images...")
    
    face_count = 0
    
    for img_path in tqdm(image_files):
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        # Detect faces
        faces = detector.detect_faces(image)
        
        # Extract each face
        for i, face in enumerate(faces):
            bbox = face['bbox']
            x1, y1, x2, y2 = bbox
            
            # Check face size
            face_width = x2 - x1
            face_height = y2 - y1
            
            if face_width < min_size or face_height < min_size:
                continue
            
            # Add padding
            pad_x = int(face_width * padding)
            pad_y = int(face_height * padding)
            
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(image.shape[1], x2 + pad_x)
            y2 = min(image.shape[0], y2 + pad_y)
            
            # Crop face
            face_crop = image[y1:y2, x1:x2]
            
            # Save face
            output_file = output_path / f"{img_path.stem}_face_{i}.jpg"
            cv2.imwrite(str(output_file), face_crop)
            face_count += 1
    
    print(f"\nExtracted {face_count} faces to {output_dir}")


def organize_dataset_for_training(
    face_dir: str,
    person_name: str,
    output_dir: str = "data/faces",
):
    """
    Organize extracted faces into training directory structure
    
    Args:
        face_dir: Directory containing face images
        person_name: Name of the person (used for directory name)
        output_dir: Base output directory
    """
    face_path = Path(face_dir)
    output_path = Path(output_dir) / person_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Copy all face images
    image_files = list(face_path.glob('*.jpg')) + list(face_path.glob('*.png'))
    
    print(f"Organizing {len(image_files)} images for {person_name}...")
    
    for i, img_file in enumerate(image_files):
        output_file = output_path / f"{person_name}_{i:03d}{img_file.suffix}"
        shutil.copy(img_file, output_file)
    
    print(f"Dataset organized at: {output_path}")


def verify_dataset(
    dataset_dir: str,
    min_images: int = 10,
):
    """
    Verify dataset is ready for training
    
    Args:
        dataset_dir: Directory containing training images
        min_images: Minimum number of images required
    """
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"Error: Dataset directory {dataset_dir} does not exist")
        return False
    
    image_files = list(dataset_path.glob('*.jpg')) + list(dataset_path.glob('*.png'))
    
    print(f"\nDataset verification for: {dataset_dir}")
    print(f"Number of images: {len(image_files)}")
    
    if len(image_files) < min_images:
        print(f"Warning: Only {len(image_files)} images found. At least {min_images} recommended.")
        return False
    
    # Check image dimensions
    sizes = []
    for img_file in image_files:
        img = cv2.imread(str(img_file))
        if img is not None:
            sizes.append((img.shape[1], img.shape[0]))
    
    if sizes:
        avg_width = sum(s[0] for s in sizes) / len(sizes)
        avg_height = sum(s[1] for s in sizes) / len(sizes)
        print(f"Average image size: {avg_width:.0f}x{avg_height:.0f}")
    
    print("Dataset is ready for training!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Prepare face dataset for LoRA training")
    parser.add_argument('--mode', choices=['extract', 'organize', 'verify'], required=True,
                      help='Operation mode')
    parser.add_argument('--input', type=str, help='Input directory')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--name', type=str, help='Person name (for organize mode)')
    parser.add_argument('--min-size', type=int, default=256, help='Minimum face size')
    parser.add_argument('--min-images', type=int, default=10, help='Minimum images for verification')
    
    args = parser.parse_args()
    
    if args.mode == 'extract':
        extract_faces_from_images(
            input_dir=args.input,
            output_dir=args.output,
            min_size=args.min_size,
        )
    elif args.mode == 'organize':
        organize_dataset_for_training(
            face_dir=args.input,
            person_name=args.name,
            output_dir=args.output or "data/faces",
        )
    elif args.mode == 'verify':
        verify_dataset(
            dataset_dir=args.input,
            min_images=args.min_images,
        )


if __name__ == '__main__':
    main()
