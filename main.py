"""
Main script for multi-face video swapping with LoRA-trained face models
"""

import argparse
import yaml
from pathlib import Path
from typing import Dict
import numpy as np

from face_detector import FaceDetector, build_face_database_from_config
from face_swapper import FaceSwapper


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_face_mappings(config: dict, detector: FaceDetector) -> Dict[str, np.ndarray]:
    """
    Setup face mappings for swapping
    Maps face names to their source embeddings
    
    Args:
        config: Configuration dict
        detector: FaceDetector instance
        
    Returns:
        Dictionary mapping face names to embeddings
    """
    face_mappings = {}
    
    # Use the registered embeddings from detector's database
    for name, embedding in detector.face_database.items():
        face_mappings[name] = embedding
    
    return face_mappings


def main():
    parser = argparse.ArgumentParser(
        description="Multi-face video swapping with LoRA-trained models"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input video or image path'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output video or image path'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['video', 'image'],
        default='video',
        help='Processing mode'
    )
    parser.add_argument(
        '--face-db',
        type=str,
        help='Path to saved face database (optional)'
    )
    parser.add_argument(
        '--save-face-db',
        type=str,
        help='Save face database to this path (optional)'
    )
    parser.add_argument(
        '--swap-mapping',
        type=str,
        nargs='+',
        help='Face swap mappings in format "source_name:target_name" (optional, uses all registered faces if not specified)'
    )
    parser.add_argument(
        '--no-enhance',
        action='store_true',
        help='Disable face enhancement'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for processing'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for video processing'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Initialize face detector
    print("\nInitializing face detector...")
    detector = FaceDetector(
        detection_model=config['detection']['detection_model'],
        recognition_model=config['detection']['recognition_model'],
        detection_threshold=config['detection']['detection_threshold'],
        similarity_threshold=config['detection']['similarity_threshold'],
        device=args.device,
    )
    
    # Load or build face database
    if args.face_db and Path(args.face_db).exists():
        print(f"Loading face database from {args.face_db}...")
        detector.load_database(args.face_db)
    else:
        print("Building face database from config...")
        detector = build_face_database_from_config(config, detector)
        
        # Save database if requested
        if args.save_face_db:
            detector.save_database(args.save_face_db)
    
    # Setup face mappings
    face_mappings = setup_face_mappings(config, detector)
    
    # Handle custom swap mappings
    if args.swap_mapping:
        custom_mappings = {}
        for mapping in args.swap_mapping:
            if ':' in mapping:
                source, target = mapping.split(':')
                if source in face_mappings:
                    custom_mappings[target] = face_mappings[source]
        face_mappings = custom_mappings
    
    print(f"\nFace mappings: {list(face_mappings.keys())}")
    
    # Initialize face swapper
    print("\nInitializing face swapper...")
    swapper = FaceSwapper(
        detector=detector,
        enhancer_model=config['swapping']['face_enhancer'] if not args.no_enhance else None,
        enhancement_strength=config['swapping']['enhancement_strength'],
        device=args.device,
    )
    
    # Process input
    if args.mode == 'video':
        swapper.swap_video(
            input_video=args.input,
            output_video=args.output,
            face_mappings=face_mappings,
            batch_size=args.batch_size,
            enhance=not args.no_enhance,
            keep_temp=config['video']['keep_temp_frames'],
            temp_dir=config['video']['temp_frame_dir'],
        )
    else:  # image mode
        swapper.swap_image(
            input_image=args.input,
            output_image=args.output,
            face_mappings=face_mappings,
            enhance=not args.no_enhance,
        )
    
    print("\nDone!")


if __name__ == '__main__':
    main()
