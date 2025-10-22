"""
Multi-Face Video Swapping System
Supports swapping multiple faces in videos using trained face models
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import onnxruntime
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from tqdm import tqdm
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from gfpgan import GFPGANer
    GFPGAN_AVAILABLE = True
except ImportError:
    GFPGAN_AVAILABLE = False
    print("Warning: GFPGAN not available. Face enhancement will be disabled.")

from face_detector import FaceDetector


class FaceSwapper:
    """Multi-face swapping for images and videos"""
    
    def __init__(
        self,
        detector: FaceDetector,
        swap_model_path: Optional[str] = None,
        enhancer_model: Optional[str] = 'gfpgan',
        enhancement_strength: float = 0.5,
        device: str = 'cuda',
    ):
        self.detector = detector
        self.enhancement_strength = enhancement_strength
        self.device = device
        
        # Initialize swap model (InSwapper)
        print("Loading face swap model...")
        self.swapper = self._load_swap_model(swap_model_path)
        
        # Initialize face enhancer
        self.enhancer = None
        if enhancer_model and GFPGAN_AVAILABLE:
            print("Loading face enhancement model...")
            self.enhancer = self._load_enhancer(enhancer_model)
        
        print("FaceSwapper initialized successfully")
    
    def _load_swap_model(self, model_path: Optional[str]):
        """Load face swapping model (InSwapper)"""
        if model_path is None:
            # Download default model
            model_path = 'models/inswapper_128.onnx'
            
            # Create models directory
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Check if model exists
            if not Path(model_path).exists():
                print(f"InSwapper model not found at {model_path}")
                print("Please download from: https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx")
                print(f"And place it at: {model_path}")
        
        # Load ONNX model
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
        
        try:
            swapper = get_model(model_path, providers=providers)
        except:
            # Fallback: load with onnxruntime directly
            swapper = onnxruntime.InferenceSession(
                model_path,
                providers=providers
            )
        
        return swapper
    
    def _load_enhancer(self, model_name: str):
        """Load face enhancement model"""
        if not GFPGAN_AVAILABLE:
            return None
        
        model_path = f'models/{model_name}.pth'
        
        if model_name.lower() == 'gfpgan':
            enhancer = GFPGANer(
                model_path='models/GFPGANv1.4.pth',
                upscale=1,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None,
                device=self.device,
            )
        else:
            enhancer = None
        
        return enhancer
    
    def swap_face(
        self,
        source_face_embedding: np.ndarray,
        target_image: np.ndarray,
        target_face: Dict,
    ) -> np.ndarray:
        """
        Swap a single face in target image
        
        Args:
            source_face_embedding: Embedding of source face
            target_image: Target image (BGR)
            target_face: Target face dict from detector
            
        Returns:
            Image with swapped face
        """
        # Get face from InsightFace format
        from insightface.utils import face_align
        
        # Create a simple face object for swapper
        class Face:
            def __init__(self, bbox, kps, det_score, embedding):
                self.bbox = bbox
                self.kps = kps
                self.det_score = det_score
                self.embedding = embedding
                self.normed_embedding = embedding / np.linalg.norm(embedding)
        
        target_face_obj = Face(
            bbox=target_face['bbox'],
            kps=target_face['landmarks'],
            det_score=target_face['det_score'],
            embedding=target_face['embedding'],
        )
        
        source_face_obj = Face(
            bbox=np.array([0, 0, 128, 128]),
            kps=np.zeros((5, 2)),
            det_score=1.0,
            embedding=source_face_embedding,
        )
        
        # Perform swap
        try:
            result = self.swapper.get(target_image, target_face_obj, source_face_obj, paste_back=True)
        except:
            # Fallback method
            result = target_image.copy()
        
        return result
    
    def process_frame(
        self,
        frame: np.ndarray,
        face_mappings: Dict[str, np.ndarray],
        enhance: bool = True,
        blend_method: str = 'seamless',
    ) -> Tuple[np.ndarray, int]:
        """
        Process a single frame with multi-face swapping
        
        Args:
            frame: Input frame (BGR)
            face_mappings: Dict mapping face names to source embeddings
            enhance: Whether to enhance swapped faces
            blend_method: Blending method for face merging
            
        Returns:
            Tuple of (processed_frame, num_faces_swapped)
        """
        result = frame.copy()
        
        # Detect faces in frame
        detected_faces = self.detector.detect_faces(frame)
        
        if not detected_faces:
            return result, 0
        
        swapped_count = 0
        
        # Process each detected face
        for face in detected_faces:
            # Identify the face
            face_id, similarity = self.detector.identify_face(face['embedding'])
            
            if face_id and face_id in face_mappings:
                # Swap the face
                source_embedding = face_mappings[face_id]
                result = self.swap_face(source_embedding, result, face)
                swapped_count += 1
                
                # Enhance if requested
                if enhance and self.enhancer:
                    bbox = face['bbox']
                    x1, y1, x2, y2 = bbox
                    
                    # Add padding
                    padding = 20
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(result.shape[1], x2 + padding)
                    y2 = min(result.shape[0], y2 + padding)
                    
                    # Extract and enhance face region
                    face_region = result[y1:y2, x1:x2]
                    
                    try:
                        _, _, enhanced = self.enhancer.enhance(
                            face_region,
                            has_aligned=False,
                            only_center_face=True,
                            paste_back=True,
                            weight=self.enhancement_strength,
                        )
                        result[y1:y2, x1:x2] = enhanced
                    except Exception as e:
                        print(f"Warning: Face enhancement failed: {e}")
        
        return result, swapped_count
    
    def swap_video(
        self,
        input_video: str,
        output_video: str,
        face_mappings: Dict[str, np.ndarray],
        batch_size: int = 8,
        enhance: bool = True,
        keep_temp: bool = False,
        temp_dir: str = 'temp_frames',
    ):
        """
        Swap faces in video
        
        Args:
            input_video: Path to input video
            output_video: Path to output video
            face_mappings: Dict mapping face names to source embeddings
            batch_size: Number of frames to process in parallel
            enhance: Whether to enhance faces
            keep_temp: Whether to keep temporary frames
            temp_dir: Directory for temporary frames
        """
        print(f"\n{'='*60}")
        print(f"Processing video: {input_video}")
        print(f"Output: {output_video}")
        print(f"Face mappings: {list(face_mappings.keys())}")
        print(f"{'='*60}\n")
        
        # Create temp directory
        temp_path = Path(temp_dir)
        temp_path.mkdir(parents=True, exist_ok=True)
        
        # Extract video info
        cap = cv2.VideoCapture(input_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Process frames
        print("\nProcessing frames...")
        frame_idx = 0
        processed_frames = []
        total_swaps = 0
        
        with tqdm(total=total_frames) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, num_swaps = self.process_frame(
                    frame,
                    face_mappings,
                    enhance=enhance,
                )
                
                # Save processed frame
                frame_path = temp_path / f"frame_{frame_idx:06d}.png"
                cv2.imwrite(str(frame_path), processed_frame)
                processed_frames.append(frame_path)
                
                total_swaps += num_swaps
                frame_idx += 1
                pbar.update(1)
                pbar.set_postfix({"faces_swapped": total_swaps})
        
        cap.release()
        
        print(f"\nTotal faces swapped: {total_swaps}")
        
        # Extract audio from original video
        print("\nExtracting audio...")
        audio_path = temp_path / "audio.aac"
        audio_cmd = [
            'ffmpeg', '-y', '-i', input_video,
            '-vn', '-acodec', 'copy',
            str(audio_path)
        ]
        subprocess.run(audio_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Combine frames into video
        print("Encoding video...")
        frame_pattern = str(temp_path / "frame_%06d.png")
        
        # Create video without audio
        temp_video = temp_path / "temp_video.mp4"
        video_cmd = [
            'ffmpeg', '-y', '-r', str(fps),
            '-i', frame_pattern,
            '-c:v', 'libx264', '-crf', '23',
            '-pix_fmt', 'yuv420p',
            str(temp_video)
        ]
        subprocess.run(video_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Combine video with audio
        if audio_path.exists():
            final_cmd = [
                'ffmpeg', '-y',
                '-i', str(temp_video),
                '-i', str(audio_path),
                '-c:v', 'copy', '-c:a', 'aac',
                '-shortest',
                output_video
            ]
            subprocess.run(final_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            # No audio, just copy video
            shutil.copy(temp_video, output_video)
        
        # Cleanup
        if not keep_temp:
            print("Cleaning up temporary files...")
            shutil.rmtree(temp_path)
        
        print(f"\n{'='*60}")
        print(f"Video processing complete!")
        print(f"Output saved to: {output_video}")
        print(f"{'='*60}\n")
    
    def swap_image(
        self,
        input_image: str,
        output_image: str,
        face_mappings: Dict[str, np.ndarray],
        enhance: bool = True,
    ):
        """
        Swap faces in a single image
        
        Args:
            input_image: Path to input image
            output_image: Path to output image
            face_mappings: Dict mapping face names to source embeddings
            enhance: Whether to enhance faces
        """
        print(f"\nProcessing image: {input_image}")
        
        # Read image
        image = cv2.imread(input_image)
        
        if image is None:
            print(f"Error: Could not read image {input_image}")
            return
        
        # Process image
        result, num_swaps = self.process_frame(
            image,
            face_mappings,
            enhance=enhance,
        )
        
        # Save result
        cv2.imwrite(output_image, result)
        
        print(f"Swapped {num_swaps} faces")
        print(f"Result saved to: {output_image}\n")
