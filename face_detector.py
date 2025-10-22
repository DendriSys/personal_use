"""
Face Detection and Recognition Module
Handles face detection, embedding extraction, and face matching
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import pickle
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model


class FaceDetector:
    """Face detection and recognition using InsightFace"""
    
    def __init__(
        self,
        detection_model: str = 'retinaface',
        recognition_model: str = 'arcface',
        detection_threshold: float = 0.5,
        similarity_threshold: float = 0.6,
        device: str = 'cuda',
    ):
        self.detection_threshold = detection_threshold
        self.similarity_threshold = similarity_threshold
        self.device = device
        
        # Initialize face analysis
        print("Initializing face detection and recognition models...")
        self.app = FaceAnalysis(
            name='buffalo_l',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0 if device == 'cuda' else -1, det_thresh=detection_threshold)
        
        # Face database for recognition
        self.face_database: Dict[str, np.ndarray] = {}
        
        print("Face detector initialized successfully")
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in an image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of face dictionaries with bbox, landmarks, and embedding
        """
        faces = self.app.get(image)
        
        results = []
        for face in faces:
            face_dict = {
                'bbox': face.bbox.astype(int),  # [x1, y1, x2, y2]
                'landmarks': face.kps.astype(int),  # 5 facial landmarks
                'det_score': float(face.det_score),
                'embedding': face.normed_embedding,
                'gender': face.gender,
                'age': face.age,
            }
            results.append(face_dict)
        
        return results
    
    def extract_face_embedding(self, image: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """Extract face embedding from a face region"""
        faces = self.detect_faces(image)
        
        if not faces:
            return None
        
        # Find face closest to bbox
        best_face = None
        best_iou = 0
        
        for face in faces:
            iou = self._calculate_iou(bbox, face['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_face = face
        
        return best_face['embedding'] if best_face else None
    
    def _calculate_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Calculate IoU between two bboxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def register_face(self, name: str, image_path: str) -> bool:
        """
        Register a face in the database from reference image
        
        Args:
            name: Name/keyword for the face
            image_path: Path to reference image
            
        Returns:
            Success status
        """
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return False
        
        faces = self.detect_faces(image)
        
        if not faces:
            print(f"Warning: No face detected in {image_path}")
            return False
        
        # Use the largest face
        largest_face = max(faces, key=lambda f: (f['bbox'][2] - f['bbox'][0]) * (f['bbox'][3] - f['bbox'][1]))
        
        self.face_database[name] = largest_face['embedding']
        print(f"Registered face: {name}")
        
        return True
    
    def register_faces_from_directory(self, name: str, directory: str) -> bool:
        """
        Register a face using multiple images from a directory
        Average embeddings for better accuracy
        
        Args:
            name: Name/keyword for the face
            directory: Directory containing reference images
            
        Returns:
            Success status
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"Error: Directory {directory} does not exist")
            return False
        
        embeddings = []
        image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        
        for img_path in dir_path.iterdir():
            if img_path.suffix.lower() in image_extensions:
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                
                faces = self.detect_faces(image)
                if faces:
                    # Use largest face
                    largest_face = max(faces, key=lambda f: (f['bbox'][2] - f['bbox'][0]) * (f['bbox'][3] - f['bbox'][1]))
                    embeddings.append(largest_face['embedding'])
        
        if not embeddings:
            print(f"Warning: No faces detected in directory {directory}")
            return False
        
        # Average embeddings
        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)  # Normalize
        
        self.face_database[name] = avg_embedding
        print(f"Registered face '{name}' from {len(embeddings)} images")
        
        return True
    
    def identify_face(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Identify a face by comparing to database
        
        Args:
            embedding: Face embedding to identify
            
        Returns:
            Tuple of (name, similarity_score) or (None, 0.0) if no match
        """
        if not self.face_database:
            return None, 0.0
        
        best_match = None
        best_similarity = 0.0
        
        for name, ref_embedding in self.face_database.items():
            # Cosine similarity
            similarity = float(np.dot(embedding, ref_embedding))
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name
        
        if best_similarity >= self.similarity_threshold:
            return best_match, best_similarity
        
        return None, best_similarity
    
    def save_database(self, path: str):
        """Save face database to disk"""
        with open(path, 'wb') as f:
            pickle.dump(self.face_database, f)
        print(f"Face database saved to {path}")
    
    def load_database(self, path: str):
        """Load face database from disk"""
        if not Path(path).exists():
            print(f"Warning: Database file {path} not found")
            return
        
        with open(path, 'rb') as f:
            self.face_database = pickle.load(f)
        print(f"Loaded {len(self.face_database)} faces from database")
    
    def get_aligned_face(self, image: np.ndarray, landmarks: np.ndarray, size: int = 512) -> np.ndarray:
        """
        Get aligned face crop for swapping
        
        Args:
            image: Input image
            landmarks: 5-point facial landmarks
            size: Output size
            
        Returns:
            Aligned face image
        """
        # Reference landmarks (5-point for 512x512 face)
        reference = np.array([
            [192.98138, 239.94708],
            [318.90277, 240.19360],
            [256.63416, 314.01935],
            [201.26117, 371.41043],
            [313.08905, 371.15118]
        ], dtype=np.float32)
        
        # Scale reference to desired size
        reference = reference * (size / 512.0)
        
        # Calculate transformation matrix
        tform = cv2.estimateAffinePartial2D(landmarks, reference)[0]
        
        # Warp image
        aligned_face = cv2.warpAffine(image, tform, (size, size), flags=cv2.INTER_LINEAR)
        
        return aligned_face


def build_face_database_from_config(config: dict, detector: FaceDetector) -> FaceDetector:
    """
    Build face database from configuration file
    
    Args:
        config: Configuration dictionary
        detector: FaceDetector instance
        
    Returns:
        FaceDetector with populated database
    """
    print("\nBuilding face database from config...")
    
    for face_key, face_config in config['faces'].items():
        name = face_config['name']
        ref_images = face_config['reference_images']
        
        if Path(ref_images).is_dir():
            detector.register_faces_from_directory(name, ref_images)
        else:
            detector.register_face(name, ref_images)
    
    print(f"\nFace database built with {len(detector.face_database)} identities\n")
    
    return detector
