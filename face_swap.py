import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import os
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceSwapper:
    def __init__(self):
        # Initialize the face analyzer
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        # Get the face swapper
        model_path = Path(__file__).parent / "inswapper_128.onnx"
        if not model_path.exists():
            raise FileNotFoundError(
                "Please download the inswapper_128.onnx model from InsightFace and place it in the project directory"
            )
        self.swapper = insightface.model_zoo.get_model(str(model_path))

    def process_image(self, img_path: str):
        """Process an image and return faces."""
        logger.info(f"Processing image: {img_path}")
        if not os.path.exists(img_path):
            logger.error(f"Image file not found: {img_path}")
            return None, []
            
        img = cv2.imread(img_path)
        if img is None:
            logger.error(f"Failed to read image: {img_path}")
            return None, []
            
        faces = self.app.get(img)
        logger.info(f"Found {len(faces)} faces in image: {img_path}")
        return img, faces

    def get_face_info(self, img_path: str) -> Tuple[Optional[np.ndarray], List[Dict]]:
        """
        Get information about all faces in an image.
        Returns the image and a list of face information dictionaries.
        """
        img, faces = self.process_image(img_path)
        if img is None:
            return None, []
        
        face_info = []
        for idx, face in enumerate(faces):
            bbox = face.bbox.astype(int)
            face_info.append({
                "index": idx,
                "bbox": bbox.tolist(),
                "face": face
            })
        
        return img, face_info

    def create_face_preview(self, img: np.ndarray, faces_info: List[Dict]) -> np.ndarray:
        """Create a preview image with numbered faces."""
        preview = img.copy()
        for face in faces_info:
            bbox = face["bbox"]
            idx = face["index"]
            # Draw rectangle around face
            cv2.rectangle(preview, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            # Put face number
            cv2.putText(preview, str(idx), (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return preview

    def swap_specific_face(self, source_img: str, target_img: str, target_face_index: int) -> Tuple[np.ndarray, bool]:
        """
        Swap face from source image onto a specific face in target image.
        target_face_index: index of the face to swap in target image
        """
        logger.info(f"Attempting face swap from {source_img} to face {target_face_index} in {target_img}")
        
        # Read and process images
        source_img_array, source_faces = self.process_image(source_img)
        target_img_array, target_faces = self.process_image(target_img)
        
        if source_img_array is None or target_img_array is None:
            logger.error("Failed to read source or target image")
            return None, False
        
        if not source_faces:
            logger.error("No face found in source image")
            return None, False
            
        if not target_faces:
            logger.error("No faces found in target image")
            return None, False
            
        if target_face_index >= len(target_faces):
            logger.error(f"Target face index {target_face_index} is out of range. Only {len(target_faces)} faces found.")
            return None, False

        # Get the first face from source and specified face from target
        source_face = source_faces[0]
        target_face = target_faces[target_face_index]
        
        try:
            # Perform face swap
            logger.info("Performing face swap operation")
            result = self.swapper.get(target_img_array, target_face, source_face, paste_back=True)
            logger.info("Face swap completed successfully")
            return result, True
        except Exception as e:
            logger.error(f"Error during face swap: {str(e)}")
            return None, False

    def swap_all_faces(self, source_img: str, target_img: str) -> Tuple[np.ndarray, bool]:
        """
        Swap face from source image onto all faces in target image.
        Returns (result_image, success) with all faces swapped in the same image.
        """
        logger.info(f"Attempting face swap from {source_img} to all faces in {target_img}")
        
        # Read and process images
        source_img_array, source_faces = self.process_image(source_img)
        target_img_array, target_faces = self.process_image(target_img)
        
        if source_img_array is None or target_img_array is None:
            logger.error("Failed to read source or target image")
            return None, False
        
        if not source_faces:
            logger.error("No face found in source image")
            return None, False
            
        if not target_faces:
            logger.error("No faces found in target image")
            return None, False

        try:
            result_img = target_img_array.copy()
            source_face = source_faces[0]  # Use first face from source image
            
            # Apply face swap to all faces in the same image
            for target_face in target_faces:
                result_img = self.swapper.get(result_img, target_face, source_face, paste_back=True)
                logger.info("Successfully swapped face")
            
            return result_img, True
        except Exception as e:
            logger.error(f"Error during face swap: {str(e)}")
            return None, False

    def process_multiple_faces(self, base_img: str, face_images: List[str], target_face_index: Optional[int] = None) -> List[Tuple[np.ndarray, bool]]:
        """
        Process multiple face swaps with the same base image.
        target_face_index: if provided, swap onto specific face; if None, swap onto all faces
        Returns a list of (result_image, success) tuples.
        """
        logger.info(f"Processing multiple faces. Base image: {base_img}, Face images: {face_images}, Target face index: {target_face_index}")
        results = []

        # First verify the base image has faces
        base_img_array, base_faces = self.process_image(base_img)
        if base_img_array is None or not base_faces:
            logger.error("No faces found in base image")
            return [(None, False)] * len(face_images)

        if target_face_index is not None:
            # Verify target face index is valid
            if target_face_index < 0 or target_face_index >= len(base_faces):
                logger.error(f"Invalid target face index {target_face_index}. Found {len(base_faces)} faces.")
                return [(None, False)] * len(face_images)

        for face_img in face_images:
            if target_face_index is not None:
                # Single face swap mode
                result, success = self.swap_specific_face(face_img, base_img, target_face_index)
                results.append((result, success))
            else:
                # Swap onto all faces in the same image
                result, success = self.swap_all_faces(face_img, base_img)
                results.append((result, success))
        
        return results 