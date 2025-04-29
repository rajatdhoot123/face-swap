import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import os
import boto3
import uuid
import json
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
import logging
from datetime import datetime

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
        
        # Configure R2 client
        self.r2_enabled = all([
            os.environ.get('R2_ACCESS_KEY_ID'),
            os.environ.get('R2_SECRET_ACCESS_KEY'),
            os.environ.get('R2_ACCOUNT_ID'),
            os.environ.get('R2_BUCKET_NAME')
        ])
        
        if self.r2_enabled:
            self.r2_client = boto3.client(
                's3',
                endpoint_url=f"https://{os.environ.get('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
                aws_access_key_id=os.environ.get('R2_ACCESS_KEY_ID'),
                aws_secret_access_key=os.environ.get('R2_SECRET_ACCESS_KEY'),
                region_name=os.environ.get('AWS_REGION', 'auto')
            )
            self.r2_bucket = os.environ.get('R2_BUCKET_NAME')
            self.r2_project_folder = os.environ.get('R2_PROJECT_FOLDER', 'storyowl')
            self.r2_public_domain = os.environ.get('NEXT_PUBLIC_R2_PUBLIC_DOMAIN', 'assets.storyowl.app')
            logger.info("R2 storage configuration loaded successfully")
        else:
            logger.warning("R2 storage configuration not found or incomplete")

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

    def upload_image_to_r2(self, image: np.ndarray, session_uuid: str, index: int = 0, upload_path_prefix: Optional[str] = None) -> str:
        """
        Upload an image to Cloudflare R2 storage.
        
        Args:
            image: The image to upload
            session_uuid: UUID for the current session
            index: Index of the image for multiple outputs
            upload_path_prefix: Custom path prefix for the upload (overrides default path construction)
            
        Returns:
            Public URL of the uploaded image
        """
        if not self.r2_enabled:
            logger.error("R2 storage is not configured")
            return ""
            
        try:
            # Create timestamp for unique filename
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"output_{timestamp}_{index}.jpg"
            
            # Create the R2 path - use custom prefix if provided
            if upload_path_prefix:
                r2_key = f"{self.r2_project_folder}/{upload_path_prefix}output/{filename}"
            else:
                r2_key = f"{self.r2_project_folder}/quick-start/{session_uuid}/output/{filename}"
            
            # Encode image to jpg
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Upload to R2
            self.r2_client.put_object(
                Bucket=self.r2_bucket,
                Key=r2_key,
                Body=buffer.tobytes(),
                ContentType='image/jpeg'
            )
            
            # Return the public URL
            public_url = f"https://{self.r2_public_domain}/{r2_key}"
            logger.info(f"Image uploaded to {public_url}")
            return public_url
            
        except Exception as e:
            logger.error(f"Error uploading image to R2: {str(e)}")
            return ""

    def process_multiple_faces(self, base_img: str, face_images: List[str], target_face_index: Optional[int] = None, 
                               session_uuid: Optional[str] = None, upload_path_prefix: Optional[str] = None) -> List[Tuple[np.ndarray, bool]]:
        """
        Process multiple face swaps with the same base image.
        target_face_index: if provided, swap onto specific face; if None, swap onto all faces
        ALWAYS returns a list of (result_image, success) tuples - R2 upload status is tracked separately.
        """
        logger.info(f"Processing multiple faces. Base image: {base_img}, Face images: {face_images}, Target face index: {target_face_index}")
        results = []
        self.result_urls = []  # Store URLs in the instance

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

        # Generate UUID if not provided
        if session_uuid is None:
            session_uuid = str(uuid.uuid4())

        for idx, face_img in enumerate(face_images):
            if target_face_index is not None:
                # Single face swap mode
                result, success = self.swap_specific_face(face_img, base_img, target_face_index)
            else:
                # Swap onto all faces in the same image
                result, success = self.swap_all_faces(face_img, base_img)
                
            results.append((result, success))
            
            # Upload to R2 if successful
            if success and result is not None and self.r2_enabled and session_uuid:
                url = self.upload_image_to_r2(result, session_uuid, idx, upload_path_prefix)
                if url:
                    self.result_urls.append(url)
        
        return results

    def get_last_upload_urls(self) -> List[str]:
        """
        Get the URLs of the last uploaded images.
        """
        if hasattr(self, 'result_urls'):
            return self.result_urls
        return []

    def swap_with_enhanced_face_in_memory(self, source_images: List[np.ndarray], target_img: np.ndarray, 
                                         target_face_index: Optional[int] = None) -> Tuple[np.ndarray, bool]:
        """
        Swap faces using in-memory images without saving to disk.
        
        Args:
            source_images: List of source face images as numpy arrays
            target_img: Target image as numpy array
            target_face_index: Index of face to swap in target (None for all faces)
            
        Returns:
            Tuple of (result_image, success)
        """
        logger.info(f"Attempting in-memory face swap with {len(source_images)} source images")
        
        if not source_images:
            logger.error("No source images provided")
            return None, False
            
        try:
            # Process the target image
            target_faces = self.app.get(target_img)
            
            if not target_faces:
                logger.error("No faces found in target image")
                return None, False
                
            # Use first source image only (for now)
            source_face = self.app.get(source_images[0])
            
            if not source_face:
                logger.error("No face found in source image")
                return None, False
                
            # Get primary source face
            primary_source_face = source_face[0]
            
            if target_face_index is not None:
                # Single face swap
                if target_face_index < 0 or target_face_index >= len(target_faces):
                    logger.error(f"Target face index {target_face_index} is out of range. Found {len(target_faces)} faces.")
                    return None, False
                    
                target_face = target_faces[target_face_index]
                result_img = self.swapper.get(target_img.copy(), target_face, primary_source_face, paste_back=True)
                logger.info("Single face swap completed successfully")
                return result_img, True
            else:
                # Swap all faces
                result_img = target_img.copy()
                for target_face in target_faces:
                    result_img = self.swapper.get(result_img, target_face, primary_source_face, paste_back=True)
                logger.info("All faces swap completed successfully")
                return result_img, True
                
        except Exception as e:
            logger.error(f"Error during in-memory face swap: {str(e)}")
            return None, False

    def combine_source_faces(self, source_faces: List[Dict]) -> Dict:
        """
        Combine multiple source faces to create an enhanced face representation.
        This method averages the facial features from multiple source images.
        """
        if not source_faces:
            return None
            
        # Initialize combined face data
        combined_face = source_faces[0].copy()
        
        # If only one face, return it directly
        if len(source_faces) == 1:
            return combined_face
            
        # Average the facial landmarks and features
        for face in source_faces[1:]:
            # Average the bounding box
            combined_face["bbox"] = [
                (a + b) / 2 for a, b in zip(combined_face["bbox"], face["bbox"])
            ]
            
            # Average the facial landmarks if available
            if hasattr(face["face"], "kps") and hasattr(combined_face["face"], "kps"):
                combined_face["face"].kps = (combined_face["face"].kps + face["face"].kps) / 2
                
            # Average the face embedding if available
            if hasattr(face["face"], "embedding") and hasattr(combined_face["face"], "embedding"):
                combined_face["face"].embedding = (combined_face["face"].embedding + face["face"].embedding) / 2
        
        return combined_face

def handle_face_swap_request(request_data: Dict) -> Dict:
    """
    Handle a face swap request and return appropriate response.
    This is the MAIN ENTRY POINT for face swapping operations.
    Always returns a standardized JSON response.
    
    Args:
        request_data: Dictionary containing request parameters
        
    Returns:
        Response dictionary with results
    """
    try:
        # Extract parameters from request
        base_image = request_data.get('base_image_url')
        swap_faces = request_data.get('swap_faces_urls', [])
        target_face_index = request_data.get('target_face_index')
        session_uuid = request_data.get('session_uuid', str(uuid.uuid4()))
        upload_path_prefix = request_data.get('uploadPathPrefix')
        
        if not base_image or not swap_faces:
            return {
                "success": False,
                "error": "Missing required parameters: base_image_url or swap_faces_urls",
                "result_urls": []
            }
            
        # Initialize face swapper
        face_swapper = FaceSwapper()
        
        # Download images if they are URLs
        # Code for downloading would go here
        
        # Process face swap for all images - ALWAYS returns list of tuples
        results = face_swapper.process_multiple_faces(
            base_img=base_image,
            face_images=swap_faces,
            target_face_index=target_face_index,
            session_uuid=session_uuid,
            upload_path_prefix=upload_path_prefix
        )
        
        # Get any uploaded URLs
        result_urls = face_swapper.get_last_upload_urls()
        
        # Check success status
        if not results:
            return {
                "success": False,
                "error": "No results returned from face swap operation"
            }
            
        # Count successful swaps
        successful_swaps = sum(1 for _, success in results if success)
        
        # Check if any successful swaps occurred
        if successful_swaps == 0:
            return {
                "success": False,
                "error": "Face swap failed for all images"
            }
            
        # Return result in the format client expects
        if result_urls:
            # If we have R2 URLs, use the first one
            result_filename = result_urls[0].split('/')[-1]
            return {
                "success": True,
                "result_filename": result_filename,
                "result_url": result_urls[0],
                "result_urls": result_urls,
                "is_all_faces": target_face_index is None
            }
        else:
            # If no R2 upload, generate a local filename
            result_filename = f"result_{session_uuid}.jpg"
            return {
                "success": True,
                "result_filename": result_filename,
                "is_all_faces": target_face_index is None
            }
            
    except Exception as e:
        logger.error(f"Error in face swap: {str(e)}")
        return {
            "success": False,
            "error": f"Error in face swap: {str(e)}"
        }
        
    return {
        "success": False,
        "error": "Unknown error occurred"
    } 