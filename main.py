from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
import uvicorn
import cv2
import numpy as np
from face_swap import FaceSwapper
import os
import shutil
from pathlib import Path
import tempfile
import uuid
import logging
import base64
import requests
from auth import get_api_key
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define request models
class SwapFacesRequest(BaseModel):
    base_image_url: HttpUrl
    swap_faces_urls: List[HttpUrl]
    target_face_index: Optional[int] = None  # None means swap onto all faces

class FaceInfo(BaseModel):
    index: int
    bbox: List[float]

class DetectFacesResponse(BaseModel):
    preview_image: str  # base64 encoded image
    faces: List[FaceInfo]

app = FastAPI(title="Face Swap API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize FaceSwapper
face_swapper = FaceSwapper()

def download_image(url: str) -> str:
    """Download image from URL and save it locally."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Create a unique filename
        file_extension = url.split(".")[-1].lower()
        if file_extension not in ['jpg', 'jpeg', 'png']:
            file_extension = 'jpg'
        unique_filename = f"image_{uuid.uuid4()}.{file_extension}"
        file_path = UPLOAD_DIR / unique_filename
        
        # Save the file
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(response.raw, buffer)
        
        logger.info(f"Image downloaded successfully: {file_path}")
        return str(file_path)
    except Exception as e:
        logger.error(f"Error downloading image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.post("/detect-faces/")
async def detect_faces(request: SwapFacesRequest, api_key: str = Depends(get_api_key)):
    """Detect faces in the base image."""
    try:
        # Download base image
        base_path = download_image(str(request.base_image_url))
        
        # Detect faces in the image
        img, faces_info = face_swapper.get_face_info(base_path)
        if img is None:
            raise HTTPException(status_code=400, detail="Failed to process image")

        # Create preview image with numbered faces
        preview = face_swapper.create_face_preview(img, faces_info)
        
        # Convert preview image to base64
        _, buffer = cv2.imencode('.jpg', preview)
        preview_base64 = base64.b64encode(buffer).decode('utf-8')

        # Prepare response
        faces = [FaceInfo(index=face["index"], bbox=face["bbox"]) for face in faces_info]
        
        return {
            "preview_image": f"data:image/jpeg;base64,{preview_base64}",
            "faces": faces,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error detecting faces: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/swap-faces/")
async def swap_faces(request: SwapFacesRequest, api_key: str = Depends(get_api_key)):
    """
    Perform face swapping with the provided image URLs.
    base_image_url: URL of the base image
    swap_faces_urls: list of URLs for faces to swap
    target_face_index: index of the face to swap in base image (None means swap onto all faces)
    """
    try:
        logger.info(f"Received face swap request: {request}")
        
        # Download all images
        base_path = download_image(str(request.base_image_url))
        swap_paths = [download_image(str(url)) for url in request.swap_faces_urls]
        
        results = []
        all_results = face_swapper.process_multiple_faces(base_path, swap_paths, request.target_face_index)
        
        for idx, (result_img, success) in enumerate(all_results):
            if success:
                result_filename = f"result_{uuid.uuid4()}.jpg"
                result_path = str(UPLOAD_DIR / result_filename)
                cv2.imwrite(result_path, result_img)
                logger.info(f"Face swap successful, saved result as: {result_filename}")
                results.append({
                    "success": True,
                    "result_filename": result_filename,
                    "is_all_faces": request.target_face_index is None
                })
            else:
                logger.error(f"Face swap failed for face {idx}")
                results.append({
                    "success": False,
                    "error": "Face swap failed"
                })
        
        return {"results": results}
    except Exception as e:
        logger.error(f"Error in face swap: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/result/{filename}")
async def get_result(filename: str, api_key: str = Depends(get_api_key)):
    """Get a processed image by filename."""
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        logger.error(f"Result file not found: {file_path}")
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(file_path))

@app.on_event("startup")
async def startup_event():
    """Clean up any old files in the uploads directory."""
    logger.info("Cleaning up uploads directory")
    for file in UPLOAD_DIR.glob("*"):
        try:
            file.unlink()
            logger.info(f"Deleted file: {file}")
        except Exception as e:
            logger.error(f"Error deleting file {file}: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 