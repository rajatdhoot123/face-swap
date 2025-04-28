from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define request models
class SwapFacesRequest(BaseModel):
    base_image: str
    swap_faces: List[str]
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

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.post("/upload-base-image/")
async def upload_base_image(file: UploadFile = File(...)):
    """Upload the base image and return detected faces."""
    try:
        # Create a unique filename
        file_extension = file.filename.split(".")[-1]
        unique_filename = f"base_{uuid.uuid4()}.{file_extension}"
        file_path = UPLOAD_DIR / unique_filename
        
        logger.info(f"Uploading base image: {file.filename} -> {unique_filename}")
        
        # Save the uploaded file
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Base image saved successfully: {file_path}")

        # Detect faces in the image
        img, faces_info = face_swapper.get_face_info(str(file_path))
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
            "filename": unique_filename,
            "preview_image": f"data:image/jpeg;base64,{preview_base64}",
            "faces": faces,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error uploading base image: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/upload-swap-faces/")
async def upload_swap_faces(files: List[UploadFile] = File(...)):
    """Upload multiple faces to be swapped."""
    try:
        filenames = []
        for file in files:
            file_extension = file.filename.split(".")[-1]
            unique_filename = f"swap_{uuid.uuid4()}.{file_extension}"
            file_path = UPLOAD_DIR / unique_filename
            
            logger.info(f"Uploading swap face: {file.filename} -> {unique_filename}")
            
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            logger.info(f"Swap face saved successfully: {file_path}")
            filenames.append(unique_filename)
        
        return {"filenames": filenames, "status": "success"}
    except Exception as e:
        logger.error(f"Error uploading swap faces: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/swap-faces/")
async def swap_faces(request: SwapFacesRequest):
    """
    Perform face swapping with the provided images.
    base_image: filename of the base image
    swap_faces: list of filenames for faces to swap
    target_face_index: index of the face to swap in base image (None means swap onto all faces)
    """
    try:
        logger.info(f"Received face swap request: {request}")
        base_path = str(UPLOAD_DIR / request.base_image)
        swap_paths = [str(UPLOAD_DIR / face) for face in request.swap_faces]
        
        # Verify files exist
        if not os.path.exists(base_path):
            logger.error(f"Base image not found: {base_path}")
            raise HTTPException(status_code=404, detail=f"Base image not found: {request.base_image}")
        
        for path in swap_paths:
            if not os.path.exists(path):
                logger.error(f"Swap face image not found: {path}")
                raise HTTPException(status_code=404, detail=f"Swap face image not found: {path}")
        
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
                    "original_face": request.swap_faces[idx],
                    "is_all_faces": request.target_face_index is None
                })
            else:
                logger.error(f"Face swap failed for: {request.swap_faces[idx]}")
                results.append({
                    "success": False,
                    "error": "Face swap failed",
                    "original_face": request.swap_faces[idx]
                })
        
        return {"results": results}
    except Exception as e:
        logger.error(f"Error in face swap: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/result/{filename}")
async def get_result(filename: str):
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