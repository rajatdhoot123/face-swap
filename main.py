from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Form, Request
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
import json

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

class StoryPage(BaseModel):
    pageNumber: int
    text: str
    imagePrompt: str
    imageUrl: Optional[str] = None
    status: Optional[str] = None
    jobId: Optional[str] = None
    jobStatus: Optional[str] = None
    replicateJobId: Optional[str] = None
    errorMessage: Optional[str] = None
    metadata: Optional[dict] = None

class Story(BaseModel):
    title: str
    description: str
    pages: List[StoryPage]

class StoryFaceSwapRequest(BaseModel):
    name: str
    gender: str
    birthDate: str
    story: Story
    uploadPathPrefix: str

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
        
        # Generate a session UUID for this request
        session_uuid = str(uuid.uuid4())
        
        # Process all faces and upload directly to R2
        all_results = face_swapper.process_multiple_faces(
            base_path, 
            swap_paths, 
            request.target_face_index,
            session_uuid
        )
        
        # Get uploaded URLs from the face swapper
        result_urls = face_swapper.get_last_upload_urls()
        
        results = []
        for idx, (result_img, success) in enumerate(all_results):
            if success:
                # Use the corresponding R2 URL if available
                result_url = result_urls[idx] if idx < len(result_urls) else ""
                logger.info(f"Face swap successful, uploaded to R2: {result_url}")
                results.append({
                    "success": True,
                    "result_url": result_url,
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

@app.post("/enhanced-swap-faces/")
async def enhanced_swap_faces(request: SwapFacesRequest, api_key: str = Depends(get_api_key)):
    """
    Perform enhanced face swapping using multiple source images to improve quality.
    This endpoint combines features from multiple source images for better results.
    """
    try:
        logger.info(f"Received enhanced face swap request: {request}")
        
        # Download all images
        base_path = download_image(str(request.base_image_url))
        swap_paths = [download_image(str(url)) for url in request.swap_faces_urls]
        
        # Generate a session UUID for this request
        session_uuid = str(uuid.uuid4())
        
        # Perform enhanced face swap
        result_img, success = face_swapper.swap_with_enhanced_face(
            swap_paths,
            base_path,
            request.target_face_index
        )
        
        if success:
            # Upload to R2 instead of local storage
            if face_swapper.r2_enabled:
                result_url = face_swapper.upload_image_to_r2(result_img, session_uuid, 0)
                logger.info(f"Enhanced face swap successful, uploaded to R2: {result_url}")
                return {
                    "success": True,
                    "result_url": result_url,
                    "is_all_faces": request.target_face_index is None
                }
            else:
                # Fallback to local storage if R2 is not configured
                result_filename = f"enhanced_result_{uuid.uuid4()}.jpg"
                result_path = str(UPLOAD_DIR / result_filename)
                cv2.imwrite(result_path, result_img)
                logger.info(f"Enhanced face swap successful, saved result as: {result_filename}")
                
                return {
                    "success": True,
                    "result_filename": result_filename,
                    "is_all_faces": request.target_face_index is None
                }
        else:
            logger.error("Enhanced face swap failed")
            raise HTTPException(status_code=400, detail="Face swap failed")
            
    except Exception as e:
        logger.error(f"Error in enhanced face swap: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/swap-faces-story/")
async def swap_faces_story(
    request: Request,
    api_key: Optional[str] = Depends(get_api_key)
):
    """
    Perform face swapping on all images in a story.
    Takes a story with image URLs and swaps faces using either uploaded photos or image URLs.
    """
    try:
        # Parse the request body
        body = await request.json()
        
        # Extract data from the JSON payload
        name = body.get('name')
        gender = body.get('gender')
        birth_date = body.get('birth_date')  # Note: changed from birthDate
        story_data = body.get('story')
        swap_faces_urls = body.get('swap_faces_urls', [])
        target_face_index = body.get('target_face_index')
        
        # Log request data for debugging
        logger.info(f"Received request with: name={name}, gender={gender}")
        logger.info(f"birth_date={birth_date}")
        logger.info(f"Swap faces URLs count: {len(swap_faces_urls)}")
        logger.info(f"Story data (truncated): {str(story_data)[:200]}...")
        
        # Parse the story JSON string or dict into our model
        try:
            if isinstance(story_data, str):
                story_obj = Story(**json.loads(story_data))
            else:
                story_obj = Story(**story_data)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in story field: {e}")
            return JSONResponse(
                status_code=422,
                content={"detail": f"Invalid JSON in story field: {str(e)}"}
            )
        except Exception as e:
            logger.error(f"Error parsing story data: {e}")
            return JSONResponse(
                status_code=422,
                content={"detail": f"Error parsing story data: {str(e)}"}
            )
        
        # Validate swap_faces_urls
        if len(swap_faces_urls) == 0:
            return JSONResponse(
                status_code=422,
                content={"detail": "No face images provided"}
            )
        
        logger.info(f"Received story face swap request for story: {story_obj.title}")
        
        # Download all swap face images
        swap_paths = []
        for url in swap_faces_urls:
            swap_path = download_image(url)
            swap_paths.append(swap_path)
            logger.info(f"Downloaded swap face image to: {swap_path}")
        
        # Process each page image
        result_story = story_obj.dict()
        session_uuid = str(uuid.uuid4())
        
        for i, page in enumerate(story_obj.pages):
            if not page.imageUrl:
                logger.warning(f"No image URL for page {i}, skipping")
                continue
                
            try:
                # Download the base image
                base_path = download_image(str(page.imageUrl))
                
                # Swap faces
                result_img, success = face_swapper.swap_with_enhanced_face(
                    swap_paths,
                    base_path,
                    target_face_index  # Use the target face index from the request
                )
                
                if success:
                    # Upload to R2 or save locally
                    if face_swapper.r2_enabled:
                        result_url = face_swapper.upload_image_to_r2(result_img, session_uuid, i)
                        result_story["pages"][i]["imageUrl"] = result_url
                        logger.info(f"Face swap successful for page {i}, uploaded to: {result_url}")
                    else:
                        # Fallback to local storage
                        result_filename = f"story_faceswap_{session_uuid}_{i}.jpg"
                        result_path = str(UPLOAD_DIR / result_filename)
                        cv2.imwrite(result_path, result_img)
                        result_story["pages"][i]["imageUrl"] = f"/result/{result_filename}"
                        logger.info(f"Face swap successful for page {i}, saved as: {result_filename}")
                else:
                    logger.warning(f"Face swap failed for page {i}")
            except Exception as e:
                logger.error(f"Error processing page {i}: {str(e)}")
                # Continue with other pages even if one fails
        
        # Clean up temporary files
        for path in swap_paths:
            try:
                os.unlink(path)
            except Exception as e:
                logger.error(f"Error deleting temporary file {path}: {str(e)}")
        
        # Add the additional request fields to the response
        response = {
            "name": name,
            "gender": gender,
            "birthDate": birth_date,
            "story": result_story
        }
        
        return response
    except Exception as e:
        logger.error(f"Error in story face swap: {str(e)}")
        # Return more detailed error info
        return JSONResponse(
            status_code=400,
            content={"detail": str(e), "type": str(type(e))}
        )

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