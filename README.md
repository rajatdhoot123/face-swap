# Face Swap API

This is a FastAPI-based application that provides face swapping functionality. It allows you to upload a base image and multiple face images to swap faces between them.

## Prerequisites

- Python 3.8+
- dlib
- OpenCV
- FastAPI
- Other dependencies listed in requirements.txt

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd faceswap
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Download the shape predictor file:
- Download `shape_predictor_68_face_landmarks.dat` from dlib's website
- Place it in the root directory of the project

## Running the Application

1. Start the server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Upload Base Image
- **URL**: `/upload-base-image/`
- **Method**: POST
- **Content-Type**: multipart/form-data
- **Parameter**: file (image file)
- **Returns**: JSON with filename and status

### 2. Upload Swap Faces
- **URL**: `/upload-swap-faces/`
- **Method**: POST
- **Content-Type**: multipart/form-data
- **Parameter**: files (multiple image files)
- **Returns**: JSON with filenames and status

### 3. Perform Face Swap
- **URL**: `/swap-faces/`
- **Method**: POST
- **Body**: JSON with base_image filename and list of swap_faces filenames
- **Returns**: JSON with results including success status and result filenames

### 4. Get Result Image
- **URL**: `/result/{filename}`
- **Method**: GET
- **Returns**: Image file

## Example Usage

1. Upload a base image:
```bash
curl -X POST -F "file=@base.jpg" http://localhost:8000/upload-base-image/
```

2. Upload faces to swap:
```bash
curl -X POST -F "files=@face1.jpg" -F "files=@face2.jpg" http://localhost:8000/upload-swap-faces/
```

3. Perform face swap:
```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"base_image": "base_123.jpg", "swap_faces": ["swap_456.jpg", "swap_789.jpg"]}' \
     http://localhost:8000/swap-faces/
```

4. Get the result:
```bash
curl http://localhost:8000/result/result_abc.jpg > result.jpg
```

## Notes

- The application automatically creates an `uploads` directory to store temporary files
- Files are cleaned up when the server starts
- Each upload generates a unique filename to prevent conflicts
- The face swapping process requires clear, front-facing faces in the images for best results # face-swap
