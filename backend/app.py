from fastapi import FastAPI, File, UploadFile, HTTPException
import torch
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
from uuid import uuid4 
from ultralytics import YOLO
import logging
import glob
import ultralytics
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from fastapi.staticfiles import StaticFiles
import traceback

app = FastAPI()

# Configure logging with more details
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Enhanced CORS configuration
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",  # Common port for live server extensions
    "*"  # Allow all origins for testing purposes
]

# Configure CORS middleware with more options
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Create directories for results if they don't exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("results/explanations", exist_ok=True)
os.makedirs("results/gradcam", exist_ok=True)

# Load YOLOv8 model
try:
    logger.info("Loading YOLOv8 model...")
    logger.info(f"Ultralytics version: {ultralytics.__version__}")
    model = YOLO('models/model.pt')
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    logger.error(traceback.format_exc())
    raise e

# Mount the static directories
app.mount("/results", StaticFiles(directory="results"), name="results")
# Also mount the uploads directory for debugging purposes
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

@app.get("/")
async def root():
    return {"message": "Welcome to the YOLOv8 detection API!"}

@app.get("/status")
async def get_status():
    return {"message": "Server is running", "status": "success"}

# Add this health check route for troubleshooting
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is running properly"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Check if file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image")
    
    # Create a unique ID for this detection
    detection_id = str(uuid4())
    
    # Create paths for saving
    upload_folder = "uploads"
    os.makedirs(upload_folder, exist_ok=True)
    
    file_extension = os.path.splitext(file.filename)[1]
    input_file_path = f"{upload_folder}/{detection_id}{file_extension}"
    
    # Save the uploaded file
    with open(input_file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    try:
        # Load image with PIL for processing
        image = Image.open(input_file_path)
        image_array = np.array(image)
        
        # Run YOLOv8 inference
        results = model(image_array)
        
        # Save the visualization image
        output_file_path = f"results/{detection_id}_result.jpg"
        results_plotted = results[0].plot()
        cv2.imwrite(output_file_path, results_plotted)
        
        # Extract detection information
        detections = []
        detection_boxes = []
        
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0]
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                detection_boxes.append([x1, y1, x2, y2])
                
                detections.append({
                    "id": i,
                    "class": class_name,
                    "confidence": float(confidence),
                    "box": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2)
                    }
                })
        
        # Generate simple explanation by highlighting the detection areas
        explanation_file_path = None
        if len(detection_boxes) > 0:
            # Create a copy of the original image for the explanation
            explanation_img = image_array.copy()
            
            # Draw rectangles for the detections
            for box in detection_boxes:
                x1, y1, x2, y2 = [int(coord) for coord in box]
                cv2.rectangle(explanation_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add semi-transparent red highlight around the fracture area
                highlight = np.zeros_like(explanation_img, dtype=np.uint8)
                # Create a slightly larger box for highlight area
                pad = 10
                cv2.rectangle(highlight, (max(0, x1-pad), max(0, y1-pad)), 
                             (min(explanation_img.shape[1], x2+pad), min(explanation_img.shape[0], y2+pad)), 
                             (0, 0, 255), -1)
                # Apply highlight with transparency
                explanation_img = cv2.addWeighted(explanation_img, 1, highlight, 0.3, 0)
            
            explanation_file_path = f"results/explanations/{detection_id}_explanation.jpg"
            cv2.imwrite(explanation_file_path, explanation_img)
        
        # Generate Grad-CAM visualization
        gradcam_file_path = None
        if len(detection_boxes) > 0:
            # Generate Grad-CAM heatmap
            gradcam_img = generate_gradcam(image_array, detection_boxes, results[0])
            
            gradcam_file_path = f"results/gradcam/{detection_id}_gradcam.jpg"
            cv2.imwrite(gradcam_file_path, gradcam_img)
        
        return {
            "detection_id": detection_id,
            "message": "Detection completed successfully",
            "result_image": f"/results/{detection_id}_result.jpg",
            "explanation_image": f"/results/explanations/{detection_id}_explanation.jpg" if explanation_file_path else None,
            "gradcam_image": f"/results/gradcam/{detection_id}_gradcam.jpg" if gradcam_file_path else None,
            "detections": detections
        }
    
    except Exception as e:
        logger.error(f"Error during detection: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error during detection: {str(e)}")

def generate_gradcam(image, boxes, result):
    """
    Generate a Grad-CAM visualization for the detected fractures.
    
    This is a simplified implementation - in a production environment, you would use
    actual gradients from the model's final convolutional layer.
    """
    # Create a copy of the image for visualization
    vis_img = image.copy()
    
    # Convert to RGB if grayscale
    if len(vis_img.shape) == 2:
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2RGB)
    elif vis_img.shape[2] == 1:
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2RGB)
    
    # Create a heatmap overlay
    height, width = vis_img.shape[:2]
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    # For each detection, add a gaussian blob to the heatmap centered on the detection
    for box in boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box]
        
        # Calculate center of the box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Box dimensions
        box_width = x2 - x1
        box_height = y2 - y1
        
        # Create a gaussian blob around the center
        y, x = np.ogrid[:height, :width]
        # Adjust sigma based on box size
        sigma_x = box_width / 6  # Cover the full box width with 3 sigma
        sigma_y = box_height / 6
        
        # Ensure sigma is not too small
        sigma_x = max(sigma_x, 10)
        sigma_y = max(sigma_y, 10)
        
        # Generate gaussian
        gaussian = np.exp(-(
            ((x - center_x) ** 2) / (2 * sigma_x ** 2) + 
            ((y - center_y) ** 2) / (2 * sigma_y ** 2)
        ))
        
        # Add to heatmap
        heatmap = np.maximum(heatmap, gaussian)
    
    # Normalize heatmap
    heatmap = np.uint8(255 * heatmap)
    
    # Apply colormap to create a visualization
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Overlay the heatmap on the image with transparency
    alpha = 0.4
    gradcam_visualization = cv2.addWeighted(vis_img, 1 - alpha, heatmap_colored, alpha, 0)
    
    # Draw the detection boxes
    for box in boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box]
        cv2.rectangle(gradcam_visualization, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return gradcam_visualization

@app.get("/gradcam/{image_id}")
async def get_gradcam(image_id: str):
    """
    Generate custom Grad-CAM visualization for a previously detected image
    """
    # Find the original result image
    result_path = f"results/{image_id}_result.jpg"
    if not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Generate path for Grad-CAM
    gradcam_path = f"results/gradcam/{image_id}_gradcam.jpg"
    
    # If already exists, return it
    if os.path.exists(gradcam_path):
        return FileResponse(gradcam_path)
    
    # Otherwise, would typically regenerate it here
    # For this example, we'll return an error if it doesn't exist
    raise HTTPException(status_code=404, detail="Grad-CAM not available for this image")
