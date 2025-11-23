import os
import sys
import time
import shutil
from datetime import datetime
from typing import List
from pathlib import Path
import asyncio
import threading

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocessing import ImagePreprocessor
from src.model import FruitClassifier
from src.prediction import FruitPredictor

# Global variables
app = FastAPI(title="Fruit Classification API", version="1.0.0")
predictor = None
model_start_time = datetime.now()
prediction_count = 0
retraining_status = {"is_retraining": False, "progress": 0, "message": ""}

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "fruit_classifier.h5"
UPLOAD_DIR = BASE_DIR / "uploads"
RETRAIN_DATA_DIR = BASE_DIR / "data" / "retrain"

# Create directories
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RETRAIN_DATA_DIR.mkdir(parents=True, exist_ok=True)


# Pydantic models
class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict
    interpretation: str
    timestamp: str


class ModelInfo(BaseModel):
    uptime_seconds: float
    prediction_count: int
    model_info: dict
    status: str


class RetrainingStatus(BaseModel):
    is_retraining: bool
    progress: int
    message: str


@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    global predictor
    try:
        predictor = FruitPredictor(str(MODEL_PATH))
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        predictor = None


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fruit Classification API</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .container {
                background: rgba(255, 255, 255, 0.1);
                padding: 30px;
                border-radius: 10px;
                backdrop-filter: blur(10px);
            }
            h1 { color: #fff; }
            .endpoint {
                background: rgba(255, 255, 255, 0.2);
                padding: 15px;
                margin: 10px 0;
                border-radius: 5px;
            }
            .method {
                display: inline-block;
                padding: 5px 10px;
                border-radius: 3px;
                font-weight: bold;
                margin-right: 10px;
            }
            .get { background: #61affe; }
            .post { background: #49cc90; }
            a { color: #fff; text-decoration: none; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🍎 Fruit Classification API</h1>
            <p>Welcome to the Fruit Classification ML Pipeline API</p>
            
            <h2>Available Endpoints:</h2>
            
            <div class="endpoint">
                <span class="method post">POST</span>
                <strong>/predict</strong> - Predict fruit class from uploaded image
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span>
                <strong>/health</strong> - Check API health status
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span>
                <strong>/model-info</strong> - Get model information and uptime
            </div>
            
            <div class="endpoint">
                <span class="method post">POST</span>
                <strong>/upload-training-data</strong> - Upload bulk training data
            </div>
            
            <div class="endpoint">
                <span class="method post">POST</span>
                <strong>/retrain</strong> - Trigger model retraining
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span>
                <strong>/retraining-status</strong> - Check retraining status
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span>
                <strong>/docs</strong> - Interactive API documentation
            </div>
            
            <p style="margin-top: 30px;">
                <a href="/docs" style="background: #49cc90; padding: 10px 20px; border-radius: 5px;">
                    📖 View API Documentation
                </a>
            </p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if predictor is None or predictor.model is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "Model not loaded"}
        )
    
    uptime = (datetime.now() - model_start_time).total_seconds()
    return {
        "status": "healthy",
        "uptime_seconds": round(uptime, 2),
        "prediction_count": prediction_count,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model-info", response_model=ModelInfo)
async def get_model_info():
    """Get model information and statistics."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    uptime = (datetime.now() - model_start_time).total_seconds()
    model_info = predictor.get_model_info()
    
    return {
        "uptime_seconds": round(uptime, 2),
        "prediction_count": prediction_count,
        "model_info": model_info,
        "status": "operational"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict the class of an uploaded fruit image.
    
    Args:
        file: Image file (jpg, jpeg, png)
        
    Returns:
        Prediction results with class, confidence, and probabilities
    """
    global prediction_count
    
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Make prediction
        start_time = time.time()
        result = predictor.predict_from_bytes(image_bytes)
        prediction_time = time.time() - start_time
        
        # Increment counter
        prediction_count += 1
        
        # Add prediction time to result
        result['prediction_time_ms'] = round(prediction_time * 1000, 2)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/upload-training-data")
async def upload_training_data(
    files: List[UploadFile] = File(...),
    class_name: str = Form(...)
):
    """
    Upload multiple images for retraining.
    
    Args:
        files: List of image files
        class_name: Class name for the images
        
    Returns:
        Upload status
    """
    if class_name not in predictor.class_names:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid class name. Must be one of: {predictor.class_names}"
        )
    
    # Create class directory
    class_dir = RETRAIN_DATA_DIR / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    
    uploaded_files = []
    errors = []
    
    for file in files:
        if not file.content_type.startswith('image/'):
            errors.append(f"{file.filename}: Not an image file")
            continue
        
        try:
            # Save file
            file_path = class_dir / file.filename
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            uploaded_files.append(file.filename)
        except Exception as e:
            errors.append(f"{file.filename}: {str(e)}")
    
    return {
        "uploaded_count": len(uploaded_files),
        "uploaded_files": uploaded_files,
        "errors": errors,
        "class_name": class_name,
        "timestamp": datetime.now().isoformat()
    }


def retrain_model_task():
    """Background task for model retraining."""
    global retraining_status, predictor, model_start_time
    
    try:
        retraining_status = {
            "is_retraining": True,
            "progress": 0,
            "message": "Initializing retraining..."
        }
        
        # Check if training data exists
        if not any(RETRAIN_DATA_DIR.iterdir()):
            retraining_status = {
                "is_retraining": False,
                "progress": 0,
                "message": "Error: No training data found"
            }
            return
        
        retraining_status["progress"] = 10
        retraining_status["message"] = "Loading data..."
        
        # Initialize components
        preprocessor = ImagePreprocessor()
        classifier = FruitClassifier(model_type='mobilenet')
        
        retraining_status["progress"] = 20
        retraining_status["message"] = "Creating data generators..."
        
        # Create data generators
        train_gen, val_gen, _ = preprocessor.create_data_generators(
            str(RETRAIN_DATA_DIR),
            str(RETRAIN_DATA_DIR),  # Using same for validation in this case
            batch_size=16
        )
        
        retraining_status["progress"] = 30
        retraining_status["message"] = "Starting retraining..."
        
        # Retrain model
        classifier.retrain(
            train_gen,
            val_gen,
            epochs=10,
            model_path=str(MODEL_PATH)
        )
        
        retraining_status["progress"] = 90
        retraining_status["message"] = "Reloading model..."
        
        # Reload predictor
        predictor = FruitPredictor(str(MODEL_PATH))
        model_start_time = datetime.now()
        
        retraining_status = {
            "is_retraining": False,
            "progress": 100,
            "message": "Retraining completed successfully!"
        }
        
    except Exception as e:
        retraining_status = {
            "is_retraining": False,
            "progress": 0,
            "message": f"Error during retraining: {str(e)}"
        }


@app.post("/retrain")
async def trigger_retraining(background_tasks: BackgroundTasks):
    """
    Trigger model retraining with uploaded data.
    
    Returns:
        Retraining status
    """
    global retraining_status
    
    if retraining_status.get("is_retraining", False):
        raise HTTPException(
            status_code=400,
            detail="Retraining already in progress"
        )
    
    # Start retraining in background
    thread = threading.Thread(target=retrain_model_task)
    thread.daemon = True
    thread.start()
    
    return {
        "message": "Retraining started",
        "status": "in_progress",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/retraining-status", response_model=RetrainingStatus)
async def get_retraining_status():
    """Get current retraining status."""
    return retraining_status


@app.get("/metrics")
async def get_metrics():
    """
    Get Prometheus-style metrics for monitoring.
    """
    uptime = (datetime.now() - model_start_time).total_seconds()
    
    metrics = f"""# HELP model_uptime_seconds Model uptime in seconds
# TYPE model_uptime_seconds gauge
model_uptime_seconds {uptime}

# HELP prediction_count_total Total number of predictions made
# TYPE prediction_count_total counter
prediction_count_total {prediction_count}

# HELP model_status Model operational status (1=operational, 0=down)
# TYPE model_status gauge
model_status {1 if predictor and predictor.model else 0}

# HELP retraining_status Retraining status (1=retraining, 0=idle)
# TYPE retraining_status gauge
retraining_status {1 if retraining_status.get("is_retraining", False) else 0}
"""
    return HTMLResponse(content=metrics)


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
