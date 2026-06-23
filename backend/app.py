import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional

# Import our pipeline
from src.pipeline.inference import TrafficViolationPipeline

app = FastAPI(title="Traffic Violation Detection System API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the ML Pipeline
# (We pass weights_dir="weights" where we downloaded the models)
pipeline = TrafficViolationPipeline(weights_dir="weights")

# Ensure required directories exist
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/violations", exist_ok=True)

# Serve the processed images statically so the frontend can display them
app.mount("/static", StaticFiles(directory="data/processed"), name="static")

# SQLite or JSON DB mock for history
VIOLATIONS_DB = []
db_file = "data/violations/violations_db.json"

def load_db():
    global VIOLATIONS_DB
    if os.path.exists(db_file):
        try:
            with open(db_file, "r") as f:
                VIOLATIONS_DB = json.load(f)
        except Exception:
            VIOLATIONS_DB = []

import json

def save_to_db(violation_entry):
    load_db()
    VIOLATIONS_DB.append(violation_entry)
    with open(db_file, "w") as f:
        json.dump(VIOLATIONS_DB, f, indent=4)

@app.get("/")
def read_root():
    return {"status": "healthy", "message": "Traffic Violation Detection API is running"}

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    # Validate extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        raise HTTPException(status_code=400, detail="Only JPG, JPEG, and PNG images are supported.")
    
    # Save the file to raw/
    raw_path = os.path.join("data/raw", file.filename)
    with open(raw_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Run inference through the pipeline
        result = pipeline.process_image(raw_path, output_dir="data/processed")
        
        # Format the static URL for the output image
        processed_filename = os.path.basename(result["processed_image"])
        static_url = f"/static/{processed_filename}"
        
        # Build response payload
        response = {
            "filename": file.filename,
            "imageUrl": static_url,
            "detections": result["detections"],
            "violations": result["violations"]
        }
        
        # Save violations to mock DB with timestamps
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for viol in result["violations"]:
            # Find associated license plate for this violation if available
            plate_text = "N/A"
            for det in result["detections"]:
                if det["type"] == "license_plate":
                    plate_text = det["text"]
                    break
            
            entry = {
                "timestamp": timestamp,
                "type": viol["type"],
                "details": viol["details"],
                "plate": plate_text,
                "imageUrl": static_url
            }
            save_to_db(entry)
            
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/history")
def get_history():
    load_db()
    return VIOLATIONS_DB

@app.get("/analytics")
def get_analytics():
    load_db()
    # Simple aggregation for analytics charts
    types_count = {}
    hourly_trends = {} # Mock / Grouped
    
    for v in VIOLATIONS_DB:
        t = v["type"]
        types_count[t] = types_count.get(t, 0) + 1
        
        # Extract date/hour
        hour = v["timestamp"].split(" ")[1].split(":")[0] + ":00"
        hourly_trends[hour] = hourly_trends.get(hour, 0) + 1
        
    return {
        "violation_types": [{"name": k, "value": v} for k, v in types_count.items()],
        "hourly_trends": [{"time": k, "count": v} for k, v in sorted(hourly_trends.items())]
    }
