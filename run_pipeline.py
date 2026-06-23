import os
import json
from src.pipeline.inference import TrafficViolationPipeline

def run_test():
    print("=== Traffic Violation Pipeline Test Harness ===")
    
    # Check that weights exist
    weights_check = ["weights/license_plate_yolov8.pt", "weights/helmet_yolov8.pt"]
    for path in weights_check:
        if not os.path.exists(path):
            print(f"[-] Missing weights file: {path}. Please run download_resources.py first.")
            return

    # Initialize Pipeline
    pipeline = TrafficViolationPipeline(weights_dir="weights")
    
    samples = [
        "data/raw/traffic_sample_1.jpg",
        "data/raw/traffic_sample_2.jpg",
        "data/raw/traffic_sample_3.png"
    ]
    
    for sample in samples:
        if not os.path.exists(sample):
            print(f"[-] Missing sample: {sample}. Skipping.")
            continue
            
        print(f"\n[+] Running inference on: {sample}")
        try:
            metadata = pipeline.process_image(sample, output_dir="data/processed")
            print(f"[+] Detections Found: {len(metadata['detections'])}")
            print(f"[+] Violations Flagged: {len(metadata['violations'])}")
            print(json.dumps(metadata['violations'], indent=2))
        except Exception as e:
            print(f"[-] Error processing {sample}: {e}")

    print("\n=== Test Finished. Processed images and metadata are stored in data/processed/ ===")

if __name__ == "__main__":
    run_test()
