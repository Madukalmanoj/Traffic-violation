import os
import shutil
import json
from ultralytics import YOLO
from src.pipeline.inference import TrafficViolationPipeline

def run_avif_test():
    target_path = "data/raw/traffic_sample_5.avif"
    
    if not os.path.exists(target_path):
        print(f"[-] AVIF image not found at {target_path}")
        return

    # Initialize pipeline
    pipeline = TrafficViolationPipeline(weights_dir="weights")
    
    print("\n==========================================")
    print("RUNNING NANO MODEL (iam-tsr)")
    print("==========================================")
    pipeline.helmet_model = YOLO("weights/helmet_yolov8.pt")
    
    # Process
    res_nano = pipeline.process_image(target_path, output_dir="data/processed")
    # Rename output to a distinct name
    shutil.copy("data/processed/processed_traffic_sample_5.jpg", "data/processed/processed_sample_5_nano.jpg")
    
    print(f"[+] Nano Model detections count: {len(res_nano['detections'])}")
    print(f"[+] Nano Model violations count: {len(res_nano['violations'])}")
    print(json.dumps(res_nano["violations"], indent=2))
    
    print("\n=============================================")
    print("RUNNING MEDIUM MODEL (jarvanlee)")
    print("=============================================")
    pipeline.helmet_model = YOLO("weights/motorcycle_helmet_yolov8.pt")
    
    res_medium = pipeline.process_image(target_path, output_dir="data/processed")
    shutil.copy("data/processed/processed_traffic_sample_5.jpg", "data/processed/processed_sample_5_medium.jpg")
        
    print(f"[+] Medium Model detections count: {len(res_medium['detections'])}")
    print(f"[+] Medium Model violations count: {len(res_medium['violations'])}")
    print(json.dumps(res_medium["violations"], indent=2))

    print("\n[+] Done.")

if __name__ == "__main__":
    run_avif_test()
