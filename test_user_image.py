import os
import shutil
import json
from ultralytics import YOLO
from src.pipeline.inference import TrafficViolationPipeline

def run_user_test():
    source_path = r"C:\Users\Manoj.M\AppData\Local\Temp\gradio\63d3f758116d7a6b84e990e17a2c776f376f1a0a7eb60ddd8c0ab3ad6e9475aa\TripleRiding.jpeg"
    target_path = "data/raw/traffic_sample_4.jpg"
    
    if not os.path.exists(source_path):
        print(f"[-] Source image not found at {source_path}")
        return
        
    # Copy file to data/raw/
    os.makedirs("data/raw", exist_ok=True)
    shutil.copy(source_path, target_path)
    print(f"[+] Copied user image to: {target_path}")

    # Initialize pipeline
    pipeline = TrafficViolationPipeline(weights_dir="weights")
    
    print("\n==========================================")
    print("RUNNING WITH MODEL 1: iam-tsr (YOLOv8-Nano)")
    print("==========================================")
    pipeline.helmet_model = YOLO("weights/helmet_yolov8.pt")
    
    # Reset processed output
    res_nano = pipeline.process_image(target_path, output_dir="data/processed")
    # Rename annotated file
    shutil.copy("data/processed/processed_traffic_sample_4.jpg", "data/processed/processed_sample_4_nano.jpg")
    print(f"[+] Nano Model detections count: {len(res_nano['detections'])}")
    print(f"[+] Nano Model violations count: {len(res_nano['violations'])}")
    print(json.dumps(res_nano["violations"], indent=2))
    
    print("\n=============================================")
    print("RUNNING WITH MODEL 2: jarvanlee (YOLOv8-Medium)")
    print("=============================================")
    pipeline.helmet_model = YOLO("weights/motorcycle_helmet_yolov8.pt")
    
    res_medium = pipeline.process_image(target_path, output_dir="data/processed")
    # Rename annotated file
    shutil.copy("data/processed/processed_traffic_sample_4.jpg", "data/processed/processed_sample_4_medium.jpg")
    print(f"[+] Medium Model detections count: {len(res_medium['detections'])}")
    print(f"[+] Medium Model violations count: {len(res_medium['violations'])}")
    print(json.dumps(res_medium["violations"], indent=2))

    print("\n[+] Verification completed. Both processed outputs saved in data/processed/")

if __name__ == "__main__":
    run_user_test()
