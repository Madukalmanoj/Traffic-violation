import os
import urllib.request
from ultralytics import YOLO

def download_alternative_model():
    os.makedirs("weights", exist_ok=True)
    destination = "weights/safety_helmet_yolov8.pt"
    url = "https://huggingface.co/sharathhhhh/safetyHelmet-detection-yolov8/resolve/main/best.pt"
    
    if not os.path.exists(destination):
        print(f"[*] Downloading alternative helmet model from {url}...")
        try:
            urllib.request.urlretrieve(url, destination)
            print(f"[+] Downloaded: {destination}")
        except Exception as e:
            print(f"[-] Failed to download: {e}")
            return False
    else:
        print(f"[~] {destination} already exists. Skipping download.")
    return True

def run_comparison():
    if not download_alternative_model():
        return
        
    print("[*] Loading models...")
    # Load current model
    current_model = YOLO("weights/helmet_yolov8.pt")
    # Load alternative model
    alternative_model = YOLO("weights/safety_helmet_yolov8.pt")
    
    test_image = "data/raw/traffic_sample_3.png"
    if not os.path.exists(test_image):
        print(f"[-] Test image {test_image} missing. Please run download_samples.py first.")
        return
        
    print(f"\n[*] Comparing detection results on: {test_image}")
    
    print("\n--- Current Model (iam-tsr/yolov8n-helmet-detection) ---")
    res1 = current_model(test_image, verbose=False)[0]
    for box in res1.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"Detected Class: {res1.names[cls_id]} | Confidence: {conf:.2f} | BBox: {box.xyxy[0].cpu().numpy().tolist()}")
        
    print("\n--- Alternative Model (sharathhhhh/safetyHelmet-detection-yolov8) ---")
    res2 = alternative_model(test_image, verbose=False)[0]
    for box in res2.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"Detected Class: {res2.names[cls_id]} | Confidence: {conf:.2f} | BBox: {box.xyxy[0].cpu().numpy().tolist()}")

if __name__ == "__main__":
    run_comparison()
