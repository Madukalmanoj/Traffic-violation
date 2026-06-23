import os
from ultralytics import YOLO

def compare():
    model_n_path = "weights/helmet_yolov8.pt"
    model_m_path = "weights/temp_jarvanlee.pt"
    
    if not os.path.exists(model_n_path) or not os.path.exists(model_m_path):
        print("[-] Missing one of the model files. Please wait for download to finish.")
        return
        
    print("[*] Loading iam-tsr (YOLOv8-Nano)...")
    model_n = YOLO(model_n_path)
    
    print("[*] Loading jarvanlee (YOLOv8-Medium)...")
    model_m = YOLO(model_m_path)
    
    # Run on sample 3
    test_image = "data/raw/traffic_sample_3.png"
    if not os.path.exists(test_image):
        print(f"[-] Missing test image: {test_image}")
        return
        
    print(f"\n[*] Running comparison on motorcycle helmet image: {test_image}")
    
    print("\n--- Model 1: iam-tsr (YOLOv8-Nano) ---")
    res_n = model_n(test_image, verbose=False)[0]
    for box in res_n.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"Class: {model_n.names[cls_id]} | Confidence: {conf*100:.1f}% | Box: {box.xyxy[0].cpu().numpy().tolist()}")
        
    print("\n--- Model 2: jarvanlee (YOLOv8-Medium) ---")
    res_m = model_m(test_image, verbose=False)[0]
    for box in res_m.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"Class: {model_m.names[cls_id]} | Confidence: {conf*100:.1f}% | Box: {box.xyxy[0].cpu().numpy().tolist()}")

if __name__ == "__main__":
    compare()
