import os
from ultralytics import YOLO

def inspect():
    model_path = "weights/temp_jarvanlee.pt"
    if not os.path.exists(model_path):
        print(f"[-] Model file {model_path} not found.")
        return
        
    print("[*] Loading jarvanlee helmet model...")
    model = YOLO(model_path)
    print(f"[+] Model loaded successfully. Architecture: YOLOv8-{model.ckpt.get('type', 'medium')}")
    print("[+] Detected Class names:")
    print(model.names)

if __name__ == "__main__":
    inspect()
