import os
import urllib.request
import sys

def create_folders():
    folders = [
        "data",
        "data/raw",
        "data/processed",
        "data/violations",
        "weights",
        "src",
        "src/pipeline",
        "backend",
        "frontend"
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"[+] Created directory: {folder}")

def download_file(url, destination):
    print(f"[*] Downloading {url} to {destination}...")
    try:
        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\rProgress: {percent}%")
            sys.stdout.flush()
        
        urllib.request.urlretrieve(url, destination, reporthook=progress_hook)
        print(f"\n[+] Successfully downloaded: {destination}")
    except Exception as e:
        print(f"\n[-] Failed to download from {url}. Error: {e}")

def main():
    print("=== Traffic Violation System: Workspace Initializer ===")
    create_folders()
    
    # Pre-trained weights for the hierarchical pipeline
    resources = {
        "license_plate_yolov8.pt": "https://huggingface.co/yasirfaizahmed/license-plate-object-detection/resolve/main/best.pt",
        "helmet_yolov8.pt": "https://huggingface.co/iam-tsr/yolov8n-helmet-detection/resolve/main/best.pt",
        "motorcycle_helmet_yolov8.pt": "https://huggingface.co/JarvanLee/yolov8-helmet-violation-detection/resolve/main/weights/best.pt"
    }
    
    for filename, url in resources.items():
        destination = os.path.join("weights", filename)
        if not os.path.exists(destination):
            download_file(url, destination)
        else:
            print(f"[~] {filename} already exists in weights/. Skipping download.")

    print("\n=== Initial Setup Complete ===")
    print("Next steps:")
    print("1. Place your sample traffic video/images in data/raw/")
    print("2. Check datasets_info.md to download raw datasets for custom fine-tuning if desired.")

if __name__ == "__main__":
    main()
