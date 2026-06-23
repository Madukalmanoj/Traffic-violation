import os
import urllib.request

def download_samples():
    os.makedirs("data/raw", exist_ok=True)
    
    samples = {
        "traffic_sample_1.jpg": "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg",
        "traffic_sample_2.jpg": "https://images.unsplash.com/photo-1558981806-ec527fa84c39?w=800", # Motorcycle image
        "traffic_sample_3.png": "https://raw.githubusercontent.com/rishiraj/Helmet-Detector/master/examples/example_01.png", # Helmet detector sample image
    }
    
    for filename, url in samples.items():
        destination = os.path.join("data/raw", filename)
        if not os.path.exists(destination):
            print(f"[*] Downloading sample: {filename} from {url}...")
            try:
                urllib.request.urlretrieve(url, destination)
                print(f"[+] Downloaded: {destination}")
            except Exception as e:
                print(f"[-] Failed to download {filename}: {e}")
        else:
            print(f"[~] {filename} already exists. Skipping.")

if __name__ == "__main__":
    download_samples()
