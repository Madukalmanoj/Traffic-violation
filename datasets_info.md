# Datasets for Traffic Violation Detection System

If you choose to fine-tune your own models on your RTX 4050 (6GB VRAM) instead of using the pre-trained weights downloaded by `download_resources.py`, here are the top curated datasets on **Roboflow Universe** and how to download them.

---

## 1. Helmet Detection Dataset
* **Name on Roboflow:** `yolov8-helmet` (by workspace `yolov8-mb4rn`)
* **Classes:** `helmet`, `no-helmet`
* **Size:** ~1,000 traffic images from varied angles and lighting conditions.
* **Why it's the best:** Pre-split into train/valid/test sets and pre-labeled in YOLOv8 text format.
* **Python Download Script:**
  ```python
  # pip install roboflow
  from roboflow import Roboflow
  rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
  project = rf.workspace("yolov8-mb4rn").project("yolov8-helmet")
  version = project.version(1)
  dataset = version.download("yolov8")
  ```

---

## 2. License Plate Detection Dataset
* **Name on Roboflow:** `YOLOv8 Number Plate Detection` (by workspace `ML`)
* **Classes:** `license-plate`
* **Size:** 5,750+ images of cars, motorcycles, and trucks with bounding boxes for plates.
* **Why it's the best:** Includes standard and skewed plate angles, great for testing OCR robustness.
* **Python Download Script:**
  ```python
  # pip install roboflow
  from roboflow import Roboflow
  rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
  project = rf.workspace("ml-drcsw").project("yolov8-number-plate-detection")
  version = project.version(2)
  dataset = version.download("yolov8")
  ```

---

## 3. Seatbelt Detection Dataset
* **Name on Roboflow:** `seatbelt-detection-yolov8` (by workspace `yolov8-seatbelt`)
* **Classes:** `seatbelt`, `no-seatbelt`
* **Size:** ~2,000 images inside the car cabin / windshield views.
* **Why it's the best:** Essential for classifying front seatbelt compliance.
* **Python Download Script:**
  ```python
  # pip install roboflow
  from roboflow import Roboflow
  rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
  project = rf.workspace("yolov8-seatbelt").project("seatbelt-detection-yolov8")
  version = project.version(1)
  dataset = version.download("yolov8")
  ```

---

## Training Command Cheat-Sheet
Once downloaded, place the dataset folder inside your workspace and run training from your shell:

```bash
# Fine-tune Helmet Detector (starts from COCO weights, trains in ~15 mins on RTX 4050)
yolo task=detect mode=train model=yolov8n.pt data=path/to/helmet/data.yaml epochs=30 imgsz=640 batch=32 device=0 amp=True

# Fine-tune License Plate Detector
yolo task=detect mode=train model=yolov8n.pt data=path/to/license_plate/data.yaml epochs=30 imgsz=640 batch=32 device=0 amp=True
```
