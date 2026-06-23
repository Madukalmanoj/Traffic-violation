import os
import cv2
import json
import torch
import shutil
import subprocess
import numpy as np
from ultralytics import YOLO

def point_in_zone(point, polygon_points):
    """Runtime check: is a tracked vehicle's reference point inside/on a
    calibrated zone?
    point: (x, y) - e.g. the bottom-center of a vehicle's bounding box
    polygon_points: list of (x, y) from calibration.json -> zones[name]
    """
    if not polygon_points or len(polygon_points) < 3:
        return False
    poly = np.array(polygon_points, dtype=np.int32)
    return cv2.pointPolygonTest(poly, point, False) >= 0

class DetectedLight:
    def __init__(self, bbox, confidence, state):
        self.bbox = bbox # [x1, y1, x2, y2]
        self.confidence = confidence
        self.state = state # 'green', 'off', 'red', 'yellow'

class VehicleTracker:
    def __init__(self, max_age=5, traffic_direction="towards"):
        self.max_age = max_age
        self.traffic_direction = traffic_direction
        self.next_id = 1
        self.tracks = {}

    def update(self, current_dets, intersection_state, x_min, y_min, x_max, y_max, w, h, calibration=None):
        vehicle_types = ["car", "bus", "truck", "motorcycle", "vehicle"]
        current_vehicles = [d for d in current_dets if d["type"] in vehicle_types]
        
        matched_det_indices = set()
        matched_track_ids = set()
        updated_tracks = {}

        has_calib = calibration is not None and "zones" in calibration
        stop_line_pts = calibration["zones"].get("stop_line", []) if has_calib else []
        exit_line_pts = calibration["zones"].get("exit_line", []) if has_calib else []
        
        use_poly_stop = len(stop_line_pts) >= 3
        use_poly_exit = len(exit_line_pts) >= 3

        # Match existing tracks
        for track_id, track in self.tracks.items():
            best_iou = 0
            best_det_idx = -1
            
            for idx, det in enumerate(current_vehicles):
                if idx in matched_det_indices:
                    continue
                iou = compute_iou(track["bbox"], det["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = idx
            
            if best_iou >= 0.20 and best_det_idx != -1:
                det = current_vehicles[best_det_idx]
                matched_det_indices.add(best_det_idx)
                matched_track_ids.add(track_id)
                
                x1, y1, x2, y2 = det["bbox"]
                center_x = (x1 + x2) // 2
                ref_pt = (center_x, y2)
                
                if use_poly_stop:
                    in_stop_line = point_in_zone(ref_pt, stop_line_pts)
                elif not has_calib:
                    line_y = y_max if self.traffic_direction == "away" else y_min
                    in_stop_line = (y2 <= line_y if self.traffic_direction == "away" else y2 >= line_y) and (x_min <= center_x <= x_max)
                else:
                    in_stop_line = False
                
                if use_poly_exit:
                    in_exit_line = point_in_zone(ref_pt, exit_line_pts)
                else:
                    in_exit_line = False
                
                crossed_stop_line = track.get("crossed_stop_line", False)
                crossed_exit_line = track.get("crossed_exit_line", False)
                crossed_legally = track.get("crossed_legally", False)
                stop_line_violation = track.get("stop_line_violation", False)
                red_light_violation = track.get("red_light_violation", False)

                if in_stop_line and not crossed_stop_line:
                    crossed_stop_line = True
                    if intersection_state == "red":
                        if not crossed_legally:
                            if use_poly_exit:
                                stop_line_violation = True
                            else:
                                red_light_violation = True
                    else:
                        crossed_legally = True

                if in_exit_line and not crossed_exit_line:
                    crossed_exit_line = True
                    if intersection_state == "red":
                        if not crossed_legally:
                            red_light_violation = True
                            stop_line_violation = False
                    else:
                        crossed_legally = True

                if crossed_legally:
                    stop_line_violation = False
                    red_light_violation = False

                updated_tracks[track_id] = {
                    "bbox": det["bbox"],
                    "type": det["type"],
                    "age": 0,
                    "crossed_stop_line": crossed_stop_line,
                    "crossed_exit_line": crossed_exit_line,
                    "crossed_legally": crossed_legally,
                    "stop_line_violation": stop_line_violation,
                    "red_light_violation": red_light_violation
                }
            else:
                track["age"] += 1
                if track["age"] <= self.max_age:
                    updated_tracks[track_id] = track
        
        # New tracks
        for idx, det in enumerate(current_vehicles):
            if idx in matched_det_indices:
                continue
            
            x1, y1, x2, y2 = det["bbox"]
            center_x = (x1 + x2) // 2
            ref_pt = (center_x, y2)
            
            if use_poly_stop:
                in_stop_line = point_in_zone(ref_pt, stop_line_pts)
            elif not has_calib:
                line_y = y_max if self.traffic_direction == "away" else y_min
                in_stop_line = (y2 <= line_y if self.traffic_direction == "away" else y2 >= line_y) and (x_min <= center_x <= x_max)
            else:
                in_stop_line = False
            
            if use_poly_exit:
                in_exit_line = point_in_zone(ref_pt, exit_line_pts)
            else:
                in_exit_line = False
                
            crossed_legally = in_stop_line or in_exit_line
            
            stop_line_violation = False
            red_light_violation = False
            
            if in_stop_line and not crossed_legally and intersection_state == "red":
                if use_poly_exit:
                    stop_line_violation = True
                else:
                    red_light_violation = True
                    
            if in_exit_line and not crossed_legally and intersection_state == "red":
                red_light_violation = True
                
            updated_tracks[self.next_id] = {
                "bbox": det["bbox"],
                "type": det["type"],
                "age": 0,
                "crossed_stop_line": in_stop_line,
                "crossed_exit_line": in_exit_line,
                "crossed_legally": crossed_legally,
                "stop_line_violation": stop_line_violation,
                "red_light_violation": red_light_violation
            }
            self.next_id += 1
            
        self.tracks = updated_tracks
        
        results = []
        for track_id, track in self.tracks.items():
            if track["age"] == 0:
                results.append({
                    "track_id": track_id,
                    "type": track["type"],
                    "bbox": track["bbox"],
                    "stop_line_violation": track["stop_line_violation"],
                    "red_light_violation": track["red_light_violation"]
                })
        return results

# Try to import EasyOCR for License Plate reading; degrade gracefully if not installed.
try:
    import easyocr
    READER = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    print("[+] EasyOCR loaded successfully.")
except ImportError:
    READER = None
    print("[!] EasyOCR not installed. OCR text extraction will be bypassed (mock plate returned).")

def draw_custom_annotation(img, x1, y1, x2, y2, color, label_str):
    h, w = img.shape[:2]
    # 1. Bounding Box Style: Fill Style (15% opacity)
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
    
    # Border Thickness: 1 pixel
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    
    # 2. Dynamic Label Scaling
    box_w = x2 - x1
    box_h = y2 - y1
    max_edge = max(box_w, box_h)
    if max_edge < 80:
        font_scale = 0.16
    elif max_edge < 150:
        font_scale = 0.18
    else:
        font_scale = 0.22
        
    # 3. Label Background Box & Standard/Overflow Placement
    (label_width, label_height), baseline = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    
    # Top Edge Overflow Prevention: y1 < 70px
    if y1 < 70:
        bg_y1 = y1
        bg_y2 = y1 + label_height + baseline + 6
        text_y = y1 + label_height + 2
    else:
        bg_y1 = y1 - label_height - baseline - 6
        bg_y2 = y1
        text_y = y1 - baseline - 2
        
    # Clamp coordinates to image boundaries
    bg_x1 = max(0, x1)
    bg_x2 = min(w, x1 + label_width + 6)
    bg_y1 = max(0, bg_y1)
    bg_y2 = min(h, bg_y2)
    
    # Draw background box (solid color matching category color)
    cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
    
    # Draw text (Pure White)
    cv2.putText(img, label_str, (x1 + 3, int(text_y)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, lineType=cv2.LINE_AA)

def compute_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection coordinates
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    # Calculate union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0
    return inter_area / union_area

class TrafficViolationPipeline:
    def __init__(self, weights_dir="weights"):
        # Load Global Model (uses YOLOv8n pretrained on COCO dataset)
        print("[*] Loading Global YOLOv8 Model (COCO)...")
        self.global_model = YOLO("yolov8n.pt")
        
        # Load custom Helmet Detector
        default_helmet_path = os.path.join(weights_dir, "helmet_yolov8.pt")
        custom_helmet_path = r"D:\Hackathons\flipkartTraffic\models\helmet_detector.pt"
        if os.path.exists(custom_helmet_path):
            helmet_path = custom_helmet_path
            print(f"[*] Custom helmet model detected at {custom_helmet_path}. Using it for testing.")
        else:
            helmet_path = default_helmet_path
        print(f"[*] Loading Helmet Model from {helmet_path}...")
        self.helmet_model = YOLO(helmet_path)
        
        # Load custom License Plate Detector
        plate_path = os.path.join(weights_dir, "license_plate_yolov8.pt")
        print(f"[*] Loading License Plate Model from {plate_path}...")
        self.plate_model = YOLO(plate_path)

        # Load custom Traffic Light Detector
        traffic_light_path = os.path.join(weights_dir, "traffic_light_yolov8.pt")
        print(f"[*] Loading Traffic Light Model from {traffic_light_path}...")
        self.traffic_light_model = YOLO(traffic_light_path)
        
        # COCO class indexes we care about
        # 0: person, 2: car, 3: motorcycle, 5: bus, 7: truck
        self.vehicle_classes = [2, 3, 5, 7]
        self.motorcycle_class = 3
        self.person_class = 0
        
        self.calibration = None
        self.load_calibration()

    def load_calibration(self, path="calibration.json"):
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    self.calibration = json.load(f)
                if "zones" not in self.calibration:
                    self.calibration = None
                else:
                    print(f"[+] Loaded calibration.json from {path}")
            except Exception as e:
                print(f"[-] Failed to load calibration: {e}")
                self.calibration = None
        else:
            self.calibration = None

    @staticmethod
    def auto_detect_stop_line(image: np.ndarray):
        """Detect the white stop line / zebra crossing Y position from road markings."""
        try:
            h, w = image.shape[:2]

            # Focus on the lower road region (40%-85% height) to exclude background structures
            roi_top = int(h * 0.40)
            roi_bottom = int(h * 0.85)
            roi = image[roi_top:roi_bottom, :]
            roi_h, roi_w = roi.shape[:2]

            # HSV white mask: low saturation, high value (road paint)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            white_mask = cv2.inRange(hsv, (0, 0, 170), (180, 50, 255))

            # Also capture slightly weathered / grey-white paint
            gray_white = cv2.inRange(hsv, (0, 0, 140), (180, 80, 255))
            
            # Yellow mask: hue range [15, 35], sat [80, 255], val [100, 255]
            yellow_mask = cv2.inRange(hsv, (15, 80, 100), (35, 255, 255))
            
            combined = cv2.bitwise_or(white_mask, gray_white)
            combined = cv2.bitwise_or(combined, yellow_mask)

            # Morphological: keep horizontal structures, remove vertical ones
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN,
                                       cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
            connected = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_h)

            # Remove tall vertical features (poles, signs, buildings)
            tall_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
            vertical_only = cv2.morphologyEx(connected, cv2.MORPH_OPEN, tall_v)
            horizontal_only = cv2.subtract(connected, vertical_only)

            # Horizontal projection: white pixels per row
            row_sums = np.sum(horizontal_only, axis=1).astype(float) / 255.0

            # Smooth profile
            kernel_size = 5
            smoothed = np.convolve(row_sums, np.ones(kernel_size) / kernel_size, mode='same')

            # Threshold: at least 8% of frame width should be white
            min_white = roi_w * 0.08
            candidates = np.where(smoothed > min_white)[0]

            if len(candidates) == 0:
                print("auto_detect_stop_line: no white bands found")
                return None

            # Group consecutive rows into bands
            bands = []
            current = [candidates[0]]
            for i in range(1, len(candidates)):
                if candidates[i] - candidates[i - 1] <= 5:
                    current.append(candidates[i])
                else:
                    bands.append(current)
                    current = [candidates[i]]
            bands.append(current)

            # Score each band
            best_y = None
            best_score = -1.0
            for band in bands:
                band_y = float(np.mean(band)) + roi_top
                band_width = float(max(smoothed[r] for r in band))
                thickness = len(band)

                if thickness < 2 or band_width < roi_w * 0.08:
                    continue

                # Prefer wider, thicker bands closer to the centre of the ROI
                rel_y = (band_y - roi_top) / max(1, roi_bottom - roi_top)
                position_score = 1.0 - abs(rel_y - 0.5) * 0.5
                width_score = band_width / roi_w
                thickness_score = min(thickness / 10.0, 1.0)
                score = width_score * 0.5 + thickness_score * 0.3 + position_score * 0.2

                if score > best_score:
                    best_score = score
                    best_y = int(band_y)

            if best_y is not None:
                print(f"[+] Auto-detected stop line at y={best_y} (score={best_score:.2f})")
            else:
                print("auto_detect_stop_line: no valid bands after filtering")

            return best_y

        except Exception as exc:
            print(f"[-] auto_detect_stop_line failed: {exc}")
            return None

    def estimate_stop_line_from_tls(self, tl_results, img_height):
        """Estimate stop line Y coordinate from detected traffic lights using perspective geometry."""
        if not tl_results:
            return None
        try:
            best_tl = max(tl_results, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            tx1, ty1, tx2, ty2 = best_tl.bbox
            th = ty2 - ty1
            if th <= 0:
                return None
            
            estimated_y = ty2 + int(0.55 * (img_height - ty2))
            clamped_y = max(int(img_height * 0.3), min(estimated_y, int(img_height * 0.85)))
            print(f"[+] Traffic light fallback: estimated stop line at y={clamped_y} using TL box {best_tl.bbox} (unclamped={estimated_y})")
            return clamped_y
        except Exception as exc:
            print(f"[-] Failed to estimate stop line from traffic lights: {exc}")
            return None

    def resolve_stop_line(self, image, tl_results):
        """Resolve the stop line Y coordinates (y_left, y_right), pole_x, and is_physical flag,
        automatically applying traffic light fallback and perspective calculation.
        """
        try:
            h, w = image.shape[:2]
            auto_stop_y = self.auto_detect_stop_line(image)
            fallback_y = self.estimate_stop_line_from_tls(tl_results, h)
            
            pole_x = None
            if tl_results:
                best_tl = max(tl_results, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                tx1, ty1, tx2, ty2 = best_tl.bbox
                pole_x = (tx1 + tx2) // 2

            if auto_stop_y is not None and fallback_y is not None:
                limit = fallback_y - int(h * 0.18)
                if auto_stop_y < limit:
                    print(f"[*] Rejection rule: auto_stop_y ({auto_stop_y}) rejected because it's significantly above ground level fallback_y ({fallback_y}). Using fallback_y instead.")
                    auto_stop_y = None

            is_physical = False
            base_y = None
            if auto_stop_y is not None:
                base_y = auto_stop_y
                is_physical = True
            elif fallback_y is not None:
                base_y = fallback_y
                is_physical = False

            if base_y is not None and pole_x is not None:
                slope_offset = int(0.08 * (h - base_y))
                if pole_x < w // 2:
                    y_left = base_y
                    y_right = base_y + slope_offset
                else:
                    y_left = base_y + slope_offset
                    y_right = base_y
                
                y_left = max(0, min(y_left, h - 1))
                y_right = max(0, min(y_right, h - 1))
                return (y_left, y_right, pole_x, is_physical)
            
            if auto_stop_y is not None:
                return (auto_stop_y, auto_stop_y, None, True)
                
            return None
        except Exception as exc:
            print(f"[-] Failed to resolve stop line: {exc}")
            return None

    def detect_zebra_crossing_box(self, img):
        h, w = img.shape[:2]
        try:
            # Focus on lower road region (40% to 90% height)
            roi_top = int(h * 0.40)
            roi_bottom = int(h * 0.90)
            roi = img[roi_top:roi_bottom, :]
            
            # HSV White/Yellow Mask
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            white_mask = cv2.inRange(hsv, (0, 0, 170), (180, 50, 255))
            gray_white = cv2.inRange(hsv, (0, 0, 140), (180, 80, 255))
            yellow_mask = cv2.inRange(hsv, (15, 80, 100), (35, 255, 255))
            
            combined = cv2.bitwise_or(white_mask, gray_white)
            combined = cv2.bitwise_or(combined, yellow_mask)
            
            # Open/Close morphology
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
            
            kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (85, 15))
            connected = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_connect)
            
            contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            candidate_boxes = []
            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                if cw > w * 0.05 and ch > 5:
                    candidate_boxes.append((x, y + roi_top, x + cw, y + roi_top + ch))
            
            if not candidate_boxes:
                return [0, int(h * 0.60), w, int(h * 0.80)]
                
            candidate_boxes = sorted(candidate_boxes, key=lambda b: b[1])
            x1_min = min(b[0] for b in candidate_boxes)
            y1_min = min(b[1] for b in candidate_boxes)
            x2_max = max(b[2] for b in candidate_boxes)
            y2_max = max(b[3] for b in candidate_boxes)
            
            # Ensure it is at least 25 pixels tall
            thickness = y2_max - y1_min
            if thickness < 25:
                y1_min = max(0, y1_min - 15)
                y2_max = min(h - 1, y2_max + 15)
                
            return [x1_min, y1_min, x2_max, y2_max]
        except Exception as e:
            print(f"[-] detect_zebra_crossing_box failed: {e}")
            return [0, int(h * 0.60), w, int(h * 0.80)]

    def process_image(self, image_path, output_dir="data/processed", tracker=None, custom_line=None, traffic_direction="towards", use_calibration=True):
        self.load_calibration()
        os.makedirs(output_dir, exist_ok=True)
        from PIL import Image
        import numpy as np
        try:
            pil_img = Image.open(image_path).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise ValueError(f"Could not read image {image_path}. Error: {e}")
            
        h, w, _ = img.shape
        print(f"[*] Processing image: {image_path} ({w}x{h})")

        # Determine dynamic font scale and box thickness
        # Scale to ensure shapes and text remain highly visible but clean
        box_thickness = max(2, int(w / 600.0))
        font_scale = max(0.38, min(0.65, w / 1200.0))
        h_font_scale = max(0.28, font_scale * 0.8)
        text_thickness = max(1, int(w / 900.0))

        # Step 0: Traffic Light Detection & Stop Line Resolution
        tl_results = []
        intersection_state = "off"
        detections = []
        violations = []
        
        has_calib = use_calibration and self.calibration is not None and "zones" in self.calibration
        signal_roi_pts = self.calibration["zones"].get("signal_roi", []) if has_calib else []
        use_signal_roi = len(signal_roi_pts) >= 2
        
        roi_offset_x = 0
        roi_offset_y = 0
        
        if use_signal_roi:
            pts = np.array(signal_roi_pts, dtype=np.int32)
            roi_x, roi_y, roi_w, roi_h = cv2.boundingRect(pts)
            pad = 15
            roi_x1 = max(0, roi_x - pad)
            roi_y1 = max(0, roi_y - pad)
            roi_x2 = min(w, roi_x + roi_w + pad)
            roi_y2 = min(h, roi_y + roi_h + pad)
            
            tl_img = img[roi_y1:roi_y2, roi_x1:roi_x2]
            roi_offset_x = roi_x1
            roi_offset_y = roi_y1
            print(f"[*] signal_roi calibration active. Running traffic light model on crop: [{roi_x1}, {roi_y1}, {roi_x2}, {roi_y2}]")
        else:
            tl_img = img
        
        try:
            tl_preds = self.traffic_light_model(tl_img, verbose=False)[0]
            for box in tl_preds.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.traffic_light_model.names.get(cls_id, "unknown").lower()
                conf = float(box.conf[0])
                tx1_crop, ty1_crop, tx2_crop, ty2_crop = map(int, box.xyxy[0].tolist())
                
                tx1 = tx1_crop + roi_offset_x
                ty1 = ty1_crop + roi_offset_y
                tx2 = tx2_crop + roi_offset_x
                ty2 = ty2_crop + roi_offset_y

                # Map class names/IDs to color states
                color_detected = "unknown"
                if cls_id == 0 or "green" in cls_name:
                    color_detected = "green"
                elif cls_id == 2 or "red" in cls_name:
                    color_detected = "red"
                elif cls_id == 3 or "yellow" in cls_name:
                    color_detected = "yellow"
                elif cls_id == 1 or "off" in cls_name:
                    color_detected = "off"

                if conf >= 0.35 and color_detected != "unknown":
                    tl_results.append(DetectedLight([tx1, ty1, tx2, ty2], conf, color_detected))
                    detections.append({
                        "type": f"traffic_light_{color_detected}",
                        "bbox": [tx1, ty1, tx2, ty2],
                        "confidence": conf
                    })
                    
                    # Annotate Traffic Light
                    tl_color = (128, 128, 128)
                    if color_detected == "red":
                        tl_color = (0, 0, 255)
                    elif color_detected == "yellow":
                        tl_color = (0, 255, 255)
                    elif color_detected == "green":
                        tl_color = (0, 255, 0)
                    cv2.rectangle(img, (tx1, ty1), (tx2, ty2), tl_color, box_thickness)
                    cv2.putText(img, f"TL: {color_detected.upper()} ({conf*100:.1f}%)", (tx1, max(10, ty1 - 3)),
                                cv2.FONT_HERSHEY_SIMPLEX, h_font_scale, tl_color, text_thickness)
        except Exception as e:
            print(f"[-] Traffic Light model prediction failed: {e}")

        if tl_results:
            best_tl = max(tl_results, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            intersection_state = best_tl.state

        # Resolve stop line zone bounding box
        if custom_line is not None:
            if len(custom_line) == 4:
                x1_pct, y1_pct, x2_pct, y2_pct = custom_line
                x_min = int(w * x1_pct)
                y_min = int(h * y1_pct)
                x_max = int(w * x2_pct)
                y_max = int(h * y2_pct)
            else:
                y_left_pct, y_right_pct = custom_line
                x_min = 0
                y_min = int(h * min(y_left_pct, y_right_pct))
                x_max = w
                y_max = int(h * max(y_left_pct, y_right_pct))
            pole_x = None
            is_physical = False
        else:
            # Auto detect zebra crossing zone bounding box
            x_min, y_min, x_max, y_max = self.detect_zebra_crossing_box(img)
            pole_x = None
            is_physical = True

        # Step 1: Run Global Bounding Box Detection
        global_results = self.global_model(img, verbose=False)[0]
        
        # Pre-filter vehicle detections with IoU-based NMS to remove overlapping/duplicate boxes
        raw_boxes = []
        for box in global_results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in self.vehicle_classes:
                continue
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            raw_boxes.append((cls_id, conf, xyxy))
            
        # Sort raw_boxes by confidence descending
        raw_boxes = sorted(raw_boxes, key=lambda x: x[1], reverse=True)
        
        filtered_boxes = []
        for r_box in raw_boxes:
            cls_id, conf, xyxy = r_box
            # Check overlap with already accepted boxes
            keep = True
            for _, _, f_xyxy in filtered_boxes:
                if compute_iou(xyxy, f_xyxy) > 0.45:
                    keep = False
                    break
            if keep:
                filtered_boxes.append(r_box)
        
        # Update tracker if provided
        tracked_vehicles = {}
        if tracker is not None:
            temp_dets = []
            for cls_id, conf, xyxy in filtered_boxes:
                label = global_results.names[cls_id].lower()
                temp_dets.append({"type": label, "bbox": xyxy.tolist()})
            tracker_calibration = self.calibration if use_calibration else None
            tracker_results = tracker.update(temp_dets, intersection_state, x_min, y_min, x_max, y_max, w, h, calibration=tracker_calibration)
            for tr in tracker_results:
                tracked_vehicles[tuple(tr["bbox"])] = tr
        
        # Find all motorcycles and vehicles in the filtered set
        for cls_id, conf, xyxy in filtered_boxes:
            x1, y1, x2, y2 = xyxy
            
            # 1. Handle Motorcycles (Helmet and Triple Riding Checks)
            if cls_id == self.motorcycle_class:
                detections.append({
                    "type": "motorcycle",
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": conf
                })
                
                # Expand motorcycle crop upwards to capture rider heads (~35% height extension)
                crop_y1 = max(0, y1 - int((y2 - y1) * 0.35))
                crop_y2 = y2
                crop_x1 = max(0, x1 - int((x2 - x1) * 0.1))
                crop_x2 = min(w, x2 + int((x2 - x1) * 0.1))
                
                moto_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
                
                # Run Helmet Detector on Crop
                helmet_results = self.helmet_model(moto_crop, verbose=False)[0]
                
                riders_count = 0
                no_helmet_count = 0
                rider_head_boxes = []
                
                for h_box in helmet_results.boxes:
                    h_cls = int(h_box.cls[0]) # Custom classes typically: 0: helmet, 1: no-helmet
                    h_conf = float(h_box.conf[0])
                    
                    # Track riders (both helmet & no_helmet represent a rider)
                    riders_count += 1
                    
                    # Assuming class 1 is no_helmet or according to model card mapping
                    cls_name = helmet_results.names[h_cls].lower()
                    
                    is_violation = "no" in cls_name or "without" in cls_name or h_cls == 1
                    if is_violation:
                        no_helmet_count += 1
                        
                    # Map cropped coordinates back to global coordinates
                    h_xyxy = h_box.xyxy[0].cpu().numpy().astype(int)
                    hx1, hy1, hx2, hy2 = h_xyxy
                    global_hx1 = crop_x1 + hx1
                    global_hy1 = crop_y1 + hy1
                    global_hx2 = crop_x1 + hx2
                    global_hy2 = crop_y1 + hy2
                    
                    rider_head_boxes.append((global_hx1, global_hy1, global_hx2, global_hy2))
                    
                    # Draw head bounding boxes and labels using custom styled annotations
                    color = (0, 0, 255) if is_violation else (180, 50, 180) # Red or Purple BGR
                    label_text = f"NO_HELMET {h_conf*100:.0f}%" if is_violation else f"HELMET {h_conf*100:.0f}%"
                    draw_custom_annotation(img, global_hx1, global_hy1, global_hx2, global_hy2, color, label_text)
                
                # Determine motorcycle label, color, and flag violations (drawn ONCE to avoid thickness overlap)
                moto_violations = []
                is_moto_violation = False
                is_red_light_violation = False
                
                # Check for Red Light Violation
                is_stop_line_violation = False
                tr_info = tracked_vehicles.get(tuple(xyxy.tolist())) if tracker is not None else None
                if tracker is not None:
                    if tr_info:
                        if tr_info["red_light_violation"]:
                            is_red_light_violation = True
                            is_moto_violation = True
                            moto_violations.append("RED LIGHT VIOLATION")
                            violations.append({
                                "type": "Red Light Violation",
                                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                "details": f"Motorcycle (Track ID: {tr_info['track_id']}) crossed exit line during RED signal"
                            })
                        elif tr_info["stop_line_violation"]:
                            is_moto_violation = True
                            moto_violations.append("STOP LINE VIOLATION")
                            violations.append({
                                "type": "Stop Line Violation",
                                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                "details": f"Motorcycle (Track ID: {tr_info['track_id']}) crossed stop line during RED signal"
                            })
                else:
                    if intersection_state == "red":
                        moto_center_x = (x1 + x2) // 2
                        ref_pt = (moto_center_x, y2)
                        
                        has_calib = use_calibration and self.calibration is not None and "zones" in self.calibration
                        stop_line_pts = self.calibration["zones"].get("stop_line", []) if has_calib else []
                        exit_line_pts = self.calibration["zones"].get("exit_line", []) if has_calib else []
                        
                        use_poly_stop = len(stop_line_pts) >= 3
                        use_poly_exit = len(exit_line_pts) >= 3
                        
                        if use_poly_stop:
                            in_stop_line = point_in_zone(ref_pt, stop_line_pts)
                        elif not has_calib:
                            line_y = y_max if traffic_direction == "away" else y_min
                            in_stop_line = (y2 <= line_y if traffic_direction == "away" else y2 >= line_y) and (x_min <= moto_center_x <= x_max)
                        else:
                            in_stop_line = False
                            
                        if use_poly_exit:
                            in_exit_line = point_in_zone(ref_pt, exit_line_pts)
                        else:
                            in_exit_line = False
                            
                        is_stop_line_viol = False
                        is_red_light_viol = False
                        
                        if in_exit_line:
                            is_red_light_viol = True
                        elif in_stop_line:
                            if use_poly_exit:
                                is_stop_line_viol = True
                            else:
                                is_red_light_viol = True
                                
                        if is_red_light_viol:
                            is_red_light_violation = True
                            is_moto_violation = True
                            moto_violations.append("RED LIGHT VIOLATION")
                            violations.append({
                                "type": "Red Light Violation",
                                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                "details": f"Motorcycle crossed exit line zone during RED signal"
                            })
                        elif is_stop_line_viol:
                            is_moto_violation = True
                            moto_violations.append("STOP LINE VIOLATION")
                            violations.append({
                                "type": "Stop Line Violation",
                                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                "details": f"Motorcycle crossed stop line zone during RED signal"
                            })
                
                if no_helmet_count > 0:
                    is_moto_violation = True
                    moto_violations.append("NO HELMET")
                    violations.append({
                        "type": "Helmet Non-Compliance",
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "details": f"{no_helmet_count} rider(s) without helmet detected"
                    })
                
                if riders_count > 2:
                    is_moto_violation = True
                    moto_violations.append(f"TRIPLE RIDING ({riders_count})")
                    
                    # Compute collective triple riding bounding box around all detected rider heads
                    rx1 = min([b[0] for b in rider_head_boxes])
                    ry1 = min([b[1] for b in rider_head_boxes])
                    rx2 = max([b[2] for b in rider_head_boxes])
                    ry2 = max([b[3] for b in rider_head_boxes])
                    
                    # Add to violations and detections
                    violations.append({
                        "type": "Triple Riding",
                        "bbox": [int(rx1), int(ry1), int(rx2), int(ry2)],
                        "details": f"{riders_count} riders detected on single motorcycle"
                    })
                    detections.append({
                        "type": "triple_riding",
                        "bbox": [int(rx1), int(ry1), int(rx2), int(ry2)],
                        "confidence": float(conf)
                    })
                    
                    # Draw collective triple riding box around heads
                    tr_color = (180, 105, 255) # Hot Pink BGR
                    draw_custom_annotation(img, rx1, ry1, rx2, ry2, tr_color, f"TRIPLE_RIDING {conf*100:.0f}%")
                
                # Determine motorcycle color and label based on violation severity hierarchy
                if is_red_light_violation:
                    color = (0, 0, 200) # Deep Red
                    label_str = f"RED_LIGHT_VIOLATION {conf*100:.0f}%"
                elif is_stop_line_violation:
                    color = (0, 255, 255) # Yellow
                    label_str = f"STOP_LINE_VIOLATION {conf*100:.0f}%"
                elif no_helmet_count > 0:
                    color = (0, 0, 255) # Red
                    label_str = f"NO_HELMET {conf*100:.0f}%"
                elif riders_count > 2:
                    color = (180, 105, 255) # Hot Pink
                    label_str = f"TRIPLE_RIDING {conf*100:.0f}%"
                else:
                    color = (0, 200, 0) # Green (Compliant)
                    label_str = f"MOTORCYCLE {conf*100:.0f}%"
                    
                draw_custom_annotation(img, x1, y1, x2, y2, color, label_str)
            
            # 2. Handle Other Vehicles (License Plate & OCR Checks)
            elif cls_id in self.vehicle_classes:
                detections.append({
                    "type": "vehicle",
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": conf
                })
                
                # Crop vehicle to look for License Plates
                vehicle_crop = img[y1:y2, x1:x2]
                plate_results = self.plate_model(vehicle_crop, verbose=False)[0]
                
                for p_box in plate_results.boxes:
                    p_conf = float(p_box.conf[0])
                    p_xyxy = p_box.xyxy[0].cpu().numpy().astype(int)
                    px1, py1, px2, py2 = p_xyxy
                    
                    # Map cropped coordinates back to global coordinates
                    global_px1 = x1 + px1
                    global_py1 = y1 + py1
                    global_px2 = x1 + px2
                    global_py2 = y1 + py2
                    
                    # Crop Plate for OCR
                    plate_crop = vehicle_crop[py1:py2, px1:px2]
                    
                    # Convert to grayscale & threshold for OCR readability
                    plate_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                    _, plate_thresh = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    # OCR Processing
                    plate_text = "UNKNOWN"
                    if READER is not None:
                        ocr_result = READER.readtext(plate_thresh)
                        if ocr_result:
                            # Concatenate text with highest confidence
                            plate_text = "".join([res[1] for res in ocr_result if res[2] > 0.3]).strip().upper()
                    else:
                        plate_text = "KA03MH1234" # Mock fallback
                    
                    # Annotate License Plate
                    color = (255, 165, 0) # Light Blue / Amber
                    label_text = f"PLATE: {plate_text} {p_conf*100:.0f}%"
                    draw_custom_annotation(img, global_px1, global_py1, global_px2, global_py2, color, label_text)
                    
                    detections.append({
                        "type": "license_plate",
                        "bbox": [int(global_px1), int(global_py1), int(global_px2), int(global_py2)],
                        "text": plate_text,
                        "confidence": p_conf
                    })
                    
                # Check for Red Light Violation
                is_red_light_violation = False
                is_stop_line_violation = False
                tr_info = tracked_vehicles.get(tuple(xyxy.tolist())) if tracker is not None else None
                if tracker is not None:
                    if tr_info:
                        if tr_info["red_light_violation"]:
                            is_red_light_violation = True
                            violations.append({
                                "type": "Red Light Violation",
                                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                "details": f"Vehicle (Track ID: {tr_info['track_id']}) crossed exit line during RED signal"
                            })
                        elif tr_info["stop_line_violation"]:
                            is_stop_line_violation = True
                            violations.append({
                                "type": "Stop Line Violation",
                                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                "details": f"Vehicle (Track ID: {tr_info['track_id']}) crossed stop line during RED signal"
                            })
                else:
                    if intersection_state == "red":
                        veh_center_x = (x1 + x2) // 2
                        ref_pt = (veh_center_x, y2)
                        
                        has_calib = use_calibration and self.calibration is not None and "zones" in self.calibration
                        stop_line_pts = self.calibration["zones"].get("stop_line", []) if has_calib else []
                        exit_line_pts = self.calibration["zones"].get("exit_line", []) if has_calib else []
                        
                        use_poly_stop = len(stop_line_pts) >= 3
                        use_poly_exit = len(exit_line_pts) >= 3
                        
                        if use_poly_stop:
                            in_stop_line = point_in_zone(ref_pt, stop_line_pts)
                        elif not has_calib:
                            line_y = y_max if traffic_direction == "away" else y_min
                            in_stop_line = (y2 <= line_y if traffic_direction == "away" else y2 >= line_y) and (x_min <= veh_center_x <= x_max)
                        else:
                            in_stop_line = False
                            
                        if use_poly_exit:
                            in_exit_line = point_in_zone(ref_pt, exit_line_pts)
                        else:
                            in_exit_line = False
                            
                        is_stop_line_viol = False
                        is_red_light_viol = False
                        
                        if in_exit_line:
                            is_red_light_viol = True
                        elif in_stop_line:
                            if use_poly_exit:
                                is_stop_line_viol = True
                            else:
                                is_red_light_viol = True
                                
                        if is_red_light_viol:
                            is_red_light_violation = True
                            violations.append({
                                "type": "Red Light Violation",
                                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                "details": f"Vehicle crossed exit line zone during RED signal"
                            })
                        elif is_stop_line_viol:
                            is_stop_line_violation = True
                            violations.append({
                                "type": "Stop Line Violation",
                                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                "details": f"Vehicle crossed stop line zone during RED signal"
                            })
                
                veh_label = global_results.names[cls_id].upper()
                track_id_str = f" (ID: {tr_info['track_id']})" if (tracker is not None and tr_info) else ""
                if is_red_light_violation:
                    color = (0, 0, 200) # Deep Red
                    label_str = f"RED_LIGHT_VIOLATION {conf*100:.0f}%"
                elif is_stop_line_violation:
                    color = (0, 255, 255) # Yellow
                    label_str = f"STOP_LINE_VIOLATION {conf*100:.0f}%"
                else:
                    color = (0, 200, 0) # Green (Compliant)
                    label_str = f"{veh_label} {conf*100:.0f}%"
                    
                draw_custom_annotation(img, x1, y1, x2, y2, color, label_str)

        # Step 2.5: Process Person, Mobile Phone, and Seatbelt detections
        # Extract persons and cell phones from the global model results
        persons = []
        cell_phones = []
        for box in global_results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            if cls_id == 0 and conf >= 0.25: # person
                persons.append({"conf": conf, "bbox": xyxy})
            elif cls_id == 67 and conf >= 0.25: # cell phone
                cell_phones.append({"conf": conf, "bbox": xyxy})
                
        # Find all vehicles from filtered_boxes
        detected_vehicles = []
        for cls_id, conf, xyxy in filtered_boxes:
            if cls_id in self.vehicle_classes:
                detected_vehicles.append(xyxy)
                
        # Loop through each person to check for seatbelt and mobile usage
        for p_idx, p in enumerate(persons):
            px1, py1, px2, py2 = p["bbox"]
            p_conf = p["conf"]
            p_center = ((px1 + px2) // 2, (py1 + py2) // 2)
            
            # 1. Check if person is inside/overlapping with a vehicle (Car, Truck, Bus)
            in_vehicle = False
            for v_bbox in detected_vehicles:
                vx1, vy1, vx2, vy2 = v_bbox
                # If person center is inside vehicle bbox OR overlap IoU is > 0.05
                if (vx1 <= p_center[0] <= vx2 and vy1 <= p_center[1] <= vy2) or compute_iou(p["bbox"], v_bbox) > 0.05:
                    in_vehicle = True
                    break
                    
            # 2. Check if using mobile phone
            uses_mobile = False
            phone_conf_val = 0.0
            for phone in cell_phones:
                ph_x1, ph_y1, ph_x2, ph_y2 = phone["bbox"]
                ph_center = ((ph_x1 + ph_x2) // 2, (ph_y1 + ph_y2) // 2)
                # If cell phone center is inside person bbox OR overlaps with person bbox
                if (px1 <= ph_center[0] <= px2 and py1 <= ph_center[1] <= py2) or compute_iou(p["bbox"], phone["bbox"]) > 0.0:
                    uses_mobile = True
                    phone_conf_val = phone["conf"]
                    break
                    
            # 3. Check seatbelt compliance using CV Hough heuristic (if in vehicle and large enough)
            has_seatbelt = True
            pw = px2 - px1
            ph = py2 - py1
            
            if in_vehicle and pw >= 40 and ph >= 60:
                cx1 = max(0, px1 + int(pw * 0.15))
                cx2 = min(w, px2 - int(pw * 0.15))
                cy1 = max(0, py1 + int(ph * 0.15))
                cy2 = min(h, py1 + int(ph * 0.65))
                
                chest_crop = img[cy1:cy2, cx1:cx2]
                if chest_crop.size > 0:
                    gray = cv2.cvtColor(chest_crop, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 50, 150)
                    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=15, maxLineGap=8)
                    
                    has_seatbelt_line = False
                    if lines is not None:
                        for line in lines:
                            lx1, ly1, lx2, ly2 = line[0]
                            dx = lx2 - lx1
                            dy = ly2 - ly1
                            if dx != 0:
                                slope = abs(dy / dx)
                                if 0.35 <= slope <= 2.5: # 20 to 68 degrees
                                    has_seatbelt_line = True
                                    break
                    if not has_seatbelt_line:
                        has_seatbelt = False
                        
            # Annotate based on findings
            if uses_mobile:
                violations.append({
                    "type": "Using Mobile",
                    "bbox": [int(px1), int(py1), int(px2), int(py2)],
                    "details": f"Person using mobile phone (confidence: {phone_conf_val*100:.0f}%)"
                })
                detections.append({
                    "type": "using_mobile",
                    "bbox": [int(px1), int(py1), int(px2), int(py2)],
                    "confidence": phone_conf_val
                })
                # Purple color BGR for USING_MOBILE
                draw_custom_annotation(img, px1, py1, px2, py2, (211, 0, 148), f"USING_MOBILE {phone_conf_val*100:.0f}%")
                
            elif not has_seatbelt:
                violations.append({
                    "type": "No Seatbelt",
                    "bbox": [int(px1), int(py1), int(px2), int(py2)],
                    "details": "Driver/Passenger detected without seatbelt"
                })
                detections.append({
                    "type": "no_seatbelt",
                    "bbox": [int(px1), int(py1), int(px2), int(py2)],
                    "confidence": 0.85
                })
                # Orange color BGR for NO_SEATBELT
                draw_custom_annotation(img, px1, py1, px2, py2, (0, 128, 255), "NO_SEATBELT 85%")
                
            else:
                # Compliant person detection
                detections.append({
                    "type": "person",
                    "bbox": [int(px1), int(py1), int(px2), int(py2)],
                    "confidence": p_conf
                })
                # Green color BGR for COMPLIANT PERSON
                draw_custom_annotation(img, px1, py1, px2, py2, (0, 200, 0), f"PERSON {p_conf*100:.0f}%")

        # Step 3: Close-up / Fallback checks
        has_motorcycle = any(d["type"] in ["motorcycle", "motorcycle_safe", "motorcycle_violation"] for d in detections)
        has_plate = any(d["type"] == "license_plate" for d in detections)
        
        # Fallback 1: Run Helmet sub-detector on full image if no motorcycle was detected
        if not has_motorcycle:
            print("[*] No primary motorcycles detected. Running helmet sub-detector on the full image...")
            # Run Helmet Detector on full image
            helmet_results = self.helmet_model(img, verbose=False)[0]
            rider_head_boxes = []
            riders_count = 0
            
            for h_box in helmet_results.boxes:
                h_cls = int(h_box.cls[0])
                h_conf = float(h_box.conf[0])
                h_xyxy = h_box.xyxy[0].cpu().numpy().astype(int)
                hx1, hy1, hx2, hy2 = h_xyxy
                
                cls_name = helmet_results.names[h_cls]
                riders_count += 1
                rider_head_boxes.append((hx1, hy1, hx2, hy2))
                
                detections.append({
                    "type": cls_name.lower().replace(" ", "_"),
                    "bbox": [int(hx1), int(hy1), int(hx2), int(hy2)],
                    "confidence": h_conf
                })
                
                # Check for violation
                is_violation = ("without" in cls_name.lower() or "no" in cls_name.lower() or h_cls == 1)
                color = (0, 0, 255) if is_violation else (180, 50, 180) # Red or Purple BGR
                label_text = f"NO_HELMET {h_conf*100:.0f}%" if is_violation else f"HELMET {h_conf*100:.0f}%"
                if is_violation:
                    violations.append({
                        "type": "Helmet Non-Compliance",
                        "bbox": [int(hx1), int(hy1), int(hx2), int(hy2)],
                        "details": "Rider without helmet detected in close-up"
                    })
                draw_custom_annotation(img, hx1, hy1, hx2, hy2, color, label_text)

            # Draw Triple Riding box if more than 2 riders are detected on the full image
            if riders_count > 2 and len(rider_head_boxes) > 0:
                rx1 = min([b[0] for b in rider_head_boxes])
                ry1 = min([b[1] for b in rider_head_boxes])
                rx2 = max([b[2] for b in rider_head_boxes])
                ry2 = max([b[3] for b in rider_head_boxes])
                
                violations.append({
                    "type": "Triple Riding",
                    "bbox": [int(rx1), int(ry1), int(rx2), int(ry2)],
                    "details": f"{riders_count} riders detected in close-up"
                })
                detections.append({
                    "type": "triple_riding",
                    "bbox": [int(rx1), int(ry1), int(rx2), int(ry2)],
                    "confidence": 0.90
                })
                
                tr_color = (180, 105, 255) # Hot Pink BGR
                draw_custom_annotation(img, rx1, ry1, rx2, ry2, tr_color, "TRIPLE_RIDING 90%")

        # Fallback 2: Run License Plate sub-detector on full image if no plate was detected
        if not has_plate:
            print("[*] No license plates detected. Running plate sub-detector on the full image...")
            # Run License Plate Detector on full image
            plate_results = self.plate_model(img, verbose=False)[0]
            for p_box in plate_results.boxes:
                p_conf = float(p_box.conf[0])
                p_xyxy = p_box.xyxy[0].cpu().numpy().astype(int)
                px1, py1, px2, py2 = p_xyxy
                
                # Crop Plate for OCR
                plate_crop = img[py1:py2, px1:px2]
                plate_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                _, plate_thresh = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                plate_text = "UNKNOWN"
                if READER is not None:
                    ocr_result = READER.readtext(plate_thresh)
                    if ocr_result:
                        plate_text = "".join([res[1] for res in ocr_result if res[2] > 0.3]).strip().upper()
                else:
                    plate_text = "KA03MH1234"
                
                detections.append({
                    "type": "license_plate",
                    "bbox": [int(px1), int(py1), int(px2), int(py2)],
                    "text": plate_text,
                    "confidence": p_conf
                })
                
                color = (255, 165, 0) # Light Blue / Amber BGR
                label_text = f"PLATE: {plate_text} {p_conf*100:.0f}%"
                draw_custom_annotation(img, px1, py1, px2, py2, color, label_text)

            # Direct OCR Fallback if still no plate detections
            has_plate_updated = any(d["type"] == "license_plate" for d in detections)
            if not has_plate_updated and READER is not None:
                print("[*] Still no license plates detected. Running direct OCR fallback on full image...")
                ocr_result = READER.readtext(img)
                if ocr_result:
                    candidates = []
                    for res in ocr_result:
                        bbox, text, conf = res
                        text_str = text.strip()
                        if not text_str:
                            continue
                        xs = [pt[0] for pt in bbox]
                        ys = [pt[1] for pt in bbox]
                        xmin, xmax = int(min(xs)), int(max(xs))
                        ymin, ymax = int(min(ys)), int(max(ys))
                        area = (xmax - xmin) * (ymax - ymin)
                        candidates.append((xmin, ymin, xmax, ymax, text_str, conf, area))
                    
                    if candidates:
                        candidates = sorted(candidates, key=lambda x: x[6], reverse=True)
                        xmin, ymin, xmax, ymax, text_str, conf, area = candidates[0]
                        clean_text = text_str.upper()
                        
                        print(f"[+] Direct OCR fallback detected candidate text: '{clean_text}' (area={area}, conf={conf*100:.1f}%)")
                        
                        detections.append({
                            "type": "license_plate",
                            "bbox": [xmin, ymin, xmax, ymax],
                            "text": clean_text,
                            "confidence": float(conf)
                        })
                        
                        color = (255, 165, 0) # Light Blue / Amber BGR
                        label_text = f"PLATE: {clean_text} {conf*100:.0f}%"
                        draw_custom_annotation(img, xmin, ymin, xmax, ymax, color, label_text)

        # Draw the translucent zones
        line_color = (128, 128, 128) # Default gray
        if intersection_state == "red":
            line_color = (0, 0, 255)
        elif intersection_state == "yellow":
            line_color = (0, 255, 255)
        elif intersection_state == "green":
            line_color = (0, 255, 0)
            
        has_calib = use_calibration and self.calibration is not None and "zones" in self.calibration
        stop_line_pts = self.calibration["zones"].get("stop_line", []) if has_calib else []
        exit_line_pts = self.calibration["zones"].get("exit_line", []) if has_calib else []
        signal_roi_pts = self.calibration["zones"].get("signal_roi", []) if has_calib else []
        
        overlay = img.copy()
        
        # 1. Draw Stop Line
        if len(stop_line_pts) >= 3 and intersection_state == "red":
            pts = np.array(stop_line_pts, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], line_color)
            cv2.polylines(img, [pts], isClosed=True, color=line_color, thickness=max(2, box_thickness))
            cx, cy = stop_line_pts[0]
            cv2.putText(img, "STOP LINE ZONE", (cx + 15, cy - 8 if cy > 20 else cy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_color, text_thickness)
        elif not has_calib and intersection_state == "red":
            # Fallback to rectangular stop zone (only active during red light)
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), line_color, -1)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), line_color, max(2, box_thickness))
            cv2.putText(img, "STOP ZONE / ZEBRA CROSSING", (x_min + 15, y_min - 8 if y_min > 20 else y_min + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_color, text_thickness)
            
        # 2. Draw Exit Line
        if len(exit_line_pts) >= 3:
            pts = np.array(exit_line_pts, dtype=np.int32)
            exit_color = (0, 165, 255) if intersection_state == "red" else line_color
            cv2.fillPoly(overlay, [pts], exit_color)
            cv2.polylines(img, [pts], isClosed=True, color=exit_color, thickness=max(2, box_thickness))
            cx, cy = exit_line_pts[0]
            cv2.putText(img, "EXIT ZONE", (cx + 15, cy - 8 if cy > 20 else cy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, exit_color, text_thickness)
            
        # Apply translucency
        cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
        
        # 3. Draw Signal ROI (outline only)
        if len(signal_roi_pts) >= 2:
            pts = np.array(signal_roi_pts, dtype=np.int32)
            cv2.polylines(img, [pts], isClosed=len(signal_roi_pts) >= 3, color=(0, 255, 0), thickness=1)
            cx, cy = signal_roi_pts[0]
            cv2.putText(img, "SIGNAL ROI", (cx, cy - 5 if cy > 10 else cy + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (0, 255, 0), 1)

        # Draw traffic signal status overlay in the top-left corner
        overlay_text = f"SIGNAL: {intersection_state.upper()}"
        overlay_color = (128, 128, 128)
        if intersection_state == "red":
            overlay_color = (0, 0, 255)
        elif intersection_state == "yellow":
            overlay_color = (0, 255, 255)
        elif intersection_state == "green":
            overlay_color = (0, 255, 0)
            
        # Draw a semi-transparent background block for the signal text
        text_w, text_h = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.2, text_thickness * 2)[0]
        cv2.rectangle(img, (10, 10), (15 + text_w + 30, 15 + text_h + 10), (30, 30, 30), -1)
        # Draw status circle
        cv2.circle(img, (25 + text_w + 10, 15 + text_h // 2 + 5), int(text_h * 0.6), overlay_color, -1)
        # Draw text
        cv2.putText(img, overlay_text, (20, 15 + text_h + 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.2, (255, 255, 255), text_thickness * 2)

        # Step 4: Write output image (always save as jpg for compatibility)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_filename = f"processed_{base_name}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, img)
        print(f"[+] Annotated image saved to: {output_path}")
        
        # Save Metadata JSON
        meta_filename = f"meta_{os.path.splitext(os.path.basename(image_path))[0]}.json"
        meta_path = os.path.join(output_dir, meta_filename)
        metadata = {
            "image": image_path,
            "processed_image": output_path,
            "intersection_state": intersection_state,
            "stop_line": {
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "x_left": x_min,
                "y_left": y_min,
                "x_right": x_max,
                "y_right": y_max,
                "pole_x": pole_x,
                "is_physical": is_physical
            },
            "detections": detections,
            "violations": violations
        }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"[+] Violation metadata saved to: {meta_path}")

        return metadata

    def process_video(self, video_path, output_dir="data/processed", custom_line=None, traffic_direction="towards", use_calibration=True):
        """
        Process a video frame-by-frame, applying tracking and yielding progress.
        """
        os.makedirs(output_dir, exist_ok=True)
        import time
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video {video_path}")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 300
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate stride based on total frames (minimum 3, maximum 5)
        stride = max(3, min(5, total_frames // 100))
        print(f"[*] Calculated video processing stride: {stride} (Total frames: {total_frames})")
        
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        temp_output_filename = f"temp_processed_{base_name}.mp4"
        temp_output_path = os.path.join(output_dir, temp_output_filename)
        
        # Write to temporary mp4 with adjusted output FPS
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_fps = max(1.0, fps / stride)
        out = cv2.VideoWriter(temp_output_path, fourcc, out_fps, (width, height))
        
        # Initialize video tracker with max_age scaled to the stride
        tracker_max_age = max(10, stride * 2)
        tracker = VehicleTracker(max_age=tracker_max_age, traffic_direction=traffic_direction)
        
        frame_idx = 0
        all_violations = []
        unique_violations_tracked = set()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            if (frame_idx - 1) % stride != 0:
                continue
            
            temp_frame_path = os.path.join(output_dir, f"temp_frame_{base_name}.jpg")
            cv2.imwrite(temp_frame_path, frame)
            
            meta = self.process_image(temp_frame_path, output_dir=output_dir, tracker=tracker, custom_line=custom_line, traffic_direction=traffic_direction, use_calibration=use_calibration)
            
            processed_frame = cv2.imread(meta["processed_image"])
            out.write(processed_frame)
            
            # Save a copy of the processed frame for live Gradio visual updates
            live_frame_path = os.path.join(output_dir, f"live_frame_{base_name}.jpg")
            try:
                shutil.copy(meta["processed_image"], live_frame_path)
            except:
                pass
            
            try:
                os.remove(temp_frame_path)
                os.remove(meta["processed_image"])
                os.remove(os.path.join(output_dir, f"meta_temp_frame_{base_name}.json"))
            except:
                pass
                
            for v in meta["violations"]:
                track_id = None
                if "Track ID:" in v["details"]:
                    try:
                        track_id = int(v["details"].split("Track ID:")[1].strip().split()[0].replace(")", ""))
                    except:
                        pass
                
                if track_id is not None:
                    violation_key = f"red_light_{track_id}"
                    if violation_key not in unique_violations_tracked:
                        unique_violations_tracked.add(violation_key)
                        all_violations.append(v)
                else:
                    violation_key = f"red_light_{v['bbox']}"
                    if violation_key not in unique_violations_tracked:
                        unique_violations_tracked.add(violation_key)
                        all_violations.append(v)
            
            for v in meta["violations"]:
                if v["type"] != "Red Light Violation":
                    violation_key = f"{v['type']}_{v['bbox']}"
                    if violation_key not in unique_violations_tracked:
                        unique_violations_tracked.add(violation_key)
                        all_violations.append(v)
                        
            yield frame_idx, total_frames, (all_violations, live_frame_path)
                
        cap.release()
        out.release()
        
        # Clean up the last live frame copy if it exists
        live_frame_path = os.path.join(output_dir, f"live_frame_{base_name}.jpg")
        if os.path.exists(live_frame_path):
            try:
                os.remove(live_frame_path)
            except:
                pass
        
        # Transcode video to standard H.264 using ffmpeg
        final_output_filename = f"processed_{base_name}.mp4"
        final_output_path = os.path.join(output_dir, final_output_filename)
        
        print(f"[*] Transcoding {temp_output_path} to {final_output_path}...")
        
        cmd = [
            "ffmpeg", "-y",
            "-i", temp_output_path,
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            final_output_path
        ]
        
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            print("[+] Transcoding complete!")
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
        except Exception as e:
            print(f"[-] ffmpeg transcoding failed: {e}. Falling back to original video.")
            shutil.copy(temp_output_path, final_output_path)
            
        yield -1, total_frames, {
            "video": video_path,
            "processed_video": final_output_path,
            "violations": all_violations
        }

if __name__ == "__main__":
    # Quick test harness
    pipeline = TrafficViolationPipeline()
    print("[+] Pipeline initialized successfully.")
