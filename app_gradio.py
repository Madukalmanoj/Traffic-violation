import os
import json
import pandas as pd
import gradio as gr
from ultralytics import YOLO
from src.pipeline.inference import TrafficViolationPipeline

# Initialize the pipeline
weights_path = "weights"
pipeline = TrafficViolationPipeline(weights_dir=weights_path)

# Ensure sample dirs exist
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Historical violation log
history_list = []

# Video frame caching to make live slider dragging extremely fast and smooth
VIDEO_FRAME_CACHE = {}

def load_session_calibration():
    import os
    import json
    default_zones = {"stop_line": [], "exit_line": [], "signal_roi": []}
    if os.path.exists("calibration.json"):
        try:
            with open("calibration.json", "r") as f:
                data = json.load(f)
            zones = data.get("zones", {})
            for k in default_zones.keys():
                if k in zones:
                    default_zones[k] = [(int(pt[0]), int(pt[1])) for pt in zones[k]]
        except Exception as e:
            print(f"[-] Error loading calibration for session: {e}")
    return default_zones

CALIBRATION_ZONES = load_session_calibration()

def save_calibration_data(w, h, zones_data):
    import json
    payload = {
        "frame_width": w,
        "frame_height": h,
        "zones": zones_data
    }
    with open("calibration.json", "w") as f:
        json.dump(payload, f, indent=2)

def check_calibration_status():
    import os
    if os.path.exists("calibration.json"):
        try:
            with open("calibration.json", "r") as f:
                data = json.load(f)
            zones = data.get("zones", {})
            status_parts = []
            for k, v in zones.items():
                if len(v) >= (3 if k != "signal_roi" else 2):
                    status_parts.append(f"<span style='color: #4ade80;'>● {k} ({len(v)} pts)</span>")
                else:
                    status_parts.append(f"<span style='color: #9ca3af;'>○ {k} (not calibrated)</span>")
            return f"<div style='background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); padding: 12px; border-radius: 8px; margin-bottom: 15px;'><strong>📁 Polygon Calibration Loaded (Active):</strong> {' &nbsp;|&nbsp; '.join(status_parts)}</div>"
        except Exception as e:
            return f"<div style='background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); padding: 12px; border-radius: 8px; margin-bottom: 15px;'><strong>⚠️ Error loading calibration.json:</strong> {str(e)}</div>"
    else:
        return "<div style='background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); padding: 12px; border-radius: 8px; margin-bottom: 15px;'><strong>ℹ️ No calibration.json found.</strong> Falling back to automatic detection / slider-based stop zone. Run <code>python zone_calibrator.py</code> to calibrate.</div>"

def get_video_first_frame(video_path):
    if len(VIDEO_FRAME_CACHE) > 5:
        VIDEO_FRAME_CACHE.clear()
    if video_path in VIDEO_FRAME_CACHE:
        return VIDEO_FRAME_CACHE[video_path].copy()
    import cv2
    try:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            VIDEO_FRAME_CACHE[video_path] = frame
            return frame.copy()
    except Exception as e:
        print(f"[-] Error reading first video frame: {e}")
    return None

def get_video_frame_at_time(video_path, timestamp_sec):
    import cv2
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0
        frame_idx = int(timestamp_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return frame
    except Exception as e:
        print(f"[-] Error reading frame at {timestamp_sec}s: {e}")
    return None

def get_video_duration(video_path):
    if not video_path:
        return 0.0
    import cv2
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if fps > 0 and frame_count > 0:
            return frame_count / fps
    except Exception as e:
        print(f"[-] Error getting video duration: {e}")
    return 0.0

def get_base64_from_file(file_path):
    import base64
    import os
    if not os.path.exists(file_path):
        return ""
    try:
        with open(file_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(file_path)[1].lower().replace(".", "")
        if ext == "jpg" or ext == "jpeg":
            ext = "jpeg"
        return f"data:image/{ext};base64,{encoded}"
    except Exception as e:
        print(f"[-] Error converting file to base64: {e}")
        return ""

def get_base64_from_numpy(img_array):
    import cv2
    import base64
    try:
        # Convert RGB (from Gradio) to BGR for OpenCV encoding
        bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".jpg", bgr)
        encoded = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception as e:
        print(f"[-] Error converting numpy to base64: {e}")
        return ""

CALIB_REF_PATH = "data/processed/calibration_reference.jpg"

def save_calib_reference(img_array):
    import cv2
    try:
        # img_array is RGB from Gradio
        bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        cv2.imwrite(CALIB_REF_PATH, bgr)
        return True
    except Exception as e:
        print(f"[-] Error saving calibration reference: {e}")
        return False

def get_base64_from_file(file_path):
    import base64
    import os
    if not os.path.exists(file_path):
        return ""
    try:
        with open(file_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(file_path)[1].lower().replace(".", "")
        if ext == "jpg" or ext == "jpeg":
            ext = "jpeg"
        return f"data:image/{ext};base64,{encoded}"
    except Exception as e:
        print(f"[-] Error converting file to base64: {e}")
        return ""

def load_initial_calib_base64():
    return get_base64_from_file(CALIB_REF_PATH)

def load_initial_calib_json():
    import os
    import json
    if os.path.exists("calibration.json"):
        try:
            with open("calibration.json", "r") as f:
                data = json.load(f)
            data["from_file"] = True
            return json.dumps(data)
        except Exception as e:
            print(f"[-] Error loading initial JSON calibration: {e}")
    return json.dumps({"w": 1280, "h": 720, "zones": {"stop_line": [], "exit_line": [], "signal_roi": []}, "from_file": False})

def toggle_image_calibration(image):
    if image is None:
        return gr.update(visible=False), ""
    try:
        if isinstance(image, dict):
            img_arr = image.get("composite", None) or image.get("background", None)
        else:
            img_arr = image
        if img_arr is None:
            return gr.update(visible=False), ""
        save_calib_reference(img_arr)
        b64_str = get_base64_from_file(CALIB_REF_PATH)
        return gr.update(visible=True), b64_str
    except Exception as e:
        print(f"[-] toggle_image_calibration failed: {e}")
        return gr.update(visible=False), ""

def toggle_video_calibration(video_path, timestamp_str="0.0"):
    if not video_path:
        return gr.update(value=0.0, maximum=10.0), ""
    try:
        duration = get_video_duration(video_path)
        if duration <= 0:
            duration = 10.0
        
        # Parse timestamp from JS time extraction
        try:
            timestamp = float(timestamp_str) if timestamp_str else 0.0
        except ValueError:
            timestamp = 0.0
            
        if timestamp < 0 or timestamp > duration:
            timestamp = 0.0
            
        frame = get_video_frame_at_time(video_path, timestamp)
        b64_str = ""
        if frame is not None:
            import cv2
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            save_calib_reference(rgb)
            b64_str = get_base64_from_file(CALIB_REF_PATH)
        return gr.update(value=timestamp, maximum=duration, step=0.1, interactive=True), b64_str
    except Exception as e:
        print(f"[-] toggle_video_calibration failed: {e}")
        return gr.update(value=0.0, maximum=10.0), ""

def handle_slider_change(video_path, timestamp):
    if not video_path:
        return ""
    try:
        frame = get_video_frame_at_time(video_path, timestamp)
        if frame is not None:
            import cv2
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            save_calib_reference(rgb)
            return get_base64_from_file(CALIB_REF_PATH)
    except Exception as e:
        print(f"[-] handle_slider_change failed: {e}")
    return ""

def handle_timestamp_change(video_path, timestamp_str):
    if not video_path:
        return "", gr.update(value=0.0)
    try:
        timestamp = float(timestamp_str) if timestamp_str else 0.0
        frame = get_video_frame_at_time(video_path, timestamp)
        b64_str = ""
        if frame is not None:
            import cv2
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            save_calib_reference(rgb)
            b64_str = get_base64_from_file(CALIB_REF_PATH)
        return b64_str, gr.update(value=timestamp)
    except Exception as e:
        print(f"[-] handle_timestamp_change failed: {e}")
    return "", gr.update(value=0.0)

def update_video_preview(video_path, use_custom):
    if video_path is None:
        return gr.update(value=None, visible=False), gr.update(visible=False)
    import cv2
    import os
    import json
    import numpy as np
    try:
        frame = get_video_first_frame(video_path)
        if frame is None:
            return gr.update(value=None, visible=False), gr.update(visible=False)
        h, w = frame.shape[:2]
        
        calib = None
        if os.path.exists("calibration.json"):
            try:
                with open("calibration.json", "r") as f:
                    calib = json.load(f)
            except:
                pass
                
        has_calib = use_custom and calib is not None and "zones" in calib
        stop_line_pts = calib["zones"].get("stop_line", []) if has_calib else []
        exit_line_pts = calib["zones"].get("exit_line", []) if has_calib else []
        signal_roi_pts = calib["zones"].get("signal_roi", []) if has_calib else []
        
        overlay = frame.copy()
        has_any_poly = len(stop_line_pts) > 0 or len(exit_line_pts) > 0 or len(signal_roi_pts) > 0
        if has_any_poly:
            if len(stop_line_pts) > 0:
                pts = np.array(stop_line_pts, dtype=np.int32)
                for pt in stop_line_pts:
                    cv2.circle(frame, tuple(pt), 6, (0, 0, 255), -1)
                if len(stop_line_pts) >= 2:
                    cv2.polylines(frame, [pts], isClosed=len(stop_line_pts) >= 3, color=(0, 0, 255), thickness=3)
                if len(stop_line_pts) >= 3:
                    cv2.fillPoly(overlay, [pts], (0, 0, 255))
                    cv2.putText(frame, "STOP LINE ZONE", (stop_line_pts[0][0] + 15, stop_line_pts[0][1] - 8), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if len(exit_line_pts) > 0:
                pts = np.array(exit_line_pts, dtype=np.int32)
                for pt in exit_line_pts:
                    cv2.circle(frame, tuple(pt), 6, (0, 165, 255), -1)
                if len(exit_line_pts) >= 2:
                    cv2.polylines(frame, [pts], isClosed=len(exit_line_pts) >= 3, color=(0, 165, 255), thickness=3)
                if len(exit_line_pts) >= 3:
                    cv2.fillPoly(overlay, [pts], (0, 165, 255))
                    cv2.putText(frame, "EXIT ZONE", (exit_line_pts[0][0] + 15, exit_line_pts[0][1] - 8), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            if len(signal_roi_pts) > 0:
                pts = np.array(signal_roi_pts, dtype=np.int32)
                for pt in signal_roi_pts:
                    cv2.circle(frame, tuple(pt), 6, (0, 255, 0), -1)
                if len(signal_roi_pts) >= 2:
                    cv2.polylines(frame, [pts], isClosed=len(signal_roi_pts) >= 3, color=(0, 255, 0), thickness=2)
                    cv2.putText(frame, "SIGNAL ROI", (signal_roi_pts[0][0], signal_roi_pts[0][1] - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        
        preview_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return gr.update(value=preview_rgb, visible=True), gr.update(visible=False)
    except Exception as e:
        print(f"[-] update_video_preview failed: {e}")
        return gr.update(value=None, visible=False), gr.update(visible=False)

def update_image_preview(image, use_custom):
    if image is None:
        return None
    import cv2
    import os
    import json
    import numpy as np
    try:
        if isinstance(image, dict):
            img_arr = image.get("composite", None) or image.get("background", None)
        else:
            img_arr = image

        if img_arr is None or not hasattr(img_arr, "shape"):
            return image

        frame = img_arr.copy()
        h, w = frame.shape[:2]
        
        calib = None
        if os.path.exists("calibration.json"):
            try:
                with open("calibration.json", "r") as f:
                    calib = json.load(f)
            except:
                pass
                
        has_calib = use_custom and calib is not None and "zones" in calib
        stop_line_pts = calib["zones"].get("stop_line", []) if has_calib else []
        exit_line_pts = calib["zones"].get("exit_line", []) if has_calib else []
        signal_roi_pts = calib["zones"].get("signal_roi", []) if has_calib else []
        
        overlay = frame.copy()
        has_any_poly = len(stop_line_pts) > 0 or len(exit_line_pts) > 0 or len(signal_roi_pts) > 0
        if has_any_poly:
            if len(stop_line_pts) > 0:
                pts = np.array(stop_line_pts, dtype=np.int32)
                for pt in stop_line_pts:
                    cv2.circle(frame, tuple(pt), 6, (255, 0, 0), -1)
                if len(stop_line_pts) >= 2:
                    cv2.polylines(frame, [pts], isClosed=len(stop_line_pts) >= 3, color=(255, 0, 0), thickness=3)
                if len(stop_line_pts) >= 3:
                    cv2.fillPoly(overlay, [pts], (255, 0, 0))
                    cv2.putText(frame, "STOP LINE ZONE", (stop_line_pts[0][0] + 15, stop_line_pts[0][1] - 8), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            if len(exit_line_pts) > 0:
                pts = np.array(exit_line_pts, dtype=np.int32)
                for pt in exit_line_pts:
                    cv2.circle(frame, tuple(pt), 6, (255, 165, 0), -1)
                if len(exit_line_pts) >= 2:
                    cv2.polylines(frame, [pts], isClosed=len(exit_line_pts) >= 3, color=(255, 165, 0), thickness=3)
                if len(exit_line_pts) >= 3:
                    cv2.fillPoly(overlay, [pts], (255, 165, 0))
                    cv2.putText(frame, "EXIT ZONE", (exit_line_pts[0][0] + 15, exit_line_pts[0][1] - 8), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            if len(signal_roi_pts) > 0:
                pts = np.array(signal_roi_pts, dtype=np.int32)
                for pt in signal_roi_pts:
                    cv2.circle(frame, tuple(pt), 6, (0, 255, 0), -1)
                if len(signal_roi_pts) >= 2:
                    cv2.polylines(frame, [pts], isClosed=len(signal_roi_pts) >= 3, color=(0, 255, 0), thickness=2)
                    cv2.putText(frame, "SIGNAL ROI", (signal_roi_pts[0][0], signal_roi_pts[0][1] - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        return frame
    except Exception as e:
        print(f"[-] update_image_preview failed: {e}")
        return image

def get_calibrator_html_code(prefix):
    code = """
    <style>
    /* Scope styles to bypass Gradio preprocessor scope prefixing */
    body.image-calib-active .gradio-container .contain #image-preview-group {
      display: none !important;
    }
    body.image-calib-active .gradio-container .contain #image-calibrator-group {
      display: block !important;
    }

    body.video-calib-active .gradio-container .contain #video-preview-group {
      display: none !important;
    }
    body.video-calib-active .gradio-container .contain #video-calibrator-group {
      display: block !important;
    }

    .{prefix}-calib-container {
      display: flex;
      flex-direction: column;
      gap: 12px;
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 12px;
      padding: 16px;
      backdrop-filter: blur(12px);
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }
    .{prefix}-calib-header {
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      padding-bottom: 8px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .{prefix}-calib-title {
      font-size: 1.05rem;
      font-weight: 700;
      color: #f8fafc;
      letter-spacing: 0.5px;
    }
    .{prefix}-calib-panel {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
    }
    .{prefix}-calib-control-group {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .{prefix}-control-label {
      font-size: 0.8rem;
      color: #94a3b8;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .{prefix}-zone-pills {
      display: flex;
      gap: 4px;
      background: rgba(0, 0, 0, 0.3);
      padding: 3px;
      border-radius: 8px;
      border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .{prefix}-zone-pill {
      border: none;
      background: transparent;
      color: #94a3b8;
      padding: 5px 10px;
      font-size: 0.8rem;
      font-weight: 700;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .{prefix}-zone-pill.active.stop-line {
      background: rgba(239, 68, 68, 0.2);
      color: #f87171;
      border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .{prefix}-zone-pill.active.exit-line {
      background: rgba(249, 115, 22, 0.2);
      color: #fb923c;
      border: 1px solid rgba(249, 115, 22, 0.3);
    }
    .{prefix}-zone-pill.active.signal-roi {
      background: rgba(34, 197, 94, 0.2);
      color: #4ade80;
      border: 1px solid rgba(34, 197, 94, 0.3);
    }
    .{prefix}-action-btns {
      display: flex;
      gap: 6px;
    }
    .{prefix}-action-btn {
      background: rgba(30, 41, 59, 0.8);
      border: 1px solid rgba(255, 255, 255, 0.1);
      color: #cbd5e1;
      padding: 6px 12px;
      font-size: 0.8rem;
      font-weight: 600;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .{prefix}-action-btn:hover {
      background: rgba(51, 65, 85, 0.9);
      color: #ffffff;
      border-color: rgba(255, 255, 255, 0.2);
    }
    .{prefix}-canvas-container {
      position: relative;
      width: 100%;
      aspect-ratio: 16 / 9;
      background: #020617;
      border-radius: 10px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      overflow: hidden;
      display: flex;
      justify-content: center;
      align-items: center;
    }
    #{prefix}-calib-canvas {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: none;
      cursor: crosshair;
    }
    .{prefix}-placeholder-msg {
      color: #64748b;
      font-size: 0.85rem;
      text-align: center;
      line-height: 1.5;
    }
    .{prefix}-coord-panel {
      background: rgba(15, 23, 42, 0.9);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 10px;
      padding: 12px;
      margin-top: 10px;
      box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
    }
    </style>

    <div class="{prefix}-calib-container">
      <div class="{prefix}-calib-header">
        <span class="{prefix}-calib-title">📸 Interactive Region Calibrator (6-Point Mode)</span>
      </div>
      
      <div class="{prefix}-calib-panel">
        <div class="{prefix}-calib-control-group">
          <span class="{prefix}-control-label">Zone:</span>
          <div class="{prefix}-zone-pills">
            <button id="{prefix}-btn-stop-line" class="{prefix}-zone-pill active stop-line" onclick="window.{prefix}SetZone('stop_line')">🟥 Stop Line</button>
            <button id="{prefix}-btn-exit-line" class="{prefix}-zone-pill exit-line" onclick="window.{prefix}SetZone('exit_line')">🟧 Exit Zone</button>
            <button id="{prefix}-btn-signal-roi" class="{prefix}-zone-pill signal-roi" onclick="window.{prefix}SetZone('signal_roi')">🟩 Signal ROI</button>
          </div>
        </div>

        <div class="{prefix}-calib-control-group">
          <span class="{prefix}-control-label">Active:</span>
          <div style="display: flex; gap: 10px; align-items: center; background: rgba(0, 0, 0, 0.3); padding: 5px 10px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.05);">
            <label style="display: flex; align-items: center; gap: 4px; font-size: 0.75rem; color: #cbd5e1; cursor: pointer; user-select: none; font-weight: 700;">
              <input type="checkbox" id="{prefix}-toggle-stop-line" checked onchange="window.{prefix}ToggleZone('stop_line', this.checked)" style="accent-color: #ef4444; width: 14px; height: 14px; cursor: pointer;"> Stop Line
            </label>
            <label style="display: flex; align-items: center; gap: 4px; font-size: 0.75rem; color: #cbd5e1; cursor: pointer; user-select: none; font-weight: 700;">
              <input type="checkbox" id="{prefix}-toggle-exit-line" checked onchange="window.{prefix}ToggleZone('exit_line', this.checked)" style="accent-color: #f97316; width: 14px; height: 14px; cursor: pointer;"> Exit Zone
            </label>
            <label style="display: flex; align-items: center; gap: 4px; font-size: 0.75rem; color: #cbd5e1; cursor: pointer; user-select: none; font-weight: 700;">
              <input type="checkbox" id="{prefix}-toggle-signal-roi" checked onchange="window.{prefix}ToggleZone('signal_roi', this.checked)" style="accent-color: #22c55e; width: 14px; height: 14px; cursor: pointer;"> Signal ROI
            </label>
          </div>
        </div>
        
        <div class="{prefix}-calib-control-group">
          <div class="{prefix}-action-btns">
            <button class="{prefix}-action-btn" onclick="window.{prefix}ResetActiveZone()">🔄 Reset Active</button>
            <button class="{prefix}-action-btn" onclick="window.{prefix}ResetAllZones()">🔄 Reset All</button>
          </div>
        </div>
      </div>

      <div class="{prefix}-canvas-container">
        <canvas id="{prefix}-calib-canvas"></canvas>
        <div id="{prefix}-no-image-msg" class="{prefix}-placeholder-msg">
          No reference frame loaded.<br>Upload an image or adjust the timestamp slider to calibrate zones.
        </div>
      </div>

      <div id="{prefix}-coord-panel" class="{prefix}-coord-panel">
        <!-- real-time coordinates -->
      </div>
      <!-- Note: Closing div is moved to the end of the script block so that script is nested inside container -->

    <script>
    (function() {
      const prefix = "{prefix}";
      if (window.{prefix}CalibImageCheckInterval) {
        clearInterval(window.{prefix}CalibImageCheckInterval);
      }
      if (window.{prefix}CalibJsonCheckInterval) {
        clearInterval(window.{prefix}CalibJsonCheckInterval);
      }

      let zones = { stop_line: [], exit_line: [], signal_roi: [] };
      let activeZone = 'stop_line';
      let draggedPoint = null;
      let hoveredPoint = null;
      let selectedPoint = null;
      let isDraggingPolygon = false;
      let dragStartPos = { x: 0, y: 0 };
      let polygonStartPoints = [];
      let img = new Image();
      let isImageLoaded = false;
      
      let zoneBackups = { stop_line: null, exit_line: null, signal_roi: null };
      let zoneEnabled = { stop_line: true, exit_line: true, signal_roi: true };
      
      const colors = {
        stop_line: { stroke: '#ef4444', fill: 'rgba(239, 68, 68, 0.22)', label: 'STOP LINE ZONE' },
        exit_line: { stroke: '#f97316', fill: 'rgba(249, 115, 22, 0.22)', label: 'EXIT ZONE' },
        signal_roi: { stroke: '#22c55e', fill: 'rgba(34, 197, 94, 0.12)', label: 'SIGNAL ROI' }
      };
      
      function getInitialPoints(zoneName, w, h) {
        let yTop, yBottom;
        if (zoneName === 'stop_line') {
          yTop = Math.round(h * 0.35);
          yBottom = Math.round(h * 0.48);
        } else if (zoneName === 'exit_line') {
          yTop = Math.round(h * 0.60);
          yBottom = Math.round(h * 0.73);
        } else {
          yTop = Math.round(h * 0.10);
          yBottom = Math.round(h * 0.25);
        }
        
        const xLeft = Math.round(w * 0.3);
        const xMid = Math.round(w * 0.5);
        const xRight = Math.round(w * 0.7);
        
        return [
          [xLeft, yTop],      // 1. Top-Left
          [xMid, yTop],       // 2. Top-Middle
          [xRight, yTop],      // 3. Top-Right
          [xRight, yBottom],   // 4. Bottom-Right
          [xMid, yBottom],    // 5. Bottom-Middle
          [xLeft, yBottom]     // 6. Bottom-Left
        ];
      }

      function enforceExactly6Points(w, h) {
        const zoneNames = ['stop_line', 'exit_line', 'signal_roi'];
        for (const zName of zoneNames) {
          if (zoneEnabled[zName]) {
            if (!zones[zName] || zones[zName].length !== 6) {
              zones[zName] = getInitialPoints(zName, w, h);
            }
          } else {
            zones[zName] = [];
          }
        }
      }

      function pointInPolygon(x, y, pts) {
        let inside = false;
        for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
          const xi = pts[i][0], yi = pts[i][1];
          const xj = pts[j][0], yj = pts[j][1];
          const intersect = ((yi > y) !== (yj > y)) &&
                            (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
          if (intersect) inside = !inside;
        }
        return inside;
      }

      function init() {
        const canvas = document.getElementById("{prefix}-calib-canvas");
        const placeholder = document.getElementById("{prefix}-no-image-msg");
        if (!canvas || !placeholder) {
          setTimeout(init, 100);
          return;
        }
        
        canvas.removeEventListener('mousedown', handleMouseDown);
        canvas.removeEventListener('mousemove', handleMouseMove);
        canvas.addEventListener('mousedown', handleMouseDown);
        canvas.addEventListener('mousemove', handleMouseMove);
        
        window.removeEventListener('mouseup', handleMouseUp);
        window.addEventListener('mouseup', handleMouseUp);
        
        // Touch supports
        canvas.removeEventListener('touchstart', handleTouchStart);
        canvas.removeEventListener('touchmove', handleTouchMove);
        canvas.removeEventListener('touchend', handleTouchEnd);
        canvas.addEventListener('touchstart', handleTouchStart, { passive: false });
        canvas.addEventListener('touchmove', handleTouchMove, { passive: false });
        canvas.addEventListener('touchend', handleTouchEnd, { passive: false });
        
        // Keyboard nudges
        window.removeEventListener('keydown', handleKeyDown);
        window.addEventListener('keydown', handleKeyDown);

        let lastSrc = "";
        window.{prefix}CalibImageCheckInterval = setInterval(() => {
          const srcTextarea = document.querySelector("#{prefix}-calib-image-src textarea, #{prefix}-calib-image-src input");
          if (srcTextarea && srcTextarea.value !== lastSrc) {
            lastSrc = srcTextarea.value;
            if (lastSrc) {
              img.onload = function() {
                isImageLoaded = true;
                canvas.style.display = "block";
                placeholder.style.display = "none";
                enforceExactly6Points(img.naturalWidth || 1280, img.naturalHeight || 720);
                saveState();
                redraw();
              };
              img.src = lastSrc;
            } else {
              isImageLoaded = false;
              canvas.style.display = "none";
              placeholder.style.display = "block";
            }
          }
        }, 250);
        
        if (prefix === "vid") {
          if (window.vidPlayerSyncInterval) {
            clearInterval(window.vidPlayerSyncInterval);
          }
          window.vidPlayerSyncInterval = setInterval(() => {
            const videoEl = document.querySelector('#video-input video');
            if (videoEl && !videoEl.dataset.hasCalibListener) {
              videoEl.dataset.hasCalibListener = "true";
              
              const syncFrame = () => {
                if (document.body.classList.contains('video-calib-active')) {
                  const tsInput = document.querySelector('#video-calib-timestamp textarea, #video-calib-timestamp input');
                  if (tsInput) {
                    const diff = Math.abs(parseFloat(tsInput.value || "0") - videoEl.currentTime);
                    if (diff > 0.05) {
                      tsInput.value = videoEl.currentTime.toString();
                      tsInput.dispatchEvent(new Event('input', { bubbles: true }));
                    }
                  }
                }
              };
              
              videoEl.addEventListener('seeked', syncFrame);
              videoEl.addEventListener('pause', syncFrame);
            }
          }, 500);
        }
        
        let lastJson = "";
        window.{prefix}CalibJsonCheckInterval = setInterval(() => {
          const jsonTextarea = document.querySelector("#{prefix}-calib-json-input textarea, #{prefix}-calib-json-input input");
          if (jsonTextarea && jsonTextarea.value !== lastJson) {
            lastJson = jsonTextarea.value;
            if (lastJson) {
              try {
                const data = JSON.parse(lastJson);
                if (data && data.zones) {
                  zones = data.zones;
                  const fromFile = data.from_file !== false; // If from_file isn't explicitly false, treat it as true
                  
                  // Update toggles and button states based on loaded zones
                  for (const zName of ['stop_line', 'exit_line', 'signal_roi']) {
                    const hasPoints = zones[zName] && zones[zName].length >= 2;
                    let isEnabled = hasPoints;
                    if (!fromFile) {
                      isEnabled = true; // Default to enabled if no saved file exists
                    }
                    
                    zoneEnabled[zName] = isEnabled;
                    
                    const toggleInput = document.getElementById(`{prefix}-toggle-${zName.replace('_', '-')}`);
                    if (toggleInput) toggleInput.checked = isEnabled;
                    
                    const btn = document.getElementById(`{prefix}-btn-${zName.replace('_', '-')}`);
                    if (btn) {
                      btn.disabled = !isEnabled;
                      btn.style.opacity = isEnabled ? "1" : "0.4";
                      btn.style.pointerEvents = isEnabled ? "auto" : "none";
                    }
                  }
                  
                  // Ensure active zone is an enabled one
                  if (!zoneEnabled[activeZone]) {
                    const remaining = Object.keys(zoneEnabled).find(k => zoneEnabled[k]);
                    if (remaining) {
                      activeZone = remaining;
                      document.querySelectorAll('.{prefix}-zone-pill').forEach(b => b.classList.remove('active'));
                      const activeBtn = document.getElementById(`{prefix}-btn-${remaining.replace('_', '-')}`);
                      if (activeBtn) activeBtn.classList.add('active');
                    } else {
                      activeZone = null;
                      document.querySelectorAll('.{prefix}-zone-pill').forEach(b => b.classList.remove('active'));
                    }
                  }
                  
                  enforceExactly6Points(img.naturalWidth || 1280, img.naturalHeight || 720);
                  redraw();
                }
              } catch(e) {
                console.error("Error parsing JSON:", e);
              }
            }
          }
        }, 500);
      }
      
      window.{prefix}ToggleZone = function(zoneName, isEnabled) {
        zoneEnabled[zoneName] = isEnabled;
        const btn = document.getElementById(`{prefix}-btn-${zoneName.replace('_', '-')}`);
        
        if (isEnabled) {
          if (btn) {
            btn.disabled = false;
            btn.style.opacity = "1";
            btn.style.pointerEvents = "auto";
          }
          // Restore points
          if (zoneBackups[zoneName] && zoneBackups[zoneName].length === 6) {
            zones[zoneName] = zoneBackups[zoneName];
          } else {
            const w = img.naturalWidth || 1280;
            const h = img.naturalHeight || 720;
            zones[zoneName] = getInitialPoints(zoneName, w, h);
          }
          // Set as active zone since user just enabled it
          window.{prefix}SetZone(zoneName);
        } else {
          if (btn) {
            btn.disabled = true;
            btn.style.opacity = "0.4";
            btn.style.pointerEvents = "none";
          }
          // Backup current points before clearing if they are valid
          if (zones[zoneName] && zones[zoneName].length === 6) {
            zoneBackups[zoneName] = zones[zoneName];
          }
          zones[zoneName] = [];
          
          // If we disabled the active zone, switch active zone to any remaining enabled zone
          if (activeZone === zoneName) {
            const remaining = Object.keys(zoneEnabled).find(k => zoneEnabled[k]);
            if (remaining) {
              window.{prefix}SetZone(remaining);
            } else {
              activeZone = null;
              document.querySelectorAll('.{prefix}-zone-pill').forEach(b => b.classList.remove('active'));
            }
          }
        }
        
        saveState();
        redraw();
      };
      
      window.{prefix}SetZone = function(zoneName) {
        if (!zoneEnabled[zoneName]) return;
        activeZone = zoneName;
        document.querySelectorAll('.{prefix}-zone-pill').forEach(btn => btn.classList.remove('active'));
        const activeBtn = document.getElementById(`{prefix}-btn-${zoneName.replace('_', '-')}`);
        if (activeBtn) activeBtn.classList.add('active');
        selectedPoint = null;
        redraw();
      };
      
      window.{prefix}ResetActiveZone = function() {
        if (isImageLoaded && activeZone) {
          zones[activeZone] = getInitialPoints(activeZone, img.naturalWidth || 1280, img.naturalHeight || 720);
          saveState();
          redraw();
        }
      };
      
      window.{prefix}ResetAllZones = function() {
        if (confirm("Reset all zones to default rectangles?")) {
          const w = img.naturalWidth || 1280;
          const h = img.naturalHeight || 720;
          
          // Reset enabled state to true for all zones
          for (const zName of ['stop_line', 'exit_line', 'signal_roi']) {
            zoneEnabled[zName] = true;
            const toggleInput = document.getElementById(`{prefix}-toggle-${zName.replace('_', '-')}`);
            if (toggleInput) toggleInput.checked = true;
            const btn = document.getElementById(`{prefix}-btn-${zName.replace('_', '-')}`);
            if (btn) {
              btn.disabled = false;
              btn.style.opacity = "1";
              btn.style.pointerEvents = "auto";
            }
          }
          
          activeZone = 'stop_line';
          document.querySelectorAll('.{prefix}-zone-pill').forEach(btn => btn.classList.remove('active'));
          const activeBtn = document.getElementById(`{prefix}-btn-stop-line`);
          if (activeBtn) activeBtn.classList.add('active');
          
          zones = {
            stop_line: getInitialPoints('stop_line', w, h),
            exit_line: getInitialPoints('exit_line', w, h),
            signal_roi: getInitialPoints('signal_roi', w, h)
          };
          saveState();
          redraw();
        }
      };
      
      function saveState() {
        const jsonTextarea = document.querySelector("#{prefix}-calib-json-input textarea, #{prefix}-calib-json-input input");
        if (jsonTextarea) {
          const payload = {
            w: img.naturalWidth || 1280,
            h: img.naturalHeight || 720,
            zones: zones
          };
          jsonTextarea.value = JSON.stringify(payload);
          jsonTextarea.dispatchEvent(new Event("input", { bubbles: true }));
        }
        updateCoordinateTable();
      }
      
      function getMousePos(e) {
        const canvas = document.getElementById("{prefix}-calib-canvas");
        const rect = canvas.getBoundingClientRect();
        
        let clientX, clientY;
        if (e.touches && e.touches.length > 0) {
          clientX = e.touches[0].clientX;
          clientY = e.touches[0].clientY;
        } else {
          clientX = e.clientX;
          clientY = e.clientY;
        }
        
        const canvasX = clientX - rect.left;
        const canvasY = clientY - rect.top;
        
        return {
          clientX,
          clientY,
          canvasX,
          canvasY,
          imgX: Math.round((canvasX / rect.width) * img.naturalWidth),
          imgY: Math.round((canvasY / rect.height) * img.naturalHeight)
        };
      }
      
      function handleMouseDown(e) {
        if (!isImageLoaded) return;
        const pos = getMousePos(e);
        
        const canvas = document.getElementById("{prefix}-calib-canvas");
        const rect = canvas.getBoundingClientRect();
        let clickedIndex = -1;
        let minDist = 15;
        
        const pts = zones[activeZone] || [];
        for (let i = 0; i < pts.length; i++) {
          const pt = pts[i];
          const screenX = (pt[0] / img.naturalWidth) * rect.width;
          const screenY = (pt[1] / img.naturalHeight) * rect.height;
          const dist = Math.sqrt(Math.pow(screenX - pos.canvasX, 2) + Math.pow(screenY - pos.canvasY, 2));
          if (dist < minDist) {
            minDist = dist;
            clickedIndex = i;
          }
        }
        
        if (clickedIndex !== -1) {
          draggedPoint = { zone: activeZone, index: clickedIndex };
          selectedPoint = { zone: activeZone, index: clickedIndex };
          canvas.style.cursor = 'grabbing';
          redraw();
        } else {
          if (pts.length >= 3 && pointInPolygon(pos.imgX, pos.imgY, pts)) {
            isDraggingPolygon = true;
            dragStartPos = { x: pos.imgX, y: pos.imgY };
            polygonStartPoints = JSON.parse(JSON.stringify(pts));
            canvas.style.cursor = 'move';
          } else {
            selectedPoint = null;
            redraw();
          }
        }
      }
      
      function handleMouseMove(e) {
        if (!isImageLoaded) return;
        const pos = getMousePos(e);
        const canvas = document.getElementById("{prefix}-calib-canvas");
        const rect = canvas.getBoundingClientRect();
        
        if (draggedPoint) {
          const clampedX = Math.max(0, Math.min(img.naturalWidth, pos.imgX));
          const clampedY = Math.max(0, Math.min(img.naturalHeight, pos.imgY));
          zones[draggedPoint.zone][draggedPoint.index] = [clampedX, clampedY];
          redraw(pos.canvasX, pos.canvasY, clampedX, clampedY);
        } else if (isDraggingPolygon) {
          const dx = pos.imgX - dragStartPos.x;
          const dy = pos.imgY - dragStartPos.y;
          const pts = zones[activeZone];
          
          let validMove = true;
          for (let i = 0; i < pts.length; i++) {
            const newX = polygonStartPoints[i][0] + dx;
            const newY = polygonStartPoints[i][1] + dy;
            if (newX < 0 || newX > img.naturalWidth || newY < 0 || newY > img.naturalHeight) {
              validMove = false;
              break;
            }
          }
          
          if (validMove) {
            for (let i = 0; i < pts.length; i++) {
              pts[i] = [polygonStartPoints[i][0] + dx, polygonStartPoints[i][1] + dy];
            }
            redraw();
          }
        } else {
          let isHovering = false;
          const pts = zones[activeZone] || [];
          for (let i = 0; i < pts.length; i++) {
            const pt = pts[i];
            const screenX = (pt[0] / img.naturalWidth) * rect.width;
            const screenY = (pt[1] / img.naturalHeight) * rect.height;
            const dist = Math.sqrt(Math.pow(screenX - pos.canvasX, 2) + Math.pow(screenY - pos.canvasY, 2));
            if (dist < 15) {
              isHovering = true;
              hoveredPoint = { zone: activeZone, index: i };
              break;
            }
          }
          if (isHovering) {
            canvas.style.cursor = 'pointer';
          } else if (pts.length >= 3 && pointInPolygon(pos.imgX, pos.imgY, pts)) {
            canvas.style.cursor = 'grab';
            hoveredPoint = null;
          } else {
            canvas.style.cursor = 'crosshair';
            hoveredPoint = null;
          }
          redraw();
        }
      }
      
      function handleMouseUp(e) {
        if (draggedPoint || isDraggingPolygon) {
          draggedPoint = null;
          isDraggingPolygon = false;
          const canvas = document.getElementById("{prefix}-calib-canvas");
          if (canvas) canvas.style.cursor = 'crosshair';
          saveState();
          redraw();
        }
      }
      
      function handleTouchStart(e) {
        e.preventDefault();
        handleMouseDown(e);
      }
      
      function handleTouchMove(e) {
        e.preventDefault();
        handleMouseMove(e);
      }
      
      function handleTouchEnd(e) {
        e.preventDefault();
        handleMouseUp(e);
      }

      function handleKeyDown(e) {
        if (!selectedPoint || !isImageLoaded) return;
        let step = e.shiftKey ? 10 : 1;
        let dx = 0, dy = 0;
        if (e.key === 'ArrowUp') dy = -step;
        else if (e.key === 'ArrowDown') dy = step;
        else if (e.key === 'ArrowLeft') dx = -step;
        else if (e.key === 'ArrowRight') dx = step;
        else return;
        
        e.preventDefault();
        const pts = zones[selectedPoint.zone];
        const pt = pts[selectedPoint.index];
        pt[0] = Math.max(0, Math.min(img.naturalWidth, pt[0] + dx));
        pt[1] = Math.max(0, Math.min(img.naturalHeight, pt[1] + dy));
        saveState();
        redraw();
      }
      
      function updateCoordinateTable() {
        const panel = document.getElementById("{prefix}-coord-panel");
        if (!panel) return;
        if (!activeZone) {
          panel.innerHTML = `<div style="text-align:center;color:#64748b;font-size:0.85rem;padding:10px;background:rgba(0,0,0,0.2);border-radius:4px;border:1px solid rgba(255,255,255,0.05);">No active zone selected. Enable a zone using the checkboxes above to edit.</div>`;
          return;
        }
        const pts = zones[activeZone] || [];
        let html = `<div style="font-weight:700;margin-bottom:6px;color:#cbd5e1;display:flex;justify-content:space-between;align-items:center;">
                      <span>📍 Coordinates Panel (${activeZone.replace('_', ' ').toUpperCase()})</span>
                      <span style="font-size:0.75rem;background:#1e293b;padding:2px 6px;border-radius:4px;color:#94a3b8;">6-Point Closed Rectangle</span>
                    </div>`;
        html += `<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;font-size:0.8rem;color:#94a3b8;">`;
        for (let i = 0; i < 6; i++) {
          const pt = pts[i] || [0, 0];
          const isSelected = selectedPoint && selectedPoint.zone === activeZone && selectedPoint.index === i;
          const style = isSelected ? "border:1px solid #3b82f6;background:rgba(59,130,246,0.1);color:#60a5fa;" : "border:1px solid rgba(255,255,255,0.08);background:rgba(0,0,0,0.2);";
          html += `<div style="padding:4px 6px;border-radius:4px;${style}text-align:center;">
                     <span style="font-weight:bold;color:#f1f5f9;">#${i+1}:</span> (${pt[0]}, ${pt[1]})
                   </div>`;
        }
        html += `</div>`;
        panel.innerHTML = html;
      }

      function redraw(dragCanvasX, dragCanvasY, dragImgX, dragImgY) {
        const canvas = document.getElementById("{prefix}-calib-canvas");
        if (!canvas || !isImageLoaded) return;
        const ctx = canvas.getContext("2d");
        
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        
        ctx.drawImage(img, 0, 0);
        
        for (const [zoneName, pts] of Object.entries(zones)) {
          if (!pts || pts.length === 0) continue;
          const style = colors[zoneName];
          
          if (pts.length >= 3) {
            ctx.fillStyle = style.fill;
            ctx.beginPath();
            ctx.moveTo(pts[0][0], pts[0][1]);
            for (let i = 1; i < pts.length; i++) {
              ctx.lineTo(pts[i][0], pts[i][1]);
            }
            ctx.closePath();
            ctx.fill();
          }
          
          if (pts.length >= 2) {
            ctx.strokeStyle = style.stroke;
            ctx.lineWidth = Math.max(3, Math.round(img.naturalWidth / 400.0));
            ctx.beginPath();
            ctx.moveTo(pts[0][0], pts[0][1]);
            for (let i = 1; i < pts.length; i++) {
              ctx.lineTo(pts[i][0], pts[i][1]);
            }
            if (pts.length >= 3) {
              ctx.closePath();
            }
            ctx.stroke();
          }
          
          for (let i = 0; i < pts.length; i++) {
            const pt = pts[i];
            const isHovered = hoveredPoint && hoveredPoint.zone === zoneName && hoveredPoint.index === i;
            const isDragged = draggedPoint && draggedPoint.zone === zoneName && draggedPoint.index === i;
            const isSelected = selectedPoint && selectedPoint.zone === zoneName && selectedPoint.index === i;
            
            ctx.fillStyle = style.stroke;
            ctx.beginPath();
            ctx.arc(pt[0], pt[1], Math.max(8, Math.round(img.naturalWidth / 120.0)), 0, Math.PI * 2);
            ctx.fill();
            
            if (isHovered || isDragged || isSelected) {
              ctx.strokeStyle = '#ffffff';
              ctx.lineWidth = Math.max(3, Math.round(img.naturalWidth / 250.0));
              ctx.stroke();
            } else {
              ctx.strokeStyle = 'rgba(255,255,255,0.6)';
              ctx.lineWidth = 1.5;
              ctx.stroke();
            }

            ctx.fillStyle = '#ffffff';
            ctx.font = `bold ${Math.max(10, Math.round(img.naturalWidth / 100.0))}px sans-serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(i + 1, pt[0], pt[1]);
            
            if (i === 0) {
              ctx.fillStyle = '#ffffff';
              ctx.font = `bold ${Math.max(14, Math.round(img.naturalWidth / 60.0))}px sans-serif`;
              ctx.textAlign = 'left';
              ctx.fillText(`${style.label}`, pt[0] + 20, pt[1] - 8);
            }
          }
        }
        
        if (draggedPoint && dragCanvasX !== undefined && dragCanvasY !== undefined) {
          drawLoupe(ctx, canvas, dragCanvasX, dragCanvasY, dragImgX, dragImgY);
        }
      }
      
      function drawLoupe(ctx, canvas, canvasX, canvasY, imgX, imgY) {
        const rect = canvas.getBoundingClientRect();
        const loupeRadius = Math.max(60, Math.round(canvas.width / 15.0));
        const zoom = 5;
        
        let loupeX = loupeRadius + 15;
        let loupeY = loupeRadius + 15;
        
        if (canvasX < rect.width / 2) {
          loupeX = canvas.width - loupeRadius - 15;
        }
        
        ctx.save();
        ctx.beginPath();
        ctx.arc(loupeX, loupeY, loupeRadius, 0, Math.PI * 2);
        ctx.fillStyle = '#090d16';
        ctx.fill();
        
        ctx.beginPath();
        ctx.arc(loupeX, loupeY, loupeRadius, 0, Math.PI * 2);
        ctx.clip();
        
        const sourceSize = (loupeRadius * 2) / zoom;
        const sx = imgX - sourceSize / 2;
        const sy = imgY - sourceSize / 2;
        
        ctx.drawImage(
          img,
          sx, sy, sourceSize, sourceSize,
          loupeX - loupeRadius, loupeY - loupeRadius, loupeRadius * 2, loupeRadius * 2
        );
        
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = Math.max(2, Math.round(canvas.width / 600.0));
        ctx.beginPath();
        ctx.moveTo(loupeX - 15, loupeY); ctx.lineTo(loupeX + 15, loupeY);
        ctx.moveTo(loupeX, loupeY - 15); ctx.lineTo(loupeX, loupeY + 15);
        ctx.stroke();

        ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
        ctx.lineWidth = 0.5;
        const cellSize = zoom;
        for (let x = loupeX - loupeRadius; x < loupeX + loupeRadius; x += cellSize) {
          ctx.beginPath();
          ctx.moveTo(x, loupeY - loupeRadius);
          ctx.lineTo(x, loupeY + loupeRadius);
          ctx.stroke();
        }
        for (let y = loupeY - loupeRadius; y < loupeY + loupeRadius; y += cellSize) {
          ctx.beginPath();
          ctx.moveTo(loupeX - loupeRadius, y);
          ctx.lineTo(loupeX + loupeRadius, y);
          ctx.stroke();
        }
        
        ctx.restore();
        
        ctx.strokeStyle = colors[activeZone].stroke;
        ctx.lineWidth = Math.max(3, Math.round(canvas.width / 400.0));
        ctx.beginPath();
        ctx.arc(loupeX, loupeY, loupeRadius, 0, Math.PI * 2);
        ctx.stroke();
        
        ctx.fillStyle = '#ffffff';
        ctx.font = `bold ${Math.max(12, Math.round(canvas.width / 80.0))}px sans-serif`;
        ctx.textAlign = 'center';
        ctx.fillText(`X: ${imgX}, Y: ${imgY}`, loupeX, loupeY + loupeRadius - 15);
      }
      
      init();
    })();
    </script>
    </div>
    """
    return code.replace("{prefix}", prefix)

def on_save_click_b64(json_str):
    if not json_str:
        return "<div style='color: #ef4444;'>❌ No calibration data to save.</div>", check_calibration_status()
    try:
        import json
        data = json.loads(json_str)
        w = data.get("w") or data.get("frame_width") or 1280
        h = data.get("h") or data.get("frame_height") or 720
        zones_data = data.get("zones", {})
        
        # Save to calibration.json
        save_calibration_data(w, h, zones_data)
        pipeline.load_calibration()
        return "<div style='background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); padding: 12px; border-radius: 8px; margin-bottom: 15px; color: #4ade80;'><strong>✅ Calibration saved successfully to calibration.json!</strong></div>", check_calibration_status()
    except Exception as e:
        return f"<div style='background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); padding: 12px; border-radius: 8px; margin-bottom: 15px; color: #ef4444;'><strong>❌ Error saving calibration:</strong> {str(e)}</div>", check_calibration_status()

def process_traffic_image(image, helmet_model_type, use_custom_line, direction_str):
    if image is None:
        return None, "NO IMAGE UPLOADED", "[]", pd.DataFrame(history_list) if history_list else pd.DataFrame(columns=["Timestamp", "Violation Type", "Details", "License Plate"])
        
    # Dynamic model swap based on user selection
    if "jarvanlee" in helmet_model_type:
        pipeline.helmet_model = YOLO("weights/motorcycle_helmet_yolov8.pt")
    else:
        custom_helmet_path = r"D:\Hackathons\flipkartTraffic\models\helmet_detector.pt"
        if os.path.exists(custom_helmet_path):
            pipeline.helmet_model = YOLO(custom_helmet_path)
        else:
            pipeline.helmet_model = YOLO("weights/helmet_yolov8.pt")
        
    # Save the input image temporary for processing
    temp_input_path = os.path.join("data/raw", "temp_upload.jpg")
    import cv2
    cv2.imwrite(temp_input_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    try:
        direction = "away" if "away" in direction_str.lower() else "towards"
        
        # Run the pipeline (using loaded calibration if use_custom_line is checked)
        metadata = pipeline.process_image(temp_input_path, output_dir="data/processed", custom_line=None, traffic_direction=direction, use_calibration=use_custom_line)
        
        # Load the annotated image (BGR) and convert back to RGB for Gradio
        annotated_bgr = cv2.imread(metadata["processed_image"])
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        
        # Format the violations output as clean JSON
        violations_json = json.dumps(metadata["violations"], indent=2)
        
        # Determine signal emoji
        signal_state = metadata.get("intersection_state", "off").lower()
        if signal_state == "red":
            signal_text = "RED 🔴"
        elif signal_state == "yellow":
            signal_text = "YELLOW 🟡"
        elif signal_state == "green":
            signal_text = "GREEN 🟢"
        else:
            signal_text = "OFF / UNKNOWN ⚫"
        
        signal_status_str = f"Detected Intersection Signal: {signal_text}"
        
        # Add to global history log for the table view
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for viol in metadata["violations"]:
            # Find associated license plate if any
            plate_text = "N/A"
            for det in metadata["detections"]:
                if det["type"] == "license_plate":
                    plate_text = det["text"]
                    break
                    
            history_list.append({
                "Timestamp": timestamp,
                "Violation Type": viol["type"],
                "Details": viol["details"],
                "License Plate": plate_text
            })
            
        history_df = pd.DataFrame(history_list) if history_list else pd.DataFrame(columns=["Timestamp", "Violation Type", "Details", "License Plate"])
        
        return annotated_rgb, signal_status_str, violations_json, history_df
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Error: {str(e)}", "[]", pd.DataFrame(history_list) if history_list else pd.DataFrame(columns=["Timestamp", "Violation Type", "Details", "License Plate"])

def process_traffic_video(video_path, helmet_model_type, use_custom_line, direction_str):
    empty_df = pd.DataFrame(columns=["Timestamp", "Violation Type", "Details", "License Plate"])
    
    if video_path is None:
        yield (
            "Error: No video uploaded.",
            gr.update(visible=False),
            "[]",
            empty_df,
            pd.DataFrame(history_list) if history_list else empty_df,
            gr.update(visible=False),
            gr.update()
        )
        return
        
    # Dynamic model swap based on user selection
    if "jarvanlee" in helmet_model_type:
        pipeline.helmet_model = YOLO("weights/motorcycle_helmet_yolov8.pt")
    else:
        custom_helmet_path = r"D:\Hackathons\flipkartTraffic\models\helmet_detector.pt"
        if os.path.exists(custom_helmet_path):
            pipeline.helmet_model = YOLO(custom_helmet_path)
        else:
            pipeline.helmet_model = YOLO("weights/helmet_yolov8.pt")
        
    logs = ["[*] Starting video analysis pipeline..."]
    global_df = pd.DataFrame(history_list) if history_list else empty_df
    
    yield "\n".join(logs), gr.update(visible=False), "[]", empty_df, global_df, gr.update(visible=False), gr.update()
    
    try:
        direction = "away" if "away" in direction_str.lower() else "towards"
        
        # Run video processing generator (using loaded calibration if use_custom_line is checked)
        res_generator = pipeline.process_video(video_path, output_dir="data/processed", custom_line=None, traffic_direction=direction, use_calibration=use_custom_line)
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for frame_idx, total_frames, data_tuple in res_generator:
            if frame_idx == -1:
                # Video processing and transcoding completed successfully!
                final_res = data_tuple
                break
                
            val, live_frame = data_tuple
            
            # Log progress frame by frame
            msg = f"[*] Processing frame {frame_idx}/{total_frames}... (Found {len(val)} unique violations)"
            logs.append(msg)
            visible_logs = "\n".join(logs[-15:])
            
            # Format current video violations
            current_rows = []
            for v in val:
                current_rows.append({
                    "Timestamp": timestamp,
                    "Violation Type": v["type"],
                    "Details": v["details"],
                    "License Plate": "N/A"
                })
            current_df = pd.DataFrame(current_rows) if current_rows else empty_df
            
            # Convert live frame to base64 to stream to the canvas
            b64_frame = get_base64_from_file(live_frame)
            
            yield visible_logs, gr.update(visible=False), json.dumps(val, indent=2), current_df, global_df, gr.update(value=live_frame, visible=True), b64_frame
            
        # Final yield
        final_violations = final_res["violations"]
        
        # Add to global history log
        for v in final_violations:
            history_list.append({
                "Timestamp": timestamp,
                "Violation Type": v["type"],
                "Details": v["details"],
                "License Plate": "N/A"
            })
            
        final_rows = []
        for v in final_violations:
            final_rows.append({
                "Timestamp": timestamp,
                "Violation Type": v["type"],
                "Details": v["details"],
                "License Plate": "N/A"
            })
        final_df = pd.DataFrame(final_rows) if final_rows else empty_df
        updated_global_df = pd.DataFrame(history_list)
        
        logs.append("[+] Analysis and H.264 transcoding complete!")
        logs.append(f"[+] Output saved to: {final_res['processed_video']}")
        visible_logs = "\n".join(logs[-15:])
        
        yield visible_logs, gr.update(value=final_res["processed_video"], visible=True), json.dumps(final_violations, indent=2), final_df, updated_global_df, gr.update(value=None, visible=False), gr.update()
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        logs.append(f"[-] Error: {str(e)}")
        yield (
            "\n".join(logs[-15:]),
            gr.update(visible=False),
            "[]",
            empty_df,
            pd.DataFrame(history_list) if history_list else empty_df,
            gr.update(visible=False),
            gr.update()
        )


# Custom CSS for state-of-the-art dark premium dashboard
css = """
body, .gradio-container {
    background-color: #0b0f19 !important;
    color: #f3f4f6 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}
.contain {
    border-radius: 16px !important;
    background: rgba(17, 24, 39, 0.5) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4) !important;
    padding: 20px !important;
    margin-bottom: 16px !important;
}
.tabs {
    border-bottom: 2px solid rgba(255, 255, 255, 0.08) !important;
}
.tab-nav button {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: #9ca3af !important;
    padding: 10px 20px !important;
    transition: all 0.2s ease !important;
}
.tab-nav button.selected {
    color: #3b82f6 !important;
    border-bottom: 3px solid #3b82f6 !important;
}
button.primary {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4) !important;
    transition: all 0.3s ease !important;
}
button.primary:hover {
    transform: translateY(-1.5px) !important;
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.6) !important;
}
input[type="range"] {
    accent-color: #3b82f6 !important;
}
.gr-form {
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    background: rgba(31, 41, 55, 0.3) !important;
}
h1 {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(to right, #60a5fa, #3b82f6, #1d4ed8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px !important;
}
img {
    -webkit-user-drag: none !important;
    user-select: none !important;
    -moz-user-select: none !important;
    -ms-user-select: none !important;
}
.hidden-textbox {
    display: none !important;
}

/* Hide calibrator groups by default */
#video-calibrator-group,
#image-calibrator-group {
    display: none !important;
}
"""

# Build the Gradio UI
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🚦 Automated Traffic Violation Detection & Classification System
        ### Computer Vision Hackathon Prototype (RTX 4050 Optimized)
        
        Upload traffic photos or videos to automatically detect vehicles, helmets, triple riding, license plates, and red light violations.
        """
    )
    
    calibration_status = gr.HTML(value=check_calibration_status())
    
    with gr.Tab("Single Image Violations"):
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="numpy", label="Upload Traffic Photo")
                with gr.Group():
                    helmet_model_dropdown = gr.Dropdown(
                        choices=["Custom flipkartTraffic (YOLOv8-Medium)", "jarvanlee (YOLOv8-Medium)"],
                        value="Custom flipkartTraffic (YOLOv8-Medium)",
                        label="Helmet Detection Model"
                    )
                    use_custom_line_image = gr.Checkbox(label="Use Custom Stop Zone Calibration", value=True)
                    direction_image = gr.Radio(choices=["Towards Camera", "Away from Camera"], value="Towards Camera", label="Traffic Flow Direction")
                
                setup_calib_img_btn = gr.Button("🔧 Setup Stop Line / Calibration", variant="secondary")
                process_btn = gr.Button("🚀 Analyze Image", variant="primary")
            
            with gr.Column(scale=1):
                # Normal image output preview group (visible by default)
                with gr.Group(elem_id="image-preview-group") as image_preview_group:
                    image_output = gr.Image(type="numpy", label="Annotated Evidence Output / Calibration Preview", interactive=False)
                    signal_status_output = gr.Textbox(label="Detected Signal State", interactive=False)
                
                # Image Calibrator Canvas group (hidden by default, always in DOM)
                with gr.Group(elem_id="image-calibrator-group") as image_calibrator_group:
                    # Hidden communication elements
                    image_calib_image_src = gr.Textbox(elem_id="img-calib-image-src", elem_classes=["hidden-textbox"], value="")
                    image_calib_json_input = gr.Textbox(elem_id="img-calib-json-input", elem_classes=["hidden-textbox"], value=load_initial_calib_json())
                    img_calibrator_html = gr.HTML(value=get_calibrator_html_code("img"))
                    with gr.Row():
                        save_img_calib_btn = gr.Button("💾 Save Calibration", variant="primary")
                        close_img_calib_btn = gr.Button("❌ Close Calibration Panel", variant="secondary")
                    img_save_status = gr.HTML(value="")

        with gr.Row():
            violations_output = gr.Code(label="Flagged Violations (JSON)", language="json")
            
    with gr.Tab("Video Violations"):
        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="Upload Traffic Video", elem_id="video-input")
                # Hidden element to store current player time before toggling
                video_calib_timestamp = gr.Textbox(elem_id="video-calib-timestamp", elem_classes=["hidden-textbox"], value="0.0")
                with gr.Group():
                    video_helmet_dropdown = gr.Dropdown(
                        choices=["Custom flipkartTraffic (YOLOv8-Medium)", "jarvanlee (YOLOv8-Medium)"],
                        value="Custom flipkartTraffic (YOLOv8-Medium)",
                        label="Helmet Detection Model"
                    )
                    use_custom_line_video = gr.Checkbox(label="Use Custom Stop Zone Calibration", value=True)
                    direction_video = gr.Radio(choices=["Towards Camera", "Away from Camera"], value="Away from Camera", label="Traffic Flow Direction")
                
                setup_calib_vid_btn = gr.Button("🔧 Setup Stop Line / Calibration", variant="secondary")
                video_process_btn = gr.Button("🚀 Analyze Video", variant="primary")
                
            with gr.Column(scale=1):
                # Normal video output preview group (visible by default)
                with gr.Group(elem_id="video-preview-group") as video_preview_group:
                    live_frame_output = gr.Image(label="Live Feed / Calibration Line Preview", type="numpy", interactive=False)
                    video_output = gr.Video(label="Final Processed Video (H.264 Playback)", visible=False)
                    video_log_output = gr.Textbox(label="Processing Log / Status", lines=6, max_lines=8, interactive=False)

                # Video Calibrator Canvas group (hidden by default, always in DOM)
                with gr.Group(elem_id="video-calibrator-group") as video_calibrator_group:
                    # Hidden communication elements
                    video_calib_image_src = gr.Textbox(elem_id="vid-calib-image-src", elem_classes=["hidden-textbox"], value="")
                    video_calib_json_input = gr.Textbox(elem_id="vid-calib-json-input", elem_classes=["hidden-textbox"], value=load_initial_calib_json())
                    vid_calibrator_html = gr.HTML(value=get_calibrator_html_code("vid"))
                    vid_calib_slider = gr.Slider(minimum=0.0, maximum=10.0, step=0.1, value=0.0, label="Select Reference Frame (Seconds)")
                    with gr.Row():
                        save_vid_calib_btn = gr.Button("💾 Save Calibration", variant="primary")
                        close_vid_calib_btn = gr.Button("❌ Close Calibration Panel", variant="secondary")
                    vid_save_status = gr.HTML(value="")
                
        with gr.Row():
            video_violations_df = gr.Dataframe(
                headers=["Timestamp", "Violation Type", "Details", "License Plate"],
                datatype=["str", "str", "str", "str"],
                label="Live Enforcement Violation Feed (Current Video)",
                interactive=False
            )
            video_violations_output = gr.Code(label="Flagged Violations (JSON)", language="json")
            
    with gr.Tab("Historical Violations Log"):
        history_table = gr.Dataframe(
            headers=["Timestamp", "Violation Type", "Details", "License Plate"],
            datatype=["str", "str", "str", "str"],
            label="All Logged Enforcement Violations",
            interactive=False
        )
 
    # Bind preview updates for Image tab
    image_input.change(
        fn=update_image_preview,
        inputs=[image_input, use_custom_line_image],
        outputs=image_output
    )
    use_custom_line_image.change(
        fn=update_image_preview,
        inputs=[image_input, use_custom_line_image],
        outputs=image_output
    )

    # Bind preview updates for Video tab
    video_input.change(
        fn=update_video_preview,
        inputs=[video_input, use_custom_line_video],
        outputs=[live_frame_output, video_output]
    )
    use_custom_line_video.change(
        fn=update_video_preview,
        inputs=[video_input, use_custom_line_video],
        outputs=[live_frame_output, video_output]
    )

    # Image Calibration trigger and save (uses client-side JS toggle)
    setup_calib_img_btn.click(
        fn=toggle_image_calibration,
        inputs=[image_input],
        outputs=[image_calib_image_src],
        js="() => { document.body.classList.add('image-calib-active'); }"
    )
    close_img_calib_btn.click(
        fn=None,
        inputs=[],
        outputs=[],
        js="() => { document.body.classList.remove('image-calib-active'); }"
    )
    save_img_calib_btn.click(
        fn=on_save_click_b64,
        inputs=[image_calib_json_input],
        outputs=[img_save_status, calibration_status]
    )

    # Video Calibration trigger, slider release, and save (uses client-side JS toggle)
    setup_calib_vid_btn.click(
        fn=toggle_video_calibration,
        inputs=[video_input, video_calib_timestamp],
        outputs=[vid_calib_slider, video_calib_image_src],
        js="() => { const videoEl = document.querySelector('#video-input video'); const tsInput = document.querySelector('#video-calib-timestamp textarea, #video-calib-timestamp input'); if (videoEl && tsInput) { tsInput.value = videoEl.currentTime.toString(); tsInput.dispatchEvent(new Event('input', { bubbles: true })); } document.body.classList.add('video-calib-active'); }"
    )
    video_calib_timestamp.change(
        fn=handle_timestamp_change,
        inputs=[video_input, video_calib_timestamp],
        outputs=[video_calib_image_src, vid_calib_slider]
    )
    close_vid_calib_btn.click(
        fn=None,
        inputs=[],
        outputs=[],
        js="() => { document.body.classList.remove('video-calib-active'); }"
    )
    vid_calib_slider.release(
        fn=None,
        inputs=[vid_calib_slider],
        outputs=[],
        js="(slider_val) => { const videoEl = document.querySelector('#video-input video'); if (videoEl) { videoEl.currentTime = parseFloat(slider_val); } }"
    )
    save_vid_calib_btn.click(
        fn=on_save_click_b64,
        inputs=[video_calib_json_input],
        outputs=[vid_save_status, calibration_status]
    )

    # Bind the image analyze button
    process_btn.click(
        fn=process_traffic_image,
        inputs=[image_input, helmet_model_dropdown, use_custom_line_image, direction_image],
        outputs=[image_output, signal_status_output, violations_output, history_table]
    )
    
    # Bind the video analyze button
    video_process_btn.click(
        fn=process_traffic_video,
        inputs=[video_input, video_helmet_dropdown, use_custom_line_video, direction_video],
        outputs=[video_log_output, video_output, video_violations_output, video_violations_df, history_table, live_frame_output, video_calib_image_src]
    )

    # Script execution enabler for dynamically inserted HTML
    observer_js = """
    () => {
        const evaluateScripts = () => {
            document.querySelectorAll(".img-calib-container script, .vid-calib-container script").forEach((script) => {
                if (!script.dataset.evaluated) {
                    script.dataset.evaluated = "true";
                    const newScript = document.createElement("script");
                    newScript.textContent = script.textContent;
                    newScript.dataset.evaluated = "true";
                    document.head.appendChild(newScript);
                }
            });
        };
        // Run once immediately for already rendered elements
        evaluateScripts();
        // Observe for any dynamically added elements
        const observer = new MutationObserver(evaluateScripts);
        observer.observe(document.body, { childList: true, subtree: true });
    }
    """
    demo.load(fn=None, inputs=None, outputs=None, js=observer_js)
    
    gr.Markdown(
        """
        ---
        **Hackathon Deployment Tip:** To generate a public shareable URL for judges, run this script with `demo.queue().launch(share=True)`.
        """
    )

if __name__ == "__main__":
    fixed_port = 61634
    print(f"[*] Starting Gradio server on fixed port: {fixed_port}")
    demo.queue().launch(server_name="127.0.0.1", server_port=fixed_port, theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"), css=css, share=False)
