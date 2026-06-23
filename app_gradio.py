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


def get_base64_from_file(file_path):
    import base64
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
        _, buffer = cv2.imencode(".jpg", img_array)
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

def load_initial_calib_base64():
    import os
    if os.path.exists(CALIB_REF_PATH):
        return get_base64_from_file(CALIB_REF_PATH)
    return ""

def load_initial_calib_json():
    import os
    import json
    if os.path.exists("calibration.json"):
        try:
            with open("calibration.json", "r") as f:
                data = json.load(f)
            return json.dumps(data)
        except Exception as e:
            print(f"[-] Error loading initial JSON calibration: {e}")
    return json.dumps({"w": 1280, "h": 720, "zones": {"stop_line": [], "exit_line": [], "signal_roi": []}})

def handle_calib_image_upload(image):
    if image is None:
        return ""
    save_calib_reference(image)
    return get_base64_from_file(CALIB_REF_PATH)

def handle_calib_video_upload(video_path):
    if video_path is None:
        return ""
    frame = get_video_first_frame(video_path)
    if frame is not None:
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        save_calib_reference(rgb)
        return get_base64_from_file(CALIB_REF_PATH)
    return ""

def on_calib_json_change(json_str):
    if not json_str:
        return check_calibration_status()
    try:
        data = json.loads(json_str)
        w = data.get("w", 1280)
        h = data.get("h", 720)
        zones_data = data.get("zones", {})
        save_calibration_data(w, h, zones_data)
        pipeline.load_calibration()
    except Exception as e:
        print(f"[-] Error saving JSON calibration: {e}")
    return check_calibration_status()

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

def get_calibrator_html_code():
    return """
    <style>
    .calib-container {
      display: flex;
      flex-direction: column;
      gap: 15px;
      background: rgba(15, 23, 42, 0.4);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 12px;
      padding: 16px;
      backdrop-filter: blur(10px);
    }
    .calib-header {
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      padding-bottom: 10px;
    }
    .calib-title {
      font-size: 1.1rem;
      font-weight: 600;
      color: #f1f5f9;
    }
    .calib-panel {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
    }
    .calib-control-group {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .control-label {
      font-size: 0.85rem;
      color: #94a3b8;
      font-weight: 500;
    }
    .zone-pills {
      display: flex;
      gap: 6px;
      background: rgba(0, 0, 0, 0.2);
      padding: 3px;
      border-radius: 8px;
    }
    .zone-pill {
      border: none;
      background: transparent;
      color: #94a3b8;
      padding: 6px 12px;
      font-size: 0.85rem;
      font-weight: 600;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.2s;
    }
    .zone-pill.active.stop-line {
      background: rgba(239, 68, 68, 0.2);
      color: #ef4444;
      border: 1px solid rgba(239, 68, 68, 0.4);
    }
    .zone-pill.active.exit-line {
      background: rgba(249, 115, 22, 0.2);
      color: #f97316;
      border: 1px solid rgba(249, 115, 22, 0.4);
    }
    .zone-pill.active.signal-roi {
      background: rgba(34, 197, 94, 0.2);
      color: #22c55e;
      border: 1px solid rgba(34, 197, 94, 0.4);
    }
    .action-btns {
      display: flex;
      gap: 6px;
    }
    .action-btn {
      background: rgba(30, 41, 59, 0.6);
      border: 1px solid rgba(255, 255, 255, 0.05);
      color: #cbd5e1;
      padding: 6px 12px;
      font-size: 0.8rem;
      font-weight: 500;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.2s;
    }
    .action-btn:hover {
      background: rgba(51, 65, 85, 0.8);
      color: #ffffff;
    }
    .canvas-container {
      position: relative;
      width: 100%;
      aspect-ratio: 16 / 9;
      background: #090d16;
      border-radius: 8px;
      border: 1px solid rgba(255, 255, 255, 0.05);
      overflow: hidden;
      display: flex;
      justify-content: center;
      align-items: center;
    }
    #calib-canvas {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: none;
      cursor: crosshair;
    }
    .placeholder-msg {
      color: #64748b;
      font-size: 0.9rem;
      text-align: center;
      line-height: 1.5;
    }
    </style>

    <div class="calib-container">
      <div class="calib-header">
        <span class="calib-title">📸 Interactive Camera Zone Calibrator</span>
      </div>
      
      <div class="calib-panel">
        <div class="calib-control-group">
          <span class="control-label">Select Active Zone:</span>
          <div class="zone-pills">
            <button id="btn-stop-line" class="zone-pill active stop-line" onclick="setZone('stop_line')">🟥 Stop Line</button>
            <button id="btn-exit-line" class="zone-pill exit-line" onclick="setZone('exit_line')">🟧 Exit Zone</button>
            <button id="btn-signal-roi" class="zone-pill signal-roi" onclick="setZone('signal_roi')">🟩 Signal ROI</button>
          </div>
        </div>
        
        <div class="calib-control-group">
          <div class="action-btns">
            <button class="action-btn undo-btn" onclick="undoPoint()">⏪ Undo Point</button>
            <button class="action-btn clear-btn" onclick="clearZone()">❌ Clear Active Zone</button>
            <button class="action-btn reset-btn" onclick="resetAllZones()">🔄 Reset All Zones</button>
          </div>
        </div>
      </div>

      <div class="canvas-container">
        <canvas id="calib-canvas"></canvas>
        <div id="no-image-msg" class="placeholder-msg">
          No calibration reference frame loaded.<br>Please upload an image or video in the left panel to begin.
        </div>
      </div>
    </div>

    <script>
    (function() {
      let zones = { stop_line: [], exit_line: [], signal_roi: [] };
      let activeZone = 'stop_line';
      let draggedPoint = null;
      let hoveredPoint = null;
      let img = new Image();
      let isImageLoaded = false;
      
      const colors = {
        stop_line: { stroke: '#ef4444', fill: 'rgba(239, 68, 68, 0.25)', label: 'STOP LINE ZONE' },
        exit_line: { stroke: '#f97316', fill: 'rgba(249, 115, 22, 0.25)', label: 'EXIT ZONE' },
        signal_roi: { stroke: '#22c55e', fill: 'rgba(34, 197, 94, 0.15)', label: 'SIGNAL ROI' }
      };
      
      function init() {
        const canvas = document.getElementById("calib-canvas");
        const placeholder = document.getElementById("no-image-msg");
        if (!canvas || !placeholder) {
          setTimeout(init, 100);
          return;
        }
        
        canvas.addEventListener('mousedown', handleMouseDown);
        canvas.addEventListener('mousemove', handleMouseMove);
        window.addEventListener('mouseup', handleMouseUp);
        
        canvas.addEventListener('touchstart', handleTouchStart, { passive: false });
        canvas.addEventListener('touchmove', handleTouchMove, { passive: false });
        canvas.addEventListener('touchend', handleTouchEnd, { passive: false });
        
        let lastSrc = "";
        setInterval(() => {
          const srcTextarea = document.querySelector("#calib-image-src textarea");
          if (srcTextarea && srcTextarea.value !== lastSrc) {
            lastSrc = srcTextarea.value;
            if (lastSrc) {
              img.src = lastSrc;
              img.onload = function() {
                isImageLoaded = true;
                canvas.style.display = "block";
                placeholder.style.display = "none";
                redraw();
              };
            } else {
              isImageLoaded = false;
              canvas.style.display = "none";
              placeholder.style.display = "block";
            }
          }
        }, 250);
        
        let lastJson = "";
        setInterval(() => {
          const jsonTextarea = document.querySelector("#calib-json-input textarea");
          if (jsonTextarea && jsonTextarea.value !== lastJson) {
            lastJson = jsonTextarea.value;
            if (lastJson) {
              try {
                const data = JSON.parse(lastJson);
                if (data && data.zones) {
                  zones = data.zones;
                  redraw();
                }
              } catch(e) {
                console.error("Error parsing JSON:", e);
              }
            }
          }
        }, 500);
      }
      
      window.setZone = function(zoneName) {
        activeZone = zoneName;
        document.querySelectorAll('.zone-pill').forEach(btn => btn.classList.remove('active'));
        const activeBtn = document.getElementById(`btn-${zoneName.replace('_', '-')}`);
        if (activeBtn) activeBtn.classList.add('active');
        redraw();
      };
      
      window.undoPoint = function() {
        if (zones[activeZone] && zones[activeZone].length > 0) {
          zones[activeZone].pop();
          saveState();
          redraw();
        }
      };
      
      window.clearZone = function() {
        if (zones[activeZone]) {
          zones[activeZone] = [];
          saveState();
          redraw();
        }
      };
      
      window.resetAllZones = function() {
        if (confirm("Are you sure you want to reset all calibrated zones?")) {
          zones = { stop_line: [], exit_line: [], signal_roi: [] };
          saveState();
          redraw();
        }
      };
      
      function saveState() {
        const jsonTextarea = document.querySelector("#calib-json-input textarea");
        if (jsonTextarea) {
          const payload = {
            w: img.naturalWidth || 1280,
            h: img.naturalHeight || 720,
            zones: zones
          };
          jsonTextarea.value = JSON.stringify(payload);
          jsonTextarea.dispatchEvent(new Event("input", { bubbles: true }));
        }
      }
      
      function getMousePos(e) {
        const canvas = document.getElementById("calib-canvas");
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
        
        const canvas = document.getElementById("calib-canvas");
        const rect = canvas.getBoundingClientRect();
        let clickedIndex = -1;
        let minDist = 12;
        
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
          canvas.style.cursor = 'grabbing';
        } else {
          if (pts.length < 10) {
            zones[activeZone].push([pos.imgX, pos.imgY]);
            saveState();
            redraw();
            draggedPoint = { zone: activeZone, index: zones[activeZone].length - 1 };
            canvas.style.cursor = 'grabbing';
          }
        }
      }
      
      function handleMouseMove(e) {
        if (!isImageLoaded) return;
        const pos = getMousePos(e);
        const canvas = document.getElementById("calib-canvas");
        const rect = canvas.getBoundingClientRect();
        
        if (draggedPoint) {
          const clampedX = Math.max(0, Math.min(img.naturalWidth, pos.imgX));
          const clampedY = Math.max(0, Math.min(img.naturalHeight, pos.imgY));
          zones[draggedPoint.zone][draggedPoint.index] = [clampedX, clampedY];
          redraw(pos.canvasX, pos.canvasY, clampedX, clampedY);
        } else {
          let isHovering = false;
          const pts = zones[activeZone] || [];
          for (let i = 0; i < pts.length; i++) {
            const pt = pts[i];
            const screenX = (pt[0] / img.naturalWidth) * rect.width;
            const screenY = (pt[1] / img.naturalHeight) * rect.height;
            const dist = Math.sqrt(Math.pow(screenX - pos.canvasX, 2) + Math.pow(screenY - pos.canvasY, 2));
            if (dist < 12) {
              isHovering = true;
              hoveredPoint = { zone: activeZone, index: i };
              break;
            }
          }
          if (isHovering) {
            canvas.style.cursor = 'pointer';
          } else {
            canvas.style.cursor = 'crosshair';
            hoveredPoint = null;
          }
          redraw();
        }
      }
      
      function handleMouseUp(e) {
        if (draggedPoint) {
          draggedPoint = null;
          const canvas = document.getElementById("calib-canvas");
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
      
      function redraw(dragCanvasX, dragCanvasY, dragImgX, dragImgY) {
        const canvas = document.getElementById("calib-canvas");
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
            
            ctx.fillStyle = style.stroke;
            ctx.beginPath();
            ctx.arc(pt[0], pt[1], Math.max(6, Math.round(img.naturalWidth / 150.0)), 0, Math.PI * 2);
            ctx.fill();
            
            if (isHovered || isDragged) {
              ctx.strokeStyle = '#ffffff';
              ctx.lineWidth = Math.max(2, Math.round(img.naturalWidth / 300.0));
              ctx.stroke();
            }
            
            if (i === 0) {
              ctx.fillStyle = '#ffffff';
              ctx.font = `bold ${Math.max(14, Math.round(img.naturalWidth / 60.0))}px sans-serif`;
              ctx.fillText(style.label, pt[0] + 15, pt[1] - 8);
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
    """

def process_traffic_image(image, helmet_model_type, use_custom_line, direction_str):
    if image is None:
        return None, "NO IMAGE UPLOADED", "[]", pd.DataFrame(history_list) if history_list else pd.DataFrame(columns=["Timestamp", "Violation Type", "Details", "License Plate"])
        
    # Dynamic model swap based on user selection
    if "jarvanlee" in helmet_model_type:
        pipeline.helmet_model = YOLO("weights/motorcycle_helmet_yolov8.pt")
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
            gr.update(visible=False)
        )
        return
        
    # Dynamic model swap based on user selection
    if "jarvanlee" in helmet_model_type:
        pipeline.helmet_model = YOLO("weights/motorcycle_helmet_yolov8.pt")
    else:
        pipeline.helmet_model = YOLO("weights/helmet_yolov8.pt")
        
    logs = ["[*] Starting video analysis pipeline..."]
    global_df = pd.DataFrame(history_list) if history_list else empty_df
    
    yield "\n".join(logs), gr.update(visible=False), "[]", empty_df, global_df, gr.update(visible=False)
    
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
            
            yield visible_logs, gr.update(visible=False), json.dumps(val, indent=2), current_df, global_df, gr.update(value=live_frame, visible=True)
            
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
        
        yield visible_logs, gr.update(value=final_res["processed_video"], visible=True), json.dumps(final_violations, indent=2), final_df, updated_global_df, gr.update(value=None, visible=False)
        
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
            gr.update(visible=False)
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
"""

# Build the Gradio UI
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"), css=css) as demo:
    gr.Markdown(
        """
        # 🚦 Automated Traffic Violation Detection & Classification System
        ### Computer Vision Hackathon Prototype (RTX 4050 Optimized)
        
        Upload traffic photos or videos to automatically detect vehicles, helmets, triple riding, license plates, and red light violations.
        """
    )
    
    session_calibration = gr.State(value=load_session_calibration)
    
    calibration_status = gr.HTML(value=check_calibration_status())
    
    with gr.Tab("Camera Zone Calibration"):
        with gr.Row():
            with gr.Column(scale=1):
                calib_image_input = gr.Image(type="numpy", label="Upload Reference Photo (Option A)")
                calib_video_input = gr.Video(label="Upload Reference Video (Option B)")
                gr.Markdown(
                    """
                    ### 🚦 Calibration Guide:
                    1. **Stop Line Zone** (Red): Draw 4 points defining the stop-line area.
                    2. **Exit Zone** (Orange): Draw 4 points further down the lane.
                    3. **Signal ROI** (Green): Draw a box around the traffic signal light.
                    
                    *Drag points directly to adjust. Click near a point to drag it. Click elsewhere to add a point.*
                    """
                )
            with gr.Column(scale=2):
                # Hidden communication elements
                calib_image_src = gr.Textbox(visible=False, elem_id="calib-image-src", value=load_initial_calib_base64())
                calib_json_input = gr.Textbox(visible=False, elem_id="calib-json-input", value=load_initial_calib_json())
                
                # HTML5 Calibrator Component
                calibrator_html = gr.HTML(value=get_calibrator_html_code())
                
    with gr.Tab("Single Image Violations"):
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="numpy", label="Upload Traffic Photo")
                with gr.Group():
                    helmet_model_dropdown = gr.Dropdown(
                        choices=["iam-tsr (YOLOv8-Nano)", "jarvanlee (YOLOv8-Medium)"],
                        value="iam-tsr (YOLOv8-Nano)",
                        label="Helmet Detection Model"
                    )
                    use_custom_line_image = gr.Checkbox(label="Use Custom Stop Zone Calibration", value=True)
                    direction_image = gr.Radio(choices=["Towards Camera", "Away from Camera"], value="Towards Camera", label="Traffic Flow Direction")
                process_btn = gr.Button("🚀 Analyze Image", variant="primary")
            
            with gr.Column(scale=1):
                image_output = gr.Image(type="numpy", label="Annotated Evidence Output / Calibration Preview", interactive=False)
                signal_status_output = gr.Textbox(label="Detected Signal State", interactive=False)
                
        with gr.Row():
            violations_output = gr.Code(label="Flagged Violations (JSON)", language="json")
            
    with gr.Tab("Video Violations"):
        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="Upload Traffic Video")
                with gr.Group():
                    video_helmet_dropdown = gr.Dropdown(
                        choices=["iam-tsr (YOLOv8-Nano)", "jarvanlee (YOLOv8-Medium)"],
                        value="iam-tsr (YOLOv8-Nano)",
                        label="Helmet Detection Model"
                    )
                    use_custom_line_video = gr.Checkbox(label="Use Custom Stop Zone Calibration", value=True)
                    direction_video = gr.Radio(choices=["Towards Camera", "Away from Camera"], value="Away from Camera", label="Traffic Flow Direction")
                video_process_btn = gr.Button("🚀 Analyze Video", variant="primary")
                
            with gr.Column(scale=1):
                live_frame_output = gr.Image(label="Live Feed / Calibration Line Preview", type="numpy", interactive=False)
                video_output = gr.Video(label="Final Processed Video (H.264 Playback)", visible=False)
                video_log_output = gr.Textbox(label="Processing Log / Status", lines=6, max_lines=8, interactive=False)
                
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

    # Bind calibration tab upload events
    calib_image_input.change(
        fn=handle_calib_image_upload,
        inputs=calib_image_input,
        outputs=calib_image_src
    )
    calib_video_input.change(
        fn=handle_calib_video_upload,
        inputs=calib_video_input,
        outputs=calib_image_src
    )
    
    # Bind calibration JSON sync event
    calib_json_input.change(
        fn=on_calib_json_change,
        inputs=calib_json_input,
        outputs=calibration_status
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
        outputs=[video_log_output, video_output, video_violations_output, video_violations_df, history_table, live_frame_output]
    )
    
    gr.Markdown(
        """
        ---
        **Hackathon Deployment Tip:** To generate a public shareable URL for judges, run this script with `demo.queue().launch(share=True)`.
        """
    )

if __name__ == "__main__":
    fixed_port = 61634
    print(f"[*] Starting Gradio server on fixed port: {fixed_port}")
    demo.queue().launch(server_name="127.0.0.1", server_port=fixed_port, share=False)
