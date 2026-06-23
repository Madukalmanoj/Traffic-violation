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

def update_video_preview(video_path, use_custom, left_x, left_y, right_x, right_y):
    if video_path is None:
        return gr.update(value=None, visible=False), gr.update(visible=False)
    import cv2
    try:
        frame = get_video_first_frame(video_path)
        if frame is None:
            return gr.update(value=None, visible=False), gr.update(visible=False)
        h, w = frame.shape[:2]
        # Draw the translucent stop zone overlay
        x_min = int(w * left_x)
        y_min = int(h * left_y)
        x_max = int(w * right_x)
        y_max = int(h * right_y)
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 0, 255), -1) # Red overlay (BGR (0,0,255))
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3) # Outline
        
        cv2.putText(frame, f"STOP ZONE PREVIEW: Min=({int(left_x*100)}%,{int(left_y*100)}%) Max=({int(right_x*100)}%,{int(right_y*100)}%)", 
                    (x_min + 15, y_min - 10 if y_min > 30 else y_min + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        preview_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return gr.update(value=preview_rgb, visible=True), gr.update(visible=False)
    except Exception as e:
        print(f"[-] update_video_preview failed: {e}")
        return gr.update(value=None, visible=False), gr.update(visible=False)

def update_image_preview(image, use_custom, left_x, left_y, right_x, right_y):
    if image is None:
        return None
    import cv2
    try:
        frame = image.copy()
        h, w = frame.shape[:2]
        # Draw the translucent stop zone overlay (image is in RGB from Gradio)
        x_min = int(w * left_x)
        y_min = int(h * left_y)
        x_max = int(w * right_x)
        y_max = int(h * right_y)
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (255, 0, 0), -1) # Red overlay (RGB (255,0,0))
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3) # Outline
        
        cv2.putText(frame, f"STOP ZONE PREVIEW: Min=({int(left_x*100)}%,{int(left_y*100)}%) Max=({int(right_x*100)}%,{int(right_y*100)}%)", 
                    (x_min + 15, y_min - 10 if y_min > 30 else y_min + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        return frame
    except Exception as e:
        print(f"[-] update_image_preview failed: {e}")
        return None

def process_traffic_image(image, helmet_model_type, use_custom_line, left_x, left_y, right_x, right_y, direction_str):
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
        custom_line = (left_x, left_y, right_x, right_y) if use_custom_line else None
        direction = "away" if "away" in direction_str.lower() else "towards"
        
        # Run the pipeline
        metadata = pipeline.process_image(temp_input_path, output_dir="data/processed", custom_line=custom_line, traffic_direction=direction)
        
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


def process_traffic_video(video_path, helmet_model_type, use_custom_line, left_x, left_y, right_x, right_y, direction_str):
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
        custom_line = (left_x, left_y, right_x, right_y) if use_custom_line else None
        direction = "away" if "away" in direction_str.lower() else "towards"
        
        # Run video processing generator
        res_generator = pipeline.process_video(video_path, output_dir="data/processed", custom_line=custom_line, traffic_direction=direction)
        
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
"""

# Build the Gradio UI
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"), css=css) as demo:
    gr.Markdown(
        """
        # 🚦 Automated Traffic Violation Detection & Classification System
        ### Computer Vision Hackathon Prototype (RTX 4050 Optimized)
        
        Upload traffic photos or videos to automatically detect vehicles, helmets, triple riding, license plates, and red light stop-line crossing violations.
        """
    )
    
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
                    with gr.Row():
                        custom_line_left_x_image = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.01, label="Zone Left X (%)")
                        custom_line_left_y_image = gr.Slider(minimum=0.0, maximum=1.0, value=0.40, step=0.01, label="Zone Top Y (%)")
                    with gr.Row():
                        custom_line_right_x_image = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, step=0.01, label="Zone Right X (%)")
                        custom_line_right_y_image = gr.Slider(minimum=0.0, maximum=1.0, value=0.80, step=0.01, label="Zone Bottom Y (%)")
                    direction_image = gr.Radio(choices=["Towards Camera", "Away from Camera"], value="Towards Camera", label="Traffic Flow Direction")
                process_btn = gr.Button("🚀 Analyze Image", variant="primary")
            
            with gr.Column(scale=1):
                image_output = gr.Image(type="numpy", label="Annotated Evidence Output / Calibration Preview")
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
                    with gr.Row():
                        custom_line_left_x_video = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.01, label="Zone Left X (%)")
                        custom_line_left_y_video = gr.Slider(minimum=0.0, maximum=1.0, value=0.40, step=0.01, label="Zone Top Y (%)")
                    with gr.Row():
                        custom_line_right_x_video = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, step=0.01, label="Zone Right X (%)")
                        custom_line_right_y_video = gr.Slider(minimum=0.0, maximum=1.0, value=0.80, step=0.01, label="Zone Bottom Y (%)")
                    direction_video = gr.Radio(choices=["Towards Camera", "Away from Camera"], value="Away from Camera", label="Traffic Flow Direction")
                video_process_btn = gr.Button("🚀 Analyze Video", variant="primary")
                
            with gr.Column(scale=1):
                live_frame_output = gr.Image(label="Live Feed / Calibration Line Preview", interactive=False)
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
 
    # Bind preview updates for Image tab (smooth real-time drag-and-drop feedback)
    image_preview_events = [
        use_custom_line_image.change,
        custom_line_left_x_image.change, custom_line_left_x_image.input,
        custom_line_left_y_image.change, custom_line_left_y_image.input,
        custom_line_right_x_image.change, custom_line_right_x_image.input,
        custom_line_right_y_image.change, custom_line_right_y_image.input,
        image_input.change
    ]
    for event in image_preview_events:
        event(
            fn=update_image_preview,
            inputs=[
                image_input, use_custom_line_image,
                custom_line_left_x_image, custom_line_left_y_image,
                custom_line_right_x_image, custom_line_right_y_image
            ],
            outputs=image_output
        )

    # Bind preview updates for Video tab (live preview overlay in place)
    video_preview_events = [
        use_custom_line_video.change,
        custom_line_left_x_video.change, custom_line_left_x_video.input,
        custom_line_left_y_video.change, custom_line_left_y_video.input,
        custom_line_right_x_video.change, custom_line_right_x_video.input,
        custom_line_right_y_video.change, custom_line_right_y_video.input,
        video_input.change
    ]
    for event in video_preview_events:
        event(
            fn=update_video_preview,
            inputs=[
                video_input, use_custom_line_video,
                custom_line_left_x_video, custom_line_left_y_video,
                custom_line_right_x_video, custom_line_right_y_video
            ],
            outputs=[live_frame_output, video_output]
        )

    # Bind the image button and the upload trigger
    process_btn.click(
        fn=process_traffic_image,
        inputs=[
            image_input, helmet_model_dropdown, use_custom_line_image,
            custom_line_left_x_image, custom_line_left_y_image,
            custom_line_right_x_image, custom_line_right_y_image,
            direction_image
        ],
        outputs=[image_output, signal_status_output, violations_output, history_table]
    )
    
    # Bind the video button and progress updates
    video_process_btn.click(
        fn=process_traffic_video,
        inputs=[
            video_input, video_helmet_dropdown, use_custom_line_video,
            custom_line_left_x_video, custom_line_left_y_video,
            custom_line_right_x_video, custom_line_right_y_video,
            direction_video
        ],
        outputs=[video_log_output, video_output, video_violations_output, video_violations_df, history_table, live_frame_output]
    )
    
    gr.Markdown(
        """
        ---
        **Hackathon Deployment Tip:** To generate a public shareable URL for judges, run this script with `demo.queue().launch(share=True)`.
        """
    )

if __name__ == "__main__":
    import socket
    def find_free_port():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('', 0))
        port = s.getsockname()[1]
        s.close()
        return port
    
    free_port = find_free_port()
    print(f"[*] Found free port: {free_port}")
    demo.queue().launch(server_name="0.0.0.0", server_port=free_port, share=True)
