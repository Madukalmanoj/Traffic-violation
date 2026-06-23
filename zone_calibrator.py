"""
Click-based zone calibration tool.

Replaces slider-based bounding boxes with polygons you draw by clicking
directly on a still frame from your camera. This fixes the perspective
distortion problem (a polygon can follow a slanted line; a rectangle can't)
and sidesteps occlusion/clutter entirely, since the zone is now a fixed
fact about the camera, not something detected from pixels every frame.

Calibrate three zones per lane:
  - stop_line   : 4 points, the corners of the stop-line strip on the road
  - exit_line   : 4 points, a second line further into the junction -
                  used to confirm a vehicle actually drove through on red,
                  not just touched the stop line
  - signal_roi  : 2+ points loosely around the signal lamp housing
                  (downstream code takes cv2.boundingRect of these)

Usage:
    python zone_calibrator.py --video path/to/intersection.mp4
    python zone_calibrator.py --image path/to/frame.jpg
    python zone_calibrator.py --video path/to/intersection.mp4 --frame-index 150

Tip: pick a frame where the stop line is NOT covered by a stopped vehicle.
If frame 0 has traffic on it, bump --frame-index until the road is clear -
you only need ONE clean frame, calibration is a one-time setup step.

Click points on the image to build the active zone. Switch zones with the
radio buttons. Save writes calibration.json next to this script; your
detection pipeline loads that file at runtime (see point_in_zone() below
for the check you'll run per tracked vehicle).
"""

import argparse
import json

import cv2
import gradio as gr
import numpy as np

CALIBRATION_PATH = "calibration.json"

ZONE_COLORS = {
    "stop_line": (255, 0, 0),    # red   (RGB, since gr.Image uses RGB)
    "exit_line": (255, 165, 0),  # orange
    "signal_roi": (0, 200, 0),   # green
}

zones = {"stop_line": [], "exit_line": [], "signal_roi": []}
base_frame = None  # calibration reference image, RGB numpy array


def grab_reference_frame(video_path=None, image_path=None, frame_index=0):
    """Get a single clean frame to calibrate against."""
    if image_path:
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(image_path)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def render_preview():
    """Draw all zones (points + polygon outline) on top of the base frame."""
    img = base_frame.copy()
    for zone_name, pts in zones.items():
        color = ZONE_COLORS[zone_name]
        for p in pts:
            cv2.circle(img, p, 5, color, -1)
        if len(pts) >= 2:
            cv2.polylines(
                img, [np.array(pts, dtype=np.int32)],
                isClosed=len(pts) >= 3, color=color, thickness=2,
            )
    return img


def zone_status():
    return "\n".join(f"{name}: {len(pts)} pts -> {pts}" for name, pts in zones.items())


def on_image_click(active_zone, evt: gr.SelectData):
    # evt.index gives the clicked pixel as (x, y). NOTE: if this ever comes
    # back as (None, None), the gr.Image component needs interactive=True -
    # that's a known quirk across several Gradio versions, not a bug here.
    x, y = evt.index
    zones[active_zone].append((int(x), int(y)))
    return render_preview(), zone_status()


def undo_last(active_zone):
    if zones[active_zone]:
        zones[active_zone].pop()
    return render_preview(), zone_status()


def clear_zone(active_zone):
    zones[active_zone] = []
    return render_preview(), zone_status()


def save_calibration():
    h, w = base_frame.shape[:2]
    payload = {"frame_width": w, "frame_height": h, "zones": zones}
    with open(CALIBRATION_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    return f"Saved {CALIBRATION_PATH}"


def point_in_zone(point, polygon_points):
    """Runtime check: is a tracked vehicle's reference point inside/on a
    calibrated zone? Use this instead of any per-frame color/contour test.

    point: (x, y) - e.g. the bottom-center of a vehicle's bounding box
    polygon_points: list of (x, y) from calibration.json -> zones[name]
    """
    poly = np.array(polygon_points, dtype=np.int32)
    return cv2.pointPolygonTest(poly, point, False) >= 0


def build_app():
    with gr.Blocks(title="Zone calibration") as demo:
        gr.Markdown(
            "Click directly on the frame to add points to the active zone. "
            "stop_line / exit_line: click the 4 corners of the strip on the "
            "road, in order. signal_roi: click 2+ points loosely around the "
            "lamp housing."
        )
        active_zone = gr.Radio(
            choices=list(zones.keys()), value="stop_line", label="Active zone"
        )
        image = gr.Image(value=render_preview(), type="numpy", interactive=True)
        status = gr.Textbox(label="Points", value=zone_status(), lines=4)

        with gr.Row():
            undo_btn = gr.Button("Undo last point")
            clear_btn = gr.Button("Clear zone")
            save_btn = gr.Button("Save calibration.json", variant="primary")
        save_msg = gr.Textbox(label="", interactive=False)

        image.select(on_image_click, inputs=[active_zone], outputs=[image, status])
        undo_btn.click(undo_last, inputs=[active_zone], outputs=[image, status])
        clear_btn.click(clear_zone, inputs=[active_zone], outputs=[image, status])
        save_btn.click(save_calibration, outputs=[save_msg])

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="Path to a video to grab a frame from")
    parser.add_argument("--image", help="Path to a still image to calibrate against")
    parser.add_argument("--frame-index", type=int, default=0)
    args = parser.parse_args()

    if not args.video and not args.image:
        raise SystemExit("Pass --video path/to/file.mp4 or --image path/to/frame.jpg")

    base_frame = grab_reference_frame(args.video, args.image, args.frame_index)
    build_app().launch()
