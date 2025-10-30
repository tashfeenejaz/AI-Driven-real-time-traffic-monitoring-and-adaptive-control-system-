from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
import cv2
from ultralytics import YOLO
import numpy as np
import os
import time
import base64

"""
AI-Driven Real-Time Traffic Monitoring and Adaptive Control System
Author: Tashfeen Ejaz
Description:
Flask-based web app that detects, counts, and manages vehicle flow per lane
using YOLOv8. It dynamically adjusts signal timing and prioritizes ambulances.
"""

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# --- Models ---
model = YOLO("yolov8s.pt")
ambulance_model = YOLO("runs/detect/train5/weights/best.pt")

# --- Globals ---
video_path = None
cap = None

ambulance_counters = {}
lane_polygons = {}
latest_lane_info = {}
current_lane_index = 0
current_lane = None
last_switch_time = time.time()
ambulance_active_until = 0
signal_order = []
congestion_start_times = {}

normal_green_time = 20
max_green_time = 40
congestion_threshold = 6
congestion_hold_time = 6  # seconds to consider congested
label_boxes = []


# ---------- UTILS ----------
def load_lanes(file_path="lanes.txt"):
    lanes = {}
    if not os.path.exists(file_path):
        return lanes
    with open(file_path, "r") as f:
        for line in f:
            if ": " not in line:
                continue
            name, coords = line.strip().split(": ")
            points = [tuple(map(int, pt.split(","))) for pt in coords.split()]
            lanes[name] = points
    return lanes


def find_label_position(polygon, frame_height, frame_width):
    ref_x, ref_y = polygon[0]
    contour = np.array(polygon, dtype=np.int32)
    if cv2.pointPolygonTest(contour, (ref_x, ref_y), False) >= 0:
        return ref_x, ref_y

    for dy in range(0, 150, 10):
        for dx in range(-100, 100, 10):
            test_x = ref_x + dx
            test_y = ref_y + dy
            if 0 <= test_x < frame_width and 0 <= test_y < frame_height:
                if cv2.pointPolygonTest(contour, (test_x, test_y), False) >= 0:
                    return test_x, test_y
    return max(10, min(ref_x, frame_width - 350)), max(30, min(ref_y, frame_height - 60))


# ---------- ROUTES ----------
@app.route('/')
def index():
    global video_path, lane_polygons, cap, signal_order, current_lane, current_lane_index

    if cap is None:
        return redirect(url_for('upload_page'))
    if not os.path.exists("lanes.txt"):
        return redirect(url_for('define_lanes'))

    lane_polygons = load_lanes()
    if not lane_polygons:
        return redirect(url_for('define_lanes'))

    # Maintain signal order from lane file
    signal_order = list(lane_polygons.keys())
    if not signal_order:
        return redirect(url_for('define_lanes'))

    if "Lane1" in signal_order:
        current_lane_index = signal_order.index("Lane1")
    else:
        current_lane_index = 0

    current_lane = signal_order[current_lane_index]

    return render_template("index.html")


@app.route('/upload_page')
def upload_page():
    return render_template("upload.html")


@app.route('/upload', methods=['POST'])
def upload():
    global cap, video_path, lane_polygons

    # ✅ Webcam integration
    if 'use_webcam' in request.form:
        video_path = None
        cap = cv2.VideoCapture(0)  # <-- webcam feed
    else:
        if 'video' not in request.files:
            return "No file part", 400
        file = request.files['video']
        if file.filename == '':
            return "No selected file", 400
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        video_path = filepath
        cap = cv2.VideoCapture(video_path)

    if os.path.exists("lanes.txt"):
        os.remove("lanes.txt")
    lane_polygons = {}

    return redirect(url_for('define_lanes'))


@app.route('/define_lanes')
def define_lanes():
    global cap
    if cap is None:
        return redirect(url_for('upload_page'))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success, frame = cap.read()

    if not success:
        return "Failed to capture frame", 500
    frame = cv2.flip(frame, 1)
    _, buffer = cv2.imencode('.jpg', frame)
    frame_data = base64.b64encode(buffer).decode('utf-8')
    return render_template("define_lanes.html", frame_data=frame_data)


@app.route('/save_lanes', methods=['POST'])
def save_lanes():
    lanes = request.get_json()
    with open("lanes.txt", "w") as f:
        for name, points in lanes.items():
            point_str = " ".join([f"{int(x)},{int(y)}" for x, y in points])
            f.write(f"{name}: {point_str}\n")
    return jsonify({"message": "Lanes saved"}), 200


@app.route('/reset', methods=['POST'])
def reset():
    global cap
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return redirect(url_for('index'))


def generate_frames():
    global cap, lane_polygons, latest_lane_info
    global current_lane_index, current_lane, last_switch_time
    global ambulance_active_until, signal_order, congestion_start_times
    global ambulance_counters

    signal_order = list(lane_polygons.keys())
    if not signal_order:
        return

    current_lane = signal_order[current_lane_index]
    green_time = normal_green_time
    last_switch_time = time.time()

    # initialize congestion timers for all lanes
    for lane in signal_order:
        congestion_start_times[lane] = None

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_height, frame_width = frame.shape[:2]
        lane_counts = {lane: 0 for lane in lane_polygons}
        label_boxes.clear()

        # --- Vehicle detection ---
        results = model(frame, conf=0.25)[0]
        boxes = results.boxes.xyxy

        # --- Ambulance detection ---
        amb_results = ambulance_model(frame, conf=0.15)[0]
        amb_boxes = amb_results.boxes.xyxy
        amb_classes = amb_results.boxes.cls.tolist()
        amb_names = ambulance_model.names

        ambulance_priority_lanes = set()
        for idx, box in enumerate(amb_boxes):
            x1, y1, x2, y2 = map(int, box.tolist())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            label = amb_names[int(amb_classes[idx])]
            if label == "ambulance":
                for lane_name, polygon in lane_polygons.items():
                    contour = np.array(polygon, dtype=np.int32)
                    if cv2.pointPolygonTest(contour, (cx, cy), False) >= 0:
                        if lane_name.lower() == "lane4":
                            continue
                        ambulance_priority_lanes.add(lane_name)
                        break

        # --- Update counters ---
        for lane in lane_polygons:
            if lane not in ambulance_counters:
                ambulance_counters[lane] = 0
            if lane in ambulance_priority_lanes:
                ambulance_counters[lane] = min(ambulance_counters[lane] + 1, 15)
            else:
                ambulance_counters[lane] = max(ambulance_counters[lane] - 1, 0)

        # --- Apply threshold ---
        ambulance_priority_lanes = {lane for lane, c in ambulance_counters.items() if c >= 2}

        # --- Count normal vehicles ---
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.tolist())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            for lane_name, polygon in lane_polygons.items():
                contour = np.array(polygon, dtype=np.int32)
                if cv2.pointPolygonTest(contour, (cx, cy), False) >= 0:
                    lane_counts[lane_name] += 1
                    break

        # --- Congestion check ---
        congested_lanes = []
        current_time = time.time()
        for lane, count in lane_counts.items():
            if count >= congestion_threshold:
                if congestion_start_times[lane] is None:
                    congestion_start_times[lane] = current_time
                elif current_time - congestion_start_times[lane] >= congestion_hold_time:
                    congested_lanes.append(lane)
            else:
                congestion_start_times[lane] = None

        # --- Ambulance Priority Logic ---
        if ambulance_priority_lanes and current_time >= ambulance_active_until:
            priority_lane = list(ambulance_priority_lanes)[0]
            current_lane = priority_lane
            current_lane_index = signal_order.index(priority_lane)
            ambulance_active_until = current_time + max_green_time
            last_switch_time = current_time
            green_time = max_green_time
        elif current_time < ambulance_active_until:
            green_time = max_green_time
        else:
            green_time = max_green_time if current_lane in congested_lanes else normal_green_time
            if current_time - last_switch_time >= green_time:
                current_lane_index = (current_lane_index + 1) % len(signal_order)
                current_lane = signal_order[current_lane_index]
                last_switch_time = time.time()
                green_time = normal_green_time

        elapsed = time.time() - last_switch_time
        remaining_time = max(0, int(green_time - elapsed))
        timer_label = f"{remaining_time}s"

        # --- Draw lanes & labels ---
        for lane_name, polygon in lane_polygons.items():
            pts = np.array(polygon, np.int32)
            is_green = (lane_name == current_lane)
            signal_color = (0, 255, 0) if is_green else (0, 0, 255)
            cv2.polylines(frame, [pts], isClosed=True, color=signal_color, thickness=2)

            # Smart label placement
            ref_x, ref_y = find_label_position(polygon, frame_height, frame_width)
            vehicle_count = lane_counts[lane_name]
            congestion = lane_name in congested_lanes
            lane_text = f"{lane_name}: {vehicle_count}"
            congestion_text = "-CNGST" if congestion else ""
            signal_text = "GREEN" if is_green else "RED"
            signal_info = f"{signal_text} ({remaining_time}s)"

            # Text sizes
            lane_size = cv2.getTextSize(lane_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            congestion_size = cv2.getTextSize(congestion_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            signal_size = cv2.getTextSize(signal_info, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

            label_height = lane_size[1] + signal_size[1] + 10
            label_width = lane_size[0] + congestion_size[0]

            # Clamp ref_y
            bottom_margin = 10
            max_y = frame_height - label_height - bottom_margin
            if ref_y > max_y:
                ref_y = max_y

            # Prevent overlaps
            while any(abs(ref_y - y) < label_height and abs(ref_x - x) < 150 for (x, y, w, h) in label_boxes):
                ref_y += label_height + 5
                if ref_y > max_y:
                    ref_y = max_y

            label_boxes.append((ref_x, ref_y, label_width, label_height))

            # Draw labels
            cv2.putText(frame, lane_text, (ref_x, ref_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            if congestion:
                offset = cv2.getTextSize(lane_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0][0]
                cv2.putText(frame, congestion_text, (ref_x + offset, ref_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, signal_info, (ref_x, ref_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, signal_color, 2)

        # --- Update lane info for API ---
        latest_lane_info = {}

        active_ambulance_lane = None
        if ambulance_priority_lanes:
            active_ambulance_lane = list(ambulance_priority_lanes)[0]

        for lane, count in lane_counts.items():
            congested = count >= congestion_threshold and congestion_start_times[lane] is not None and (current_time - congestion_start_times[lane]) >= congestion_hold_time
            if lane == active_ambulance_lane:
                priority = True
                priority_reason = "ambulance"
            elif lane == current_lane:
                priority = True
                priority_reason = "normal"
            else:
                priority = False
                priority_reason = None

            latest_lane_info[lane] = {
                "count": count,
                "congested": congested,
                "priority": priority,
                "priority_reason": priority_reason
            }

        latest_lane_info["current_green"] = current_lane
        latest_lane_info["timer"] = timer_label
        latest_lane_info["timestamp"] = time.time()

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/lane_data')
def lane_data():
    return jsonify(latest_lane_info)


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
