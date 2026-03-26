from ultralytics import YOLO
from pymavlink import mavutil
from pymavlink.dialects.v20 import ardupilotmega as mavlink2
import cv2
import time
import numpy as np

# ── 1. Connect to ArduPilot ──
print("Connecting to ArduPilot...")
master = mavutil.mavlink_connection(
    'udp:0.0.0.0:14552',
    input=False,
    dialect='ardupilotmega'
)
master.mav.srcSystem = 255
print("Connected!")

# ── 2. Load YOLOv8 ──
print("Loading YOLOv8-Nano...")
model = YOLO('yolov8n.pt')
print("Model loaded!")

# ── 3. Open Camera ──
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(f"Camera opened! Width: {frame_width}px")

# ── 4. Class weights ──
# How dangerous is each class?
# Higher = more dangerous = closer avoidance
DANGER_CLASSES = {
    'person': 1.0,
    'car': 1.0,
    'truck': 1.0,
    'bus': 1.0,
    'bicycle': 0.8,
    'motorcycle': 0.8,
    'dog': 0.6,
    'chair': 0.5,
    'dining table': 0.5,
}

def estimate_distance(box_width, box_height, class_name):
    """
    Estimate distance based on bounding box size.
    Larger box = closer object.
    Returns distance in centimeters.
    """
    # Box area as fraction of frame
    box_area = box_width * box_height
    frame_area = frame_width * 480

    area_fraction = box_area / frame_area

    # Calibration: if object fills 50% of frame = ~1m away
    # if object fills 5% of frame = ~4m away
    if area_fraction > 0.5:
        return 80    # very close ~0.8m
    elif area_fraction > 0.3:
        return 150   # close ~1.5m
    elif area_fraction > 0.1:
        return 250   # medium ~2.5m
    elif area_fraction > 0.05:
        return 400   # far ~4m
    else:
        return 800   # very far ~8m

def send_obstacles(distances):
    """Send obstacle distance array to ArduPilot"""
    msg = mavlink2.MAVLink_obstacle_distance_message(
        int(time.time() * 1e6),
        mavlink2.MAV_DISTANCE_SENSOR_UNKNOWN,
        distances,
        5, 20, 1500
    )
    master.mav.send(msg)

print("\n🚀 Vision pipeline running! Press Q to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()

    # ── 5. Run YOLOv8 ──
    results = model(frame, verbose=False)

    # ── 6. Build obstacle array ──
    distances = [65535] * 72  # 65535 = no obstacle

    for box in results[0].boxes:
        class_name = results[0].names[int(box.cls)]
        confidence = float(box.conf)

        # Only process confident detections
        if confidence < 0.4:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        box_width = x2 - x1
        box_height = y2 - y1
        center_x = (x1 + x2) / 2

        # ── 7. Map to sector ──
        # 72 sectors, each 5 degrees
        # Center of frame = sector 0 (forward)
        sector = int((center_x / frame_width) * 72)
        sector = max(0, min(71, sector))

        # ── 8. Estimate distance ──
        dist_cm = estimate_distance(
            box_width, box_height, class_name
        )

        # Apply danger weight
        danger = DANGER_CLASSES.get(class_name, 0.3)
        dist_cm = int(dist_cm / danger)

        # Keep minimum distance per sector
        if dist_cm < distances[sector]:
            distances[sector] = dist_cm

        print(f"Detected: {class_name:15} | "
              f"Conf: {confidence:.2f} | "
              f"Dist: {dist_cm}cm | "
              f"Sector: {sector}")

    # ── 9. Send to ArduPilot ──
    send_obstacles(distances)

    # ── 10. Show latency ──
    latency = (time.time() - start) * 1000

    # Draw info on frame
    cv2.putText(
        frame,
        f'Latency: {latency:.0f}ms | '
        f'Detections: {len(results[0].boxes)}',
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 255, 0), 2
    )

    cv2.imshow('Vision Pipeline', results[0].plot())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Pipeline stopped.")



