from ultralytics import YOLO
import cv2
import time

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)

print("YOLOv8 loaded! Press Q to quit")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    start = time.time()
    results = model(frame, verbose=False)
    latency = (time.time() - start) * 1000
    
    # Print detections to terminal
    for box in results[0].boxes:
        class_name = results[0].names[int(box.cls)]
        confidence = float(box.conf)
        print(f"Detected: {class_name} | Confidence: {confidence:.2f} | Latency: {latency:.0f}ms")
    
    cv2.imshow('YOLOv8 Detection', results[0].plot())
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
