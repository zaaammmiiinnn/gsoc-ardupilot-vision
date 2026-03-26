from transformers import pipeline
from PIL import Image
import cv2
import numpy as np
import time

print("Loading Depth Anything v2...")
pipe = pipeline(
    "depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf"
)
print("Loaded! Starting camera...")

cap = cv2.VideoCapture(0)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Show original camera always
    cv2.imshow('Original Camera', frame)

    # Run depth every 8th frame only
    if frame_count % 8 == 0:
        pil_img = Image.fromarray(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        start = time.time()
        depth = pipe(pil_img)["depth"]
        latency = (time.time() - start) * 1000

        depth_np = np.array(depth)
        depth_norm = cv2.normalize(
            depth_np, None, 0, 255,
            cv2.NORM_MINMAX, cv2.CV_8U
        )
        depth_colored = cv2.applyColorMap(
            depth_norm, cv2.COLORMAP_MAGMA
        )

        cv2.putText(
            depth_colored,
            f'Depth Anything v2 | {latency:.0f}ms',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 255), 2
        )

        # Show depth map
        cv2.imshow('Depth Map', depth_colored)
        print(f"Frame {frame_count} | Depth: {latency:.0f}ms")

    frame_count += 1

    # Important — must call waitKey to show windows
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()