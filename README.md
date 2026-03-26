# GSoC 2026 — ArduPilot Vision Obstacle Avoidance

## Project
Vision-based obstacle detection with improved BendyRuler 
avoidance for ArduPilot — Google Summer of Code 2026.

## Validated Results
- YOLOv8-Nano: **45ms latency, 0.90 confidence**
- Depth Anything v2: **368ms per frame (async)**
- OBSTACLE_DISTANCE MAVLink: **10Hz to ArduPilot SITL**
- Full pipeline: Camera → YOLO → Distance → MAVLink → SITL ✅

## Files
- `vision_pipeline.py` — Complete vision pipeline
- `send_obstacle.py` — MAVLink OBSTACLE_DISTANCE sender
- `test_yolo.py` — YOLOv8-Nano benchmark
- `test_depth.py` — Depth Anything v2 benchmark

## Mentors
- Sanket Sharma (GSoC Mentor)
- Rhys Mainwaring (Dev-Team)

## Organization
ArduPilot — ardupilot.org

