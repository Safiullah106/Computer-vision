# Evolution of Object Detection: From YOLOv8 to RT-DETR 🚀

This repository contains my hands-on journey exploring, implementing, and benchmarking the transition from traditional Convolutional Neural Network (CNN) architectures to modern One-Stage Detectors and cutting-edge Vision Transformers (ViTs). 

All models were implemented, tested, and optimized using Python, OpenCV, and the Ultralytics framework on cloud GPUs via Google Colab.

---

## 📌 Project Overview & Evolution

The project traces the architectural improvements in computer vision through practical video analytics implementation:

1. **YOLOv8 (Persistent Tracking)**: Implemented multi-object vehicle tracking with `persist=True`. This step moved beyond standalone frame detection to map temporal trajectories using persistent object IDs.
2. **YOLOv10 (End-to-End Head)**: Explored the elimination of **NMS (Non-Maximum Suppression)**. YOLOv10's Consistent Dual Assignments architecture significantly dropped post-processing latency, yielding a cleaner, faster raw prediction stream.
3. **YOLO26 (Edge-Optimized Pose Tracking)**: Deployed flagship edge-optimized human pose estimation (`yolo26n-pose.pt`) over high-motion dance and fitness videos, tracking the 17 standard COCO keypoint joints in real-time.
4. **RT-DETR (Real-Time Vision Transformer)**: The final breakthrough. Implemented a memory-efficient `stream=True` pipeline to bypass traditional CNN localized bottlenecks. 

---

## 📈 Key Findings & Architectural Insights

* **CNNs vs. Transformers**: While YOLO models are exceptionally lightweight and optimized for the edge, they occasionally suffer from bounding box "flicker" or dropped detections in high-density scenes when objects overlap.
* **Global Context Attention**: RT-DETR treats the frame as a global network of relationship queries. In a chaotic intersection video, it successfully and stably isolated layered objects (e.g., a `person` holding a `handbag` underneath an `umbrella`) alongside fine-grained urban infrastructure (`traffic light`, `clock`, `tv`).

---

## 🛠️ Tech Stack & Dependencies

* **Language**: Python 3.x
* **Framework**: Ultralytics (YOLOv8, YOLOv10, YOLO26, RT-DETR)
* **Video Processing**: OpenCV (`cv2`), `yt-dlp` (for direct YouTube streaming)
* **Environment**: Google Colab (T4 Cloud GPU) / Anaconda Jupyter Notebook

---

## 🚀 Getting Started & Code Examples

### 1. Install Dependencies
```bash
pip install ultralytics opencv-python yt-dlp
```

### 2. Run Memory-Efficient RT-DETR Pipeline
This script streams inference frame-by-frame, avoiding RAM bottlenecks while processing high-density videos:

```python
import cv2
import os
from ultralytics import RTDETR

# Load Real-Time Transformer Model
model = RTDETR('rtdetr-l.pt')

input_video_path = "your_test_video.mp4"
output_video_path = "output_rtdetr.mp4"

# OpenCV Setup
cap = cv2.VideoCapture(input_video_path)
w, h, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

print("Processing video with RT-DETR...")
results = model.predict(source=input_video_path, conf=0.3, stream=True)

for r in results:
    annotated_frame = r.plot()
    video_writer.write(annotated_frame)

cap.release()
video_writer.release()
print("Saved output directly to storage!")
```

---

## 📁 Repository Structure
```text
├── .gitignore
├── README.md
├── requirements.txt
├── scripts/
│   ├── yolov8_tracking.py
│   ├── yolov10_inference.py
│   ├── yolo26_pose_estimation.py
│   └── rtdetr_stream_pipeline.py
└── samples/
    └── (Place your input/output video clips here)
