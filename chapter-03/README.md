# Chapter 3: YOLO26 Deployment and Real-Time Inference

In this chapter, we will learn how to use a trained YOLO26 model in real applications.

Training a model is not the final step.

After training, the model must be tested and deployed in a real environment.

This chapter focuses on the following workflow:

```text
Trained Model
      ↓
Inference
      ↓
Input Source
      ↓
Detection
      ↓
Results
      ↓
Real Application
```

---

# Table of Contents

* What Is Inference?
* Loading a Trained Model
* Running Inference on Images
* Running Inference on Videos
* Using a Webcam
* Using an IP Camera
* Real-Time Detection
* Multiple Camera Processing
* Performance Optimization
* GPU Optimization
* Model Export
* Deployment Considerations

---

# 1. What Is Inference?

Inference means using a trained model to make predictions.

During training:

```text
Dataset → Model Learning
```

During inference:

```text
New Input → Trained Model → Prediction
```

The input can be:

* An image
* A video
* A webcam stream
* An IP camera
* A live video stream

---

# 2. Loading a Trained Model

First, load your trained model.

Example:

```python
from ultralytics import YOLO

model = YOLO("best.pt")
```

The `best.pt` file is usually the model that produced the best validation results during training.

---

# 3. Running Inference on an Image

You can run detection on an image.

Example:

```python
from ultralytics import YOLO

model = YOLO("best.pt")

results = model("image.jpg")
```

YOLO will process the image and return detection results.

You can save the result using:

```python
results[0].save()
```

---

# 4. Running Inference on a Video

YOLO can also process video files.

Example:

```python
from ultralytics import YOLO

model = YOLO("best.pt")

results = model("video.mp4", stream=True)

for result in results:
    result.show()
```

The model processes the video frame by frame.

The general workflow is:

```text
Video
   ↓
Frame 1 → YOLO
Frame 2 → YOLO
Frame 3 → YOLO
   ↓
Detection Results
```

---

# 5. Using a Webcam

You can use a local webcam as an input source.

Example:

```python
from ultralytics import YOLO

model = YOLO("best.pt")

results = model(0, stream=True)

for result in results:
    result.show()
```

The value:

```text
0
```

usually represents the first camera connected to the computer.

If multiple cameras are available:

```text
0 → First Camera
1 → Second Camera
2 → Third Camera
```

---

# 6. Using an IP Camera

YOLO can also process an IP camera stream.

For example:

```python
from ultralytics import YOLO

model = YOLO("best.pt")

stream_url = "rtsp://camera-address/stream"

results = model(stream_url, stream=True)

for result in results:
    result.show()
```

Many IP cameras use protocols such as:

* RTSP
* HTTP
* RTMP

RTSP is commonly used for security cameras.

---

# 7. Real-Time Object Detection

Real-time detection means processing frames fast enough for the application.

The basic process is:

```text
Camera
   ↓
Video Frame
   ↓
YOLO Model
   ↓
Detection
   ↓
Display or Action
```

Performance is usually measured using:

```text
FPS
```

FPS means:

```text
Frames Per Second
```

For example:

```text
5 FPS  → Slow
15 FPS → Moderate
30 FPS → Real-Time
60 FPS → Very Fast
```

The required FPS depends on the project.

Not every application requires 30 FPS.

For example, a slow industrial monitoring system may work correctly at 5 FPS.

---

# 8. Measuring Inference Performance

Do not evaluate a deployment system using only FPS.

You should also measure:

* Inference latency
* GPU usage
* VRAM usage
* CPU usage
* Input resolution
* Number of cameras

For example:

```text
Camera Resolution: 1920 × 1080
Model Input Size: 640
FPS: 30
GPU Usage: 70%
VRAM Usage: 6 GB
```

This information helps you understand the real system performance.

---

# 9. Multiple Camera Processing

Processing multiple cameras is more complex than processing a single camera.

For example:

```text
Camera 1
Camera 2
Camera 3
Camera 4
Camera 5
Camera 6
```

Each camera produces multiple video frames.

The system must:

```text
Receive Frames
      ↓
Decode Video
      ↓
Preprocess Frames
      ↓
Run YOLO Inference
      ↓
Process Results
```

The GPU must process a large amount of data.

Therefore, a system should be benchmarked using the actual number of cameras.

Do not assume that:

```text
One Camera = 30 FPS
```

means:

```text
Six Cameras = 180 FPS
```

Real systems have additional costs such as:

* Video decoding
* CPU processing
* Memory transfers
* Network latency
* Result processing

---

# 10. Improving Real-Time Performance

There are several ways to improve performance.

---

## Use a Smaller Model

For example:

```text
YOLO26n → Faster
YOLO26s → Balanced
YOLO26m → More Accurate
YOLO26l → Larger and Slower
```

Start with the smallest model that provides acceptable accuracy.

---

## Reduce Image Size

For example:

```text
1280 → High Cost
640  → Balanced
416  → Faster
```

Reducing the input size can significantly increase FPS.

However, it may reduce small object detection performance.

---

## Process Fewer Frames

Some applications do not need to process every video frame.

For example:

```text
Camera: 30 FPS
YOLO Processing: 10 FPS
```

The system can skip some frames.

This reduces GPU usage.

---

## Use GPU Acceleration

A CUDA-compatible GPU can significantly improve inference performance.

Make sure that:

* NVIDIA drivers are installed
* CUDA is supported
* PyTorch detects the GPU

Example:

```python
import torch

print(torch.cuda.is_available())
```

---

# 11. Model Export

A trained model can be exported to different formats.

Example:

```python
from ultralytics import YOLO

model = YOLO("best.pt")

model.export(format="onnx")
```

Common formats include:

```text
PyTorch
ONNX
TensorRT
OpenVINO
CoreML
```

Different formats are suitable for different deployment environments.

---

# 12. Choosing a Deployment Format

A simple comparison:

| Format   | Best For                  |
| -------- | ------------------------- |
| PyTorch  | Development and Testing   |
| ONNX     | Cross-platform Deployment |
| TensorRT | NVIDIA GPU                |
| OpenVINO | Intel Hardware            |
| CoreML   | Apple Devices             |

The best format depends on the target hardware.

---

# 13. Deployment Architecture

A production system can look like this:

```text
Camera
   ↓
Stream Manager
   ↓
Frame Processing
   ↓
YOLO Inference
   ↓
Detection Results
   ↓
Database / API / Dashboard
```

The model is only one part of the complete system.

A production application may also need:

* APIs
* Logging
* Monitoring
* Error handling
* Storage
* Authentication
* Alert systems

---

# 14. A Simple API Architecture

For example:

```text
Client
   ↓
FastAPI
   ↓
YOLO Model
   ↓
Inference
   ↓
JSON Response
```

A response may contain:

```json
{
    "detections": [
        {
            "class": "drone",
            "confidence": 0.92,
            "bbox": [120, 80, 300, 250]
        }
    ]
}
```

This architecture allows other applications to use the YOLO model through an API.

---

# 15. Using Docker for Deployment

Docker can make deployment easier.

A Docker container can include:

* Python
* Required libraries
* YOLO
* Application code

This helps create a reproducible environment.

The basic workflow is:

```text
Application Code
      +
Dependencies
      +
Configuration
      ↓
Docker Image
      ↓
Container
      ↓
Deployment
```

Docker is especially useful when deploying the application to another system or server.

---

# 16. Monitoring a Production System

After deployment, the system should be monitored.

Important metrics include:

```text
GPU Usage
CPU Usage
VRAM Usage
RAM Usage
FPS
Inference Latency
Camera Connection Status
Error Rate
```

A system may work correctly during testing but fail after running for several days.

For this reason, production monitoring is important.

---

# 17. Benchmarking Before Deployment

Before selecting a final model, create a benchmark.

For example:

| Model   | Accuracy |    FPS |   VRAM | Cameras |
| ------- | -------: | -----: | -----: | ------: |
| YOLO26n |   Medium |   High |    Low |       6 |
| YOLO26s |     High | Medium | Medium |       4 |
| YOLO26m |   Higher |  Lower |   High |       2 |

The values above are only examples.

Your benchmark should be based on:

* Your dataset
* Your GPU
* Your input resolution
* Your number of cameras
* Your real application

This is extremely important.

A model that performs well in a laboratory environment may not perform well in production.

---

# 18. Recommended Deployment Workflow

A good deployment workflow is:

```text
1. Train Multiple Models
        ↓
2. Evaluate Accuracy
        ↓
3. Measure FPS
        ↓
4. Measure GPU and VRAM Usage
        ↓
5. Test Real Input Sources
        ↓
6. Test Multiple Cameras
        ↓
7. Select the Best Model
        ↓
8. Export the Model
        ↓
9. Build the Application
        ↓
10. Deploy
        ↓
11. Monitor Performance
```

---

# Final Recommendations

A successful YOLO project does not end after training.

The final model should be tested in the real environment where it will be used.

Always test:

* Real cameras
* Real lighting conditions
* Real object sizes
* Real hardware
* Real network conditions

The final goal should be:

```text
Good Accuracy
+
Acceptable Speed
+
Stable System
+
Efficient Hardware Usage
=
Successful Deployment
```

---

# Project Complete

You have now learned the complete basic YOLO workflow:

```text
Chapter 1
YOLO Fundamentals and Optimization
        ↓
Chapter 2
Training and Evaluation
        ↓
Chapter 3
Deployment and Real-Time Inference
```

The next step is to build real projects and experiment with different datasets, models, hardware configurations, and deployment architectures.

Remember:

```text
Good Data
+
Good Model
+
Correct Training
+
Real Benchmarking
+
Proper Deployment
=
A Successful AI System
```
