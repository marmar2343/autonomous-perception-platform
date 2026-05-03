# Autonomous Perception Platform

🚧 **Work in Progress** 🚧

End-to-end autonomous perception system with real-time object detection, depth estimation, and data streaming pipeline

## Features

* Real-time object detection using YOLOv8
* Monocular depth estimation using DepthAnything v2
* 3D position fusion — combining detections with  depth data

## Tech Stack

### Computer Vision
* YOLOv8
* DepthAnything v2
* OpenCV

### Data Engineering
* Coming soon...

### MLOps
* Coming soon...

## Installation

1. Clone repo 
 ```
 git clone https://github.com/marmar2343/autonomous-perception-platform.git
```
2. Make conda enviroment 
```
conda create -n perception-platform python=3.11
conda activate perception-platform
```
3. Install requirements 
```
pip install -r requirements.txt
```

## Usage

Run the video pipeline:

```
python scripts/video_pipeline.py
```

## Roadmap

- [x] 3D scene perception
- [ ] Multi-object tracking
- [ ] Streaming data pipeline
- [ ] MLOps and containerization
- [ ] Multimodal VLM module
- [ ] Benchmark and documentation






