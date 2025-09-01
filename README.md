# YOLOv5 Image & Video Object Detection

This Python project performs object detection on images and videos using the YOLOv5 model from Ultralytics. Detected objects are drawn with bounding boxes and confidence scores, and outputs can be saved locally.

## Features

- Detects objects in images and videos.

- Uses YOLOv5n (auto-downloads weights if missing).

- Saves processed images/videos to specified folders.

- Displays inference time for video frames.

- Has Real-Time detection.

## Create a virtual environment, install requirements and start!
```bash
cd path/to/this/project

## Mac:
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python3 main.py

## Windows:
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python main.py
```

## Usage
- If you want to place videos or photos for this object detection, you need to modify the names in main.py
- If there no videos or images, it will open your camera and start Real Time detection 
- Videos will be saved in ./video_output
- Photos will be saved in ./photo_output

## Examples
They are published
