# Bear Detection System

A minimal, focused implementation of bear detection using YOLOv8 computer vision model.

## How It Works

The system uses a pre-trained YOLO (You Only Look Once) neural network to detect bears in real-time video streams. Here's the process:

1. **Video Input**: Captures frames from a camera or video file
2. **Object Detection**: Each frame is processed by the YOLO model
3. **Bear Identification**: The model identifies objects and their classes
4. **Filtering**: Only bear detections above the confidence threshold are kept
5. **Output**: Displays bounding boxes around detected bears

# Quick Start Guide

### Step 1: Install Requirements
```bash
pip install -r requirements.txt
```

### Step 2: Run Setup
```bash
python setup.py
```
This will:
- Verify dependencies
- Download the YOLO model
- Test your camera
- Check for GPU support

### Step 3: Start Detecting Bears
```bash
# Use your camera
python bear_detection.py

# Use a video file
python bear_detection.py --source video.mp4
```

Close the detection window to stop the program.

## How the Program Works

### Core Workflow

1. **Input Stage**: The program captures video frames from your camera or a video file
2. **Detection Stage**: Each frame is processed by the YOLO neural network
3. **Classification Stage**: The AI identifies objects and determines if they are bears
4. **Output Stage**: Bounding boxes are drawn around detected bears

### Technical Process

```
Camera/Video → Frame Extraction → YOLO Model → Object Detection → Bear Filter → Display
     ↓               ↓                 ↓              ↓              ↓           ↓
  30 FPS      640x640 resize    Neural Network   Confidence     Class='bear'  Boxes
```

### What Happens Inside

1. **Frame Capture**: OpenCV captures frames at ~30 FPS
2. **Preprocessing**: Images are resized to 640x640 pixels
3. **Neural Network**: YOLO processes the entire image in one pass
4. **Detection Output**: Model outputs bounding boxes, classes, and confidence scores
5. **Filtering**: Only keep detections where class='bear' and confidence > threshold
6. **Visualization**: Draw green boxes around bears with confidence scores

## Common Commands

```bash
# Basic detection with default settings
python bear_detection.py

# Use high accuracy model with low threshold
python bear_detection.py --model yolov8x.pt --confidence 0.3

# Process video without display (for logging)
python bear_detection.py --source video.mp4 --no-display

# Use second camera
python bear_detection.py --source 1

# Set window size
python bear_detection.py --width 1080 --height 720
```

## Troubleshooting

### No bears detected?
- Lower confidence threshold: `--confidence 0.3`
- Ensure good lighting
- Test with clear bear images first
- Try larger model: `--model yolov8x.pt`

### Too slow?
- Use smaller model: `--model yolov8n.pt`
- Check GPU is enabled
- Reduce video resolution
- Process every nth frame

### Camera not working?
- Try different camera index: `--source 1` or `--source 2`
- Check camera permissions
- Test with video file instead

### Python API Usage
```python
from bear_detection import BearDetector

# Initialize detector
detector = BearDetector(model_path='yolov8x.pt', confidence_threshold=0.5)

# Process video stream
detector.process_video_stream(source=0, display=True)

# Or detect in a single frame
import cv2
frame = cv2.imread('image.jpg')
detections = detector.detect_bears(frame)
```
## System Requirements

### Minimum
- Python 3.8+
- 4GB RAM
- Any modern CPU
- Webcam or video file

### Recommended
- Python 3.10+
- 8GB+ RAM
- NVIDIA GPU with CUDA support
- 1080p+ camera

## Further Development Ideas

- Add detection logging to CSV/database
- Implement alert system (email/SMS)
- Add multi-camera support
- Create web interface
- Train custom model on bear-specific dataset
- Add species classification (black bear vs grizzly)
- Implement tracking across frames
- Add motion detection pre-filter for efficiency
