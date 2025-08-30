# Quick Start Guide

## 5-Minute Setup

### Step 1: Install Python Dependencies
```bash
pip install ultralytics opencv-python torch torchvision numpy
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

## Understanding YOLO

YOLO (You Only Look Once) is a real-time object detection system that:
- Processes entire images in a single neural network evaluation
- Divides image into grid cells
- Each cell predicts bounding boxes and class probabilities
- Uses Non-Maximum Suppression to remove duplicate detections

### Why YOLO for Bear Detection?

- **Speed**: Can process 30-140 FPS depending on model size
- **Accuracy**: Modern YOLO versions have excellent detection rates
- **Versatility**: Pre-trained on diverse datasets including animals
- **Efficiency**: Single-pass detection is computationally efficient

## Key Parameters Explained

### Confidence Threshold (0.0 - 1.0)
- **0.3**: More detections, might include false positives
- **0.5**: Balanced (default)
- **0.7**: High confidence only, might miss some bears

### Model Selection
- **yolov8n**: 3 MB, 140 FPS on GPU, basic accuracy
- **yolov8s**: 22 MB, 100 FPS on GPU, good accuracy
- **yolov8m**: 50 MB, 60 FPS on GPU, better accuracy
- **yolov8l**: 84 MB, 35 FPS on GPU, high accuracy
- **yolov8x**: 131 MB, 20 FPS on GPU, best accuracy

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
```

## Learning Path

### Beginner (Start Here)
1. Run the basic detection to see it work
2. Try different confidence thresholds
3. Test with different video sources
4. Read about object detection basics

### Intermediate
1. Experiment with different YOLO models
2. Modify the code to add logging
3. Learn about neural networks and computer vision
4. Try the Python API examples

### Advanced
1. Train custom model on bear-specific dataset
2. Implement real-time tracking across frames
3. Add species classification
4. Optimize for embedded devices

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

## Next Steps

1. **Customize Detection**: Modify `bear_detection.py` to add your own features
2. **Add Logging**: Save detections to CSV or database
3. **Create Alerts**: Add email/SMS notifications when bears detected
4. **Train Custom Model**: Use your own bear images for better accuracy
5. **Build Interface**: Create web or mobile app using the detection API