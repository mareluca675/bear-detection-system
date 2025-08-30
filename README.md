# Bear Detection System - Simplified

A minimal, focused implementation of bear detection using YOLOv8 computer vision model.

## How It Works

The system uses a pre-trained YOLO (You Only Look Once) neural network to detect bears in real-time video streams. Here's the process:

1. **Video Input**: Captures frames from a camera or video file
2. **Object Detection**: Each frame is processed by the YOLO model
3. **Bear Identification**: The model identifies objects and their classes
4. **Filtering**: Only bear detections above the confidence threshold are kept
5. **Output**: Displays bounding boxes around detected bears

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

Close the detection window to stop the program.

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

## Technical Explanation

### YOLO Architecture
YOLO divides an image into a grid and predicts bounding boxes and class probabilities for each grid cell in a single forward pass. This makes it extremely fast compared to region-based methods.

### Detection Pipeline
1. **Pre-processing**: Input image is resized to model input size (typically 640x640)
2. **Inference**: Neural network processes the image in one pass
3. **Post-processing**: Non-maximum suppression removes duplicate detections
4. **Filtering**: Class-specific filtering for bear-related classes

### Model Variants
- **YOLOv8n**: Nano - Fastest, lowest accuracy (3.2M parameters)
- **YOLOv8s**: Small - Good balance (11.2M parameters)
- **YOLOv8m**: Medium - Better accuracy (25.9M parameters)
- **YOLOv8l**: Large - High accuracy (43.7M parameters)
- **YOLOv8x**: Extra Large - Highest accuracy (68.2M parameters)

## Learning Resources

### Computer Vision Fundamentals
- **Object Detection Basics**: [Stanford CS231n Course](http://cs231n.stanford.edu/)
- **YOLO Paper**: [Original YOLO Paper](https://arxiv.org/abs/1506.02640)
- **YOLOv8 Documentation**: [Ultralytics Docs](https://docs.ultralytics.com/)

### Deep Learning Concepts
- **Convolutional Neural Networks**: [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
- **PyTorch Tutorials**: [Official PyTorch Tutorials](https://pytorch.org/tutorials/)
- **Transfer Learning**: [TensorFlow Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)

### Practical Implementation
- **OpenCV Python**: [OpenCV-Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- **Real-time Object Detection**: [PyImageSearch Tutorials](https://pyimagesearch.com/category/object-detection/)
- **YOLO Training Guide**: [Train Custom YOLO](https://blog.roboflow.com/how-to-train-yolov8/)

### Wildlife Detection Specific
- **Camera Trap ML**: [MegaDetector Project](https://github.com/microsoft/CameraTraps)
- **Wildlife Conservation Tech**: [WILDLABS Resources](https://wildlabs.net/)
- **Computer Vision for Conservation**: [Conservation AI](https://www.conservationai.co.uk/)

## Key Concepts Explained

### Confidence Threshold
The confidence threshold (0-1) determines the minimum certainty required for a detection to be considered valid. Lower values = more detections but more false positives.

### Bounding Boxes
Rectangular boxes defined by coordinates (x1, y1, x2, y2) that indicate the location of detected objects in the image.

### Non-Maximum Suppression (NMS)
Algorithm that removes overlapping bounding boxes, keeping only the one with highest confidence score for each object.

### Frames Per Second (FPS)
Number of images the system can process per second. Depends on:
- Model size
- Hardware (CPU vs GPU)
- Image resolution
- Number of objects in scene

## Performance Optimization

### For Better Speed
- Use smaller model (yolov8n or yolov8s)
- Reduce input resolution
- Enable GPU acceleration (CUDA)
- Process every nth frame instead of all frames

### For Better Accuracy
- Use larger model (yolov8l or yolov8x)
- Increase input resolution
- Lower confidence threshold
- Ensure good lighting and image quality

## Troubleshooting

### Common Issues

**No detections:**
- Check if confidence threshold is too high
- Verify model is loaded correctly
- Ensure input source is working
- Test with a clear image of a bear

**Low FPS:**
- Switch to smaller model
- Check if GPU is being used
- Reduce video resolution
- Close other applications

**GPU not detected:**
```python
import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.device_count())   # Number of GPUs
```

## Understanding the Code

### Core Components

1. **BearDetector Class**: Main detection engine
   - Loads and manages YOLO model
   - Processes frames for detection
   - Filters results for bear-specific classes

2. **Detection Method**: Processes single frames
   - Runs inference on image
   - Extracts bounding boxes and confidence scores
   - Filters for bear detections

3. **Video Processing**: Handles continuous streams
   - Captures frames from source
   - Applies detection to each frame
   - Displays results in real-time

### Data Flow
```
Video Source → Frame Capture → YOLO Model → Detection Results → Filtering → Display
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

## License
MIT License - Free to use and modify

## Further Development Ideas

- Add detection logging to CSV/database
- Implement alert system (email/SMS)
- Add multi-camera support
- Create web interface
- Train custom model on bear-specific dataset
- Add species classification (black bear vs grizzly)
- Implement tracking across frames
- Add motion detection pre-filter for efficiency