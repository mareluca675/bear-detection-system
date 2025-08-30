"""
Bear Detection System - Simplified Version
A minimal implementation using YOLOv8 for bear detection
"""

import cv2
import torch
from ultralytics import YOLO
import numpy as np
from pathlib import Path
import argparse
import sys
import time


class BearDetector:
    """Simple bear detection using YOLOv8"""
    
    def __init__(self, model_path='yolov8x.pt', confidence_threshold=0.5):
        """
        Initialize the bear detector
        
        Args:
            model_path: Path to YOLO model file
            confidence_threshold: Minimum confidence for detections (0-1)
        """
        self.confidence_threshold = confidence_threshold
        
        # Load YOLO model
        try:
            self.model = YOLO(model_path)
            print(f"Model loaded: {model_path}")
            
            # Use GPU if available
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Using device: {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    
    def detect_bears(self, frame):
        """
        Detect bears in a single frame
        
        Args:
            frame: OpenCV image (numpy array)
            
        Returns:
            List of detections, each containing:
            - bbox: [x1, y1, x2, y2] coordinates
            - confidence: Detection confidence (0-1)
            - label: Class name
        """
        # Run inference
        results = self.model(frame, device=self.device, verbose=False)
        
        detections = []
        
        # Process results
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    confidence = float(boxes.conf[i])
                    class_id = int(boxes.cls[i])
                    class_name = self.model.names[class_id]
                    
                    # Check if it's a bear and meets confidence threshold
                    if 'bear' in class_name.lower() and confidence >= self.confidence_threshold:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                        
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'label': class_name
                        })
        
        return detections
    
    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes on frame
        
        Args:
            frame: OpenCV image
            detections: List of detection dictionaries
            
        Returns:
            Frame with drawn detections
        """
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            label = detection['label']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label with confidence
            text = f"{label}: {confidence:.2f}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Background for text
            cv2.rectangle(frame, (x1, y1 - text_size[1] - 4), 
                         (x1 + text_size[0], y1), (0, 255, 0), -1)
            
            # Draw text
            cv2.putText(frame, text, (x1, y1 - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return frame
    
    def process_video_stream(self, source=0, display=True):
        """
        Process video stream from camera or file
        
        Args:
            source: Camera index (0, 1, ...) or video file path
            display: Whether to show the video output
        """
        # Open video source
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video source {source}")
            return
        
        print("Processing video stream. Close window to exit.")
        
        frame_count = 0
        fps = 0
        prev_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                frame_count += 1
                
                # Calculate FPS
                current_time = time.time()
                if current_time - prev_time >= 1.0:
                    fps = frame_count / (current_time - prev_time)
                    frame_count = 0
                    prev_time = current_time
                
                # Detect bears
                detections = self.detect_bears(frame)
                
                # Display if requested
                if display:
                    # Draw detections on frame
                    display_frame = self.draw_detections(frame.copy(), detections)
                    
                    # Add FPS in bottom left corner
                    height = display_frame.shape[0]
                    fps_text = f"FPS: {fps:.1f}"
                    cv2.putText(display_frame, fps_text, (10, height - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Add "Bear detected." in top right corner if bears are present
                    if detections:
                        width = display_frame.shape[1]
                        bear_text = "Bear detected."
                        text_size = cv2.getTextSize(bear_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        text_x = width - text_size[0] - 10
                        cv2.putText(display_frame, bear_text, (text_x, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Show frame
                    cv2.imshow('Bear Detection', display_frame)
                    
                    # Check if window was closed
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                        break
                    
                    # Check if window still exists
                    if cv2.getWindowProperty('Bear Detection', cv2.WND_PROP_VISIBLE) < 1:
                        break
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()


def main():
    """Main function with command line interface"""
    
    parser = argparse.ArgumentParser(description='Bear Detection System')
    parser.add_argument('--source', type=str, default=0,
                       help='Video source: camera index (0,1,2) or video file path')
    parser.add_argument('--model', type=str, default='yolov8x.pt',
                       help='Path to YOLO model file')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Minimum confidence threshold (0-1)')
    parser.add_argument('--no-display', action='store_true',
                       help='Run without displaying video')
    
    args = parser.parse_args()
    
    # Convert source to int if it's a camera index
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
    
    print("Starting Bear Detection System")
    print(f"Configuration:")
    print(f"  Source: {source}")
    print(f"  Model: {args.model}")
    print(f"  Confidence threshold: {args.confidence}")
    print(f"  Display: {not args.no_display}")
    print()
    
    # Create detector
    detector = BearDetector(
        model_path=args.model,
        confidence_threshold=args.confidence
    )
    
    # Start processing
    detector.process_video_stream(
        source=source,
        display=not args.no_display
    )


if __name__ == "__main__":
    main()