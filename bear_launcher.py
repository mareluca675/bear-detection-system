"""
Bear Detection System - Quick Start Launcher
Simple script to quickly start bear detection with optimal settings
"""

import sys
import os
import argparse
from pathlib import Path

# Attempt to import required modules
try:
    import cv2
    import torch
    from ultralytics import YOLO
    import numpy as np
except ImportError as e:
    print(f"Error: Missing required module - {e}")
    print("\nPlease install dependencies first:")
    print("pip install opencv-python ultralytics torch torchvision numpy")
    sys.exit(1)


def quick_detect():
    """Quick detection with minimal setup - just works!"""
    
    print("=" * 60)
    print("🐻 BEAR DETECTION SYSTEM - QUICK START")
    print("=" * 60)
    
    # Auto-download model if not present
    model_path = Path("yolov8x.pt")
    if not model_path.exists():
        print("Downloading detection model (one-time setup)...")
        model = YOLO("yolov8x.pt")  # Auto-downloads
    else:
        model = YOLO(model_path)
    
    # Configure for bear detection
    print(f"Model loaded: YOLOv8m")
    print(f"Device: {'CUDA (GPU)' if torch.cuda.is_available() else 'CPU'}")
    
    # Open default camera
    print("\nStarting camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        print("Please check:")
        print("  1. Camera is connected")
        print("  2. Camera permissions are granted")
        print("  3. No other app is using the camera")
        return
    
    # Set resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("\n✓ Camera ready!")
    print("\nControls:")
    print("  • Press 'q' to quit")
    print("  • Press 's' to save screenshot")
    print("  • Press 'r' to reset detection counter")
    print("\n" + "=" * 60)
    print("Monitoring for bears...\n")
    
    # Detection counters
    frame_count = 0
    detection_count = 0
    bear_detected = False
    last_detection_frame = 0
    
    # Performance tracking
    fps_counter = []
    import time
    
    # Bear-related class names to look for
    bear_keywords = ['bear', 'animal', 'mammal']
    
    try:
        while True:
            start_time = time.time()
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to read frame")
                continue
            
            frame_count += 1
            
            # Run detection
            results = model(frame, conf=0.4, verbose=False)
            
            # Process detections
            current_bear_detected = False
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        # Get class name
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id].lower()
                        confidence = float(box.conf[0])
                        
                        # Check if it might be a bear
                        is_bear = any(keyword in class_name for keyword in bear_keywords)
                        
                        if is_bear or 'bear' in class_name:
                            current_bear_detected = True
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            # Draw detection
                            color = (0, 0, 255)  # Red
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                            
                            # Label
                            label = f"BEAR {confidence:.2f}"
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                                        (x1 + label_size[0], y1), color, -1)
                            cv2.putText(frame, label, (x1, y1 - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            
                            # Alert
                            if frame_count - last_detection_frame > 30:  # New detection
                                detection_count += 1
                                last_detection_frame = frame_count
                                print(f"⚠️  BEAR DETECTED! Count: {detection_count} | Confidence: {confidence:.2%}")
            
            # Update bear status
            bear_detected = current_bear_detected
            
            # Draw status overlay
            status_color = (0, 0, 255) if bear_detected else (0, 255, 0)
            status_text = "⚠ BEAR DETECTED ⚠" if bear_detected else "✓ CLEAR"
            
            # Status background
            cv2.rectangle(frame, (10, 10), (250, 70), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (250, 70), status_color, 2)
            cv2.putText(frame, status_text, (20, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            # FPS counter
            fps = 1.0 / (time.time() - start_time)
            fps_counter.append(fps)
            if len(fps_counter) > 30:
                fps_counter.pop(0)
            avg_fps = sum(fps_counter) / len(fps_counter)
            
            # Performance overlay
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, frame.shape[0] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Detections: {detection_count}", (10, frame.shape[0] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Bear Detection System', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"bear_detection_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot saved: {filename}")
            elif key == ord('r'):
                detection_count = 0
                print("🔄 Detection counter reset")
    
    except KeyboardInterrupt:
        print("\n\nStopping detection...")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 60)
        print("Session Summary:")
        print(f"  • Total frames processed: {frame_count}")
        print(f"  • Bears detected: {detection_count} times")
        print(f"  • Average FPS: {avg_fps:.1f}")
        print("=" * 60)


if __name__ == "__main__":
    quick_detect()