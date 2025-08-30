"""
Setup script for Bear Detection System
Downloads YOLO model and verifies installation
"""

import subprocess
import sys
from pathlib import Path


def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'ultralytics',
        'opencv-python',
        'torch',
        'numpy'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print("Missing packages detected:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nPlease run: pip install -r requirements.txt")
        return False
    
    print("All dependencies installed successfully")
    return True


def download_model(model_name='yolov8x.pt'):
    """Download YOLO model if not present"""
    from ultralytics import YOLO
    
    print(f"Checking for model: {model_name}")
    
    # This will automatically download if not present
    try:
        model = YOLO(model_name)
        print(f"Model {model_name} is ready")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


def test_camera():
    """Test if camera is accessible"""
    import cv2
    
    print("Testing camera access...")
    cap = cv2.VideoCapture(0)
    
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret:
            print("Camera test successful")
            return True
    
    print("Warning: Could not access camera")
    print("You can still use video files as input")
    return False


def check_gpu():
    """Check if GPU is available"""
    import torch
    
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        return True
    else:
        print("No GPU detected - will use CPU (slower performance)")
        return False


def main():
    """Run setup process"""
    print("Bear Detection System - Setup")
    print("=" * 40)
    
    # Check dependencies
    print("\n1. Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    # Check GPU
    print("\n2. Checking hardware...")
    check_gpu()
    
    # Download model
    print("\n3. Setting up YOLO model...")
    if not download_model():
        print("Failed to download model")
        sys.exit(1)
    
    # Test camera
    print("\n4. Testing camera...")
    test_camera()
    
    print("\n" + "=" * 40)
    print("Setup complete!")
    print("\nTo start detection, run:")
    print("  python bear_detection.py")
    print("\nFor help:")
    print("  python bear_detection.py --help")


if __name__ == "__main__":
    main()