"""
Real-Time Bear Detection System
A modular, high-performance computer vision system for detecting bears in video streams.
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time
import logging
from pathlib import Path
import json
from collections import deque
import threading
from queue import Queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Represents a single detection result"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_name: str
    timestamp: float
    frame_id: int


@dataclass
class SystemConfig:
    """System configuration parameters"""
    model_path: str = "yolov8m.pt"  # Medium model for balance
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.45
    target_fps: int = 30
    enable_gpu: bool = True
    detection_buffer_size: int = 30
    min_detection_area: int = 1000  # Minimum bbox area in pixels
    
    # Bear-specific parameters
    bear_class_names: List[str] = None
    
    def __post_init__(self):
        if self.bear_class_names is None:
            # Default bear-related class names from various datasets
            self.bear_class_names = ['bear', 'black bear', 'grizzly bear', 
                                    'polar bear', 'brown bear', 'animal']


class VideoSource(ABC):
    """Abstract base class for video sources"""
    
    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the video source"""
        pass
    
    @abstractmethod
    def release(self):
        """Release the video source"""
        pass
    
    @abstractmethod
    def get_fps(self) -> float:
        """Get the FPS of the video source"""
        pass


class CameraSource(VideoSource):
    """USB/Webcam video source"""
    
    def __init__(self, camera_id: int = 0, resolution: Tuple[int, int] = (1280, 720)):
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")
        
        logger.info(f"Camera initialized: {resolution[0]}x{resolution[1]}")
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        return self.cap.read()
    
    def release(self):
        self.cap.release()
    
    def get_fps(self) -> float:
        return self.cap.get(cv2.CAP_PROP_FPS) or 30.0


class RTSPSource(VideoSource):
    """RTSP stream video source (for drones)"""
    
    def __init__(self, rtsp_url: str):
        self.cap = cv2.VideoCapture(rtsp_url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open RTSP stream: {rtsp_url}")
        
        logger.info(f"RTSP stream connected: {rtsp_url}")
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        return self.cap.read()
    
    def release(self):
        self.cap.release()
    
    def get_fps(self) -> float:
        return self.cap.get(cv2.CAP_PROP_FPS) or 30.0


class BearDetector:
    """High-performance bear detection engine"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.device = 'cuda' if config.enable_gpu and torch.cuda.is_available() else 'cpu'
        
        # Initialize YOLO model
        self.model = YOLO(config.model_path)
        self.model.to(self.device)
        
        # Get class names from model
        self.class_names = self.model.names
        
        # Detection history for temporal filtering
        self.detection_history = deque(maxlen=config.detection_buffer_size)
        
        # Performance metrics
        self.fps_history = deque(maxlen=30)
        self.last_inference_time = 0
        
        logger.info(f"Bear detector initialized on {self.device}")
        logger.info(f"Model: {config.model_path}")
    
    def detect(self, frame: np.ndarray, frame_id: int) -> List[DetectionResult]:
        """
        Perform bear detection on a single frame
        
        Args:
            frame: Input frame (BGR format)
            frame_id: Unique frame identifier
            
        Returns:
            List of DetectionResult objects
        """
        start_time = time.time()
        
        # Run inference
        results = self.model(
            frame, 
            conf=self.config.confidence_threshold,
            iou=self.config.nms_threshold,
            device=self.device,
            verbose=False
        )
        
        detections = []
        timestamp = time.time()
        
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    # Extract detection info
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.class_names[class_id]
                    
                    # Filter for bear-related classes
                    if self._is_bear_detection(class_name):
                        # Apply minimum area filter
                        area = (x2 - x1) * (y2 - y1)
                        if area >= self.config.min_detection_area:
                            detection = DetectionResult(
                                bbox=(int(x1), int(y1), int(x2), int(y2)),
                                confidence=float(confidence),
                                class_name=class_name,
                                timestamp=timestamp,
                                frame_id=frame_id
                            )
                            detections.append(detection)
        
        # Update performance metrics
        self.last_inference_time = time.time() - start_time
        self.fps_history.append(1.0 / self.last_inference_time if self.last_inference_time > 0 else 0)
        
        # Add to history for temporal filtering
        self.detection_history.append(detections)
        
        return detections
    
    def _is_bear_detection(self, class_name: str) -> bool:
        """Check if the detected class is bear-related"""
        class_name_lower = class_name.lower()
        
        # Check against configured bear class names
        for bear_class in self.config.bear_class_names:
            if bear_class.lower() in class_name_lower:
                return True
        
        # Additional heuristic: if using COCO model, bears might be detected as "teddy bear"
        # but we should be careful with this
        if 'bear' in class_name_lower and 'teddy' not in class_name_lower:
            return True
        
        return False
    
    def get_filtered_detections(self) -> List[DetectionResult]:
        """
        Apply temporal filtering to reduce false positives
        Returns detections that appear consistently across frames
        """
        if len(self.detection_history) < 3:
            return self.detection_history[-1] if self.detection_history else []
        
        # Simple temporal filtering: require detection in at least 2 of last 3 frames
        recent_detections = list(self.detection_history)[-3:]
        detection_count = sum(1 for d in recent_detections if len(d) > 0)
        
        if detection_count >= 2:
            return self.detection_history[-1]
        return []
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        return {
            'avg_fps': np.mean(self.fps_history) if self.fps_history else 0,
            'last_inference_ms': self.last_inference_time * 1000,
            'device': self.device
        }


class DetectionPipeline:
    """Main detection pipeline orchestrator"""
    
    def __init__(self, video_source: VideoSource, config: SystemConfig):
        self.video_source = video_source
        self.config = config
        self.detector = BearDetector(config)
        
        # Processing state
        self.is_running = False
        self.frame_count = 0
        self.detection_count = 0
        
        # Performance tracking
        self.start_time = None
        self.total_processing_time = 0
        
        # Event callbacks (for extensibility)
        self.on_detection_callbacks = []
        
    def register_detection_callback(self, callback):
        """Register a callback for detection events"""
        self.on_detection_callbacks.append(callback)
    
    def _trigger_detection_callbacks(self, detections: List[DetectionResult], frame: np.ndarray):
        """Trigger all registered detection callbacks"""
        for callback in self.on_detection_callbacks:
            try:
                callback(detections, frame)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def _draw_detections(self, frame: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
        """Draw bounding boxes and labels on frame"""
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            
            # Draw bounding box
            color = (0, 0, 255)  # Red for bear detection
            thickness = 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label with confidence
            label = f"BEAR: {detection.confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Background for label
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw text
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add alert indicator for high confidence detections
            if detection.confidence > 0.7:
                cv2.putText(frame, "⚠ BEAR DETECTED ⚠", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return frame
    
    def _draw_metrics(self, frame: np.ndarray, metrics: Dict[str, float]) -> np.ndarray:
        """Draw performance metrics on frame"""
        y_offset = frame.shape[0] - 60
        
        # FPS counter
        fps_text = f"FPS: {metrics['avg_fps']:.1f}"
        cv2.putText(frame, fps_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Inference time
        inference_text = f"Inference: {metrics['last_inference_ms']:.1f}ms"
        cv2.putText(frame, inference_text, (10, y_offset + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Device
        device_text = f"Device: {metrics['device'].upper()}"
        cv2.putText(frame, device_text, (10, y_offset + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def run(self, display: bool = True, save_video: Optional[str] = None):
        """
        Run the detection pipeline
        
        Args:
            display: Whether to display the video with detections
            save_video: Path to save output video (optional)
        """
        self.is_running = True
        self.start_time = time.time()
        
        # Video writer setup if saving
        video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = self.video_source.get_fps()
            video_writer = None  # Will initialize after getting first frame
        
        logger.info("Starting detection pipeline...")
        
        try:
            while self.is_running:
                # Read frame
                ret, frame = self.video_source.read()
                if not ret:
                    logger.warning("Failed to read frame")
                    break
                
                # Initialize video writer with frame dimensions
                if save_video and video_writer is None:
                    h, w = frame.shape[:2]
                    video_writer = cv2.VideoWriter(save_video, fourcc, fps, (w, h))
                
                # Perform detection
                detections = self.detector.detect(frame, self.frame_count)
                
                # Apply temporal filtering
                filtered_detections = self.detector.get_filtered_detections()
                
                # Update counters
                self.frame_count += 1
                if filtered_detections:
                    self.detection_count += len(filtered_detections)
                    self._trigger_detection_callbacks(filtered_detections, frame)
                
                # Get performance metrics
                metrics = self.detector.get_performance_metrics()
                
                # Visualization
                if display or save_video:
                    display_frame = frame.copy()
                    display_frame = self._draw_detections(display_frame, filtered_detections)
                    display_frame = self._draw_metrics(display_frame, metrics)
                    
                    if save_video:
                        video_writer.write(display_frame)
                    
                    if display:
                        cv2.imshow('Bear Detection System', display_frame)
                        
                        # Check for exit
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                        elif key == ord('s'):  # Screenshot
                            cv2.imwrite(f'bear_detection_{self.frame_count}.jpg', display_frame)
                            logger.info(f"Screenshot saved: bear_detection_{self.frame_count}.jpg")
                
                # Log periodic stats
                if self.frame_count % 100 == 0:
                    elapsed = time.time() - self.start_time
                    logger.info(f"Processed {self.frame_count} frames in {elapsed:.1f}s, "
                               f"{self.detection_count} total detections, "
                               f"Avg FPS: {metrics['avg_fps']:.1f}")
        
        except KeyboardInterrupt:
            logger.info("Detection stopped by user")
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
        finally:
            self.is_running = False
            
            # Cleanup
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
            
            # Final stats
            if self.start_time:
                total_time = time.time() - self.start_time
                avg_fps = self.frame_count / total_time if total_time > 0 else 0
                logger.info(f"\n=== Final Statistics ===")
                logger.info(f"Total frames: {self.frame_count}")
                logger.info(f"Total detections: {self.detection_count}")
                logger.info(f"Total time: {total_time:.1f}s")
                logger.info(f"Average FPS: {avg_fps:.1f}")
    
    def stop(self):
        """Stop the detection pipeline"""
        self.is_running = False


class BearDetectionSystem:
    """Main system controller with easy API"""
    
    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        self.pipeline = None
        self.video_source = None
    
    def start_camera_detection(self, camera_id: int = 0, 
                              resolution: Tuple[int, int] = (1280, 720),
                              display: bool = True):
        """
        Start detection from USB camera
        
        Args:
            camera_id: Camera device ID
            resolution: Camera resolution
            display: Whether to display video output
        """
        try:
            # Initialize camera source
            self.video_source = CameraSource(camera_id, resolution)
            
            # Create pipeline
            self.pipeline = DetectionPipeline(self.video_source, self.config)
            
            # Optional: Add detection logger
            def log_detection(detections, frame):
                for d in detections:
                    logger.warning(f"🐻 BEAR DETECTED! Confidence: {d.confidence:.2f}, "
                                 f"Location: {d.bbox}")
            
            self.pipeline.register_detection_callback(log_detection)
            
            # Run pipeline
            self.pipeline.run(display=display)
            
        finally:
            if self.video_source:
                self.video_source.release()
    
    def start_rtsp_detection(self, rtsp_url: str, display: bool = True):
        """
        Start detection from RTSP stream (e.g., drone)
        
        Args:
            rtsp_url: RTSP stream URL
            display: Whether to display video output
        """
        try:
            # Initialize RTSP source
            self.video_source = RTSPSource(rtsp_url)
            
            # Create pipeline
            self.pipeline = DetectionPipeline(self.video_source, self.config)
            
            # Run pipeline
            self.pipeline.run(display=display)
            
        finally:
            if self.video_source:
                self.video_source.release()
    
    def stop(self):
        """Stop the detection system"""
        if self.pipeline:
            self.pipeline.stop()


# Example usage and testing
def main():
    """Main entry point for the bear detection system"""
    
    # Configure system
    config = SystemConfig(
        model_path="yolov8m.pt",  # Use medium model for balance
        confidence_threshold=0.45,  # Lower threshold to catch more potential bears
        nms_threshold=0.45,
        target_fps=30,
        enable_gpu=True,
        min_detection_area=2000  # Adjust based on expected bear size
    )
    
    # Create system
    system = BearDetectionSystem(config)
    
    # Start detection from default camera
    print("Starting Bear Detection System...")
    print("Press 'q' to quit, 's' to save screenshot")
    print("-" * 50)
    
    try:
        # For camera feed
        system.start_camera_detection(camera_id=0, display=True)
        
        # For RTSP drone feed (uncomment to use)
        # system.start_rtsp_detection("rtsp://drone_ip:port/stream", display=True)
        
    except Exception as e:
        logger.error(f"System error: {e}")
        print(f"Error: {e}")
        print("Make sure your camera is connected and not in use by another application")


if __name__ == "__main__":
    main()