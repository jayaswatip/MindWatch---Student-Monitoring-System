import cv2
import numpy as np
import torch
from collections import defaultdict

class MindWatchAnalyzer:
    def __init__(self, model_path: str):
        """
        Initialize MindWatch Analyzer with local YOLO model
        """
        self.model_path = model_path

        # Load YOLO model
        print(f"Loading model from: {model_path}")
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.model_type = "yolov8"
            print("✓ YOLOv8 model loaded successfully")
        except ImportError:
            try:
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
                self.model_type = "yolov5"
                print("✓ YOLOv5 model loaded successfully")
            except Exception as e:
                print(f"Error loading model: {e}")
                raise
