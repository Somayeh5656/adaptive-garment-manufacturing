#!/usr/bin/env python3
"""
yolos_detector.py
-----------------
Wrapper around the YOLOS‑Fashionpedia model for fabric detection.
Used by the perception node (perception_node.py) to locate fabric in images
from the Gazebo camera.  Defect detection is currently simulated.
"""

import torch
from transformers import YolosImageProcessor, YolosForObjectDetection
import numpy as np
import cv2
from PIL import Image


class YOLOSDetector:
    """Loads YOLOS fine‑tuned on Fashionpedia and provides detection methods."""

    def __init__(self):
        print("Loading YOLOS model...")
        self.processor = YolosImageProcessor.from_pretrained(
            "valentinafeve/yolos-fashionpedia"
        )
        self.model = YolosForObjectDetection.from_pretrained(
            "valentinafeve/yolos-fashionpedia"
        )
        self.model.eval()

        # Possible defect categories (simulated for now)
        self.defect_categories = ['stain', 'hole', 'tear', 'weaving_error']
        print("YOLOS model loaded!")

    def detect_fabric(self, image):
        """
        Run object detection on an image (OpenCV BGR or PIL).
        Returns a list of dicts with category, confidence, bbox, and area.
        """
        # Convert OpenCV BGR to PIL RGB if needed
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = image

        # Preprocess and run inference
        inputs = self.processor(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post‑process to get bounding boxes and labels
        target_sizes = torch.tensor([pil_image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.5
        )[0]

        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            category = self.model.config.id2label[label.item()]
            detections.append({
                'category': category,
                'confidence': score.item(),
                'bbox': box.tolist(),
                'area': (box[2] - box[0]) * (box[3] - box[1])
            })

        return detections

    def detect_defects(self, image):
        """
        Simulate fabric defect detection.
        In a real deployment, a separate defect detection model would be used.
        """
        height, width = image.shape[:2]
        defects = []

        # 30% chance to generate a random defect for demonstration
        if np.random.random() > 0.7:
            defect_type = np.random.choice(self.defect_categories)
            x = np.random.randint(20, width - 70)
            y = np.random.randint(20, height - 70)
            w = np.random.randint(30, 60)
            h = np.random.randint(30, 60)

            defects.append({
                'type': defect_type,
                'bbox': [x, y, w, h],
                'confidence': np.random.random() * 0.5 + 0.5
            })

        return defects