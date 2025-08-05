#!/usr/bin/env python3
"""
Your Computer Vision Project Template
====================================

This is a template for creating your own computer vision project.
Modify this file to suit your specific needs.
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os
import time
from typing import List, Dict, Any

class YourDetector:
    """
    Your custom object detector.
    
    TODO: Modify this class for your specific project:
    1. Change the class name to match your project
    2. Update the target_classes list
    3. Modify the colors dictionary
    4. Customize the analysis logic
    5. Update the visualization
    """
    
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize your detector.
        
        Args:
            confidence_threshold: Minimum confidence score for detections
        """
        self.confidence_threshold = confidence_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Initialize YOLO model
        self.model = YOLO('yolov8n.pt')
        
        # TODO: Update these for your project
        # Define which classes you want to detect
        self.target_classes = [
            'person',  # Example: detect people
            'car',     # Example: detect cars
            # Add more classes as needed
        ]
        
        # TODO: Update colors for your classes
        self.colors = {
            'person': (0, 255, 0),    # Green
            'car': (255, 0, 0),       # Blue
            'default': (255, 255, 255) # White
        }
    
    def detect_objects(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in the given image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of detection dictionaries
        """
        # Run inference
        results = self.model(image, conf=self.confidence_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Get confidence and class
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]
                    
                    # TODO: Filter for your target classes
                    if class_name in self.target_classes:
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': class_name,
                            'area': (x2 - x1) * (y2 - y1)
                        }
                        detections.append(detection)
        
        return detections
    
    def analyze_scene(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the scene and provide detailed information.
        
        TODO: Customize this analysis for your project needs.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary with analysis results
        """
        detections = self.detect_objects(image)
        
        # TODO: Customize your analysis logic
        # Example: Count different types of objects
        object_counts = {}
        for detection in detections:
            class_name = detection['class_name']
            object_counts[class_name] = object_counts.get(class_name, 0) + 1
        
        # Calculate average confidence
        avg_confidence = 0
        if detections:
            avg_confidence = sum(d['confidence'] for d in detections) / len(detections)
        
        # TODO: Generate custom summary
        total_objects = len(detections)
        if total_objects == 0:
            summary = "No objects detected."
        else:
            summary = f"Detected {total_objects} objects with {avg_confidence:.2f} average confidence."
        
        return {
            'total_objects': total_objects,
            'object_counts': object_counts,
            'detections': detections,
            'average_confidence': avg_confidence,
            'summary': summary
        }
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict], 
                       show_labels: bool = True, show_confidence: bool = True) -> np.ndarray:
        """
        Draw bounding boxes around detected objects.
        
        TODO: Customize visualization for your project.
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            show_labels: Whether to show class labels
            show_confidence: Whether to show confidence scores
            
        Returns:
            Image with drawn bounding boxes
        """
        annotated_image = image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Get color for this class
            color = self.colors.get(class_name, self.colors['default'])
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            label_parts = []
            if show_labels:
                label_parts.append(class_name)
            if show_confidence:
                label_parts.append(f"{confidence:.2f}")
            
            if label_parts:
                label = " ".join(label_parts)
                
                # Calculate label position
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                label_y = y1 - 10 if y1 - 10 > label_size[1] else y1 + label_size[1]
                
                # Draw label background
                cv2.rectangle(annotated_image, (x1, label_y - label_size[1] - 10),
                             (x1 + label_size[0], label_y), color, -1)
                
                # Draw label text
                cv2.putText(annotated_image, label, (x1, label_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated_image
    
    def process_image(self, image_path: str, output_path: str = None) -> Dict[str, Any]:
        """
        Process a single image.
        
        Args:
            image_path: Path to input image
            output_path: Path to save annotated image (optional)
            
        Returns:
            Analysis results dictionary
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Analyze scene
        analysis = self.analyze_scene(image)
        
        # Draw detections
        annotated_image = self.draw_detections(image, analysis['detections'])
        
        # Save annotated image
        if output_path:
            cv2.imwrite(output_path, annotated_image)
            print(f"Annotated image saved to: {output_path}")
        
        return analysis

def main():
    """Example usage of your detector."""
    # Initialize detector
    detector = YourDetector(confidence_threshold=0.5)
    
    # Example: Process an image
    image_path = "test_image.jpg"  # Replace with your image path
    
    if os.path.exists(image_path):
        print(f"Processing image: {image_path}")
        analysis = detector.process_image(image_path, "output.jpg")
        
        print(f"Analysis Results:")
        print(f"  - Total objects: {analysis['total_objects']}")
        print(f"  - Object counts: {analysis['object_counts']}")
        print(f"  - Average confidence: {analysis['average_confidence']:.3f}")
        print(f"  - Summary: {analysis['summary']}")
    else:
        print(f"Image not found: {image_path}")
        print("Please provide a valid image path to test your detector.")

if __name__ == "__main__":
    main() 