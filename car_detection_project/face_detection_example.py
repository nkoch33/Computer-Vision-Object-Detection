#!/usr/bin/env python3
"""
Face Detection Project Example
==============================

This is an example of how to modify the car detection project
for a different use case - face detection.
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os
import time
from typing import List, Dict, Any

class FaceDetector:
    """
    A face detection system based on the car detection project structure.
    """
    
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize the face detector.
        
        Args:
            confidence_threshold: Minimum confidence score for detections
        """
        self.confidence_threshold = confidence_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Initialize YOLO model (YOLOv8 can detect people, which includes faces)
        self.model = YOLO('yolov8n.pt')
        
        # Focus on person class (which includes faces)
        self.target_classes = ['person']
        
        # Colors for visualization
        self.colors = {
            'person': (0, 255, 0),  # Green for faces
            'default': (255, 255, 255)  # White for other objects
        }
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in the given image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of face detection dictionaries
        """
        # Run inference
        results = self.model(image, conf=self.confidence_threshold)
        
        face_detections = []
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
                    
                    # Only process person detections (which include faces)
                    if class_name in self.target_classes:
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': class_name,
                            'area': (x2 - x1) * (y2 - y1)
                        }
                        face_detections.append(detection)
        
        return face_detections
    
    def analyze_faces(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the scene and provide information about detected faces.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary with analysis results
        """
        detections = self.detect_faces(image)
        
        # Count faces
        face_count = len(detections)
        
        # Calculate average confidence
        avg_confidence = 0
        if face_count > 0:
            avg_confidence = sum(d['confidence'] for d in detections) / face_count
        
        # Generate summary
        if face_count == 0:
            summary = "No faces detected in the image."
        elif face_count == 1:
            summary = f"Detected 1 face with {avg_confidence:.2f} average confidence."
        else:
            summary = f"Detected {face_count} faces with {avg_confidence:.2f} average confidence."
        
        return {
            'total_faces': face_count,
            'detections': detections,
            'average_confidence': avg_confidence,
            'summary': summary
        }
    
    def draw_faces(self, image: np.ndarray, detections: List[Dict], 
                   show_labels: bool = True, show_confidence: bool = True) -> np.ndarray:
        """
        Draw bounding boxes around detected faces.
        
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
        Process a single image and detect faces.
        
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
        
        # Analyze faces
        analysis = self.analyze_faces(image)
        
        # Draw detections
        annotated_image = self.draw_faces(image, analysis['detections'])
        
        # Save annotated image
        if output_path:
            cv2.imwrite(output_path, annotated_image)
            print(f"Annotated image saved to: {output_path}")
        
        return analysis

def main():
    """Example usage of the face detector."""
    # Initialize detector
    detector = FaceDetector(confidence_threshold=0.5)
    
    # Example: Process an image
    image_path = "test_image.jpg"  # Replace with your image path
    
    if os.path.exists(image_path):
        print(f"Processing image: {image_path}")
        analysis = detector.process_image(image_path, "face_detection_output.jpg")
        
        print(f"Analysis Results:")
        print(f"  - Total faces: {analysis['total_faces']}")
        print(f"  - Average confidence: {analysis['average_confidence']:.3f}")
        print(f"  - Summary: {analysis['summary']}")
    else:
        print(f"Image not found: {image_path}")
        print("Please provide a valid image path to test the face detector.")

if __name__ == "__main__":
    main() 