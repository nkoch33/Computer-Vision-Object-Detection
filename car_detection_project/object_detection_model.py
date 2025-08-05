import cv2
import numpy as np
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
import time
from typing import List, Tuple, Dict, Any
import json

class CarObjectDetector:
    """
    A comprehensive object detection model for detecting cars and surrounding objects
    in real-world scenarios using YOLO (You Only Look Once).
    """
    
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.5):
        """
        Initialize the object detector.
        
        Args:
            model_path: Path to custom trained model (optional)
            confidence_threshold: Minimum confidence score for detections
        """
        self.confidence_threshold = confidence_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Initialize YOLO model
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            # Use pre-trained YOLO model
            self.model = YOLO('yolov8n.pt')
        
        # Define classes we're interested in (COCO dataset classes)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
        # Focus on transportation and relevant objects
        self.vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'airplane', 'train', 'boat']
        self.traffic_classes = ['traffic light', 'stop sign', 'parking meter']
        self.person_class = ['person']
        
    def detect_objects(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in the given image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of detection dictionaries with bounding boxes, confidence, and class info
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
                    class_name = self.class_names[class_id]
                    
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
        Analyze the scene and provide detailed information about detected objects.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing scene analysis
        """
        detections = self.detect_objects(image)
        
        # Categorize detections
        vehicles = [d for d in detections if d['class_name'] in self.vehicle_classes]
        traffic_signals = [d for d in detections if d['class_name'] in self.traffic_classes]
        people = [d for d in detections if d['class_name'] in self.person_class]
        other_objects = [d for d in detections if d['class_name'] not in 
                        self.vehicle_classes + self.traffic_classes + self.person_class]
        
        # Calculate statistics
        analysis = {
            'total_objects': len(detections),
            'vehicles': {
                'count': len(vehicles),
                'types': {v['class_name']: len([x for x in vehicles if x['class_name'] == v['class_name']]) 
                         for v in vehicles},
                'detections': vehicles
            },
            'traffic_signals': {
                'count': len(traffic_signals),
                'types': {t['class_name']: len([x for x in traffic_signals if x['class_name'] == t['class_name']]) 
                         for t in traffic_signals},
                'detections': traffic_signals
            },
            'people': {
                'count': len(people),
                'detections': people
            },
            'other_objects': {
                'count': len(other_objects),
                'detections': other_objects
            },
            'scene_summary': self._generate_scene_summary(vehicles, traffic_signals, people)
        }
        
        return analysis
    
    def _generate_scene_summary(self, vehicles: List, traffic_signals: List, people: List) -> str:
        """Generate a natural language summary of the scene."""
        summary_parts = []
        
        if vehicles:
            vehicle_types = [v['class_name'] for v in vehicles]
            vehicle_counts = {}
            for vt in vehicle_types:
                vehicle_counts[vt] = vehicle_counts.get(vt, 0) + 1
            
            vehicle_desc = []
            for vt, count in vehicle_counts.items():
                vehicle_desc.append(f"{count} {vt}{'s' if count > 1 else ''}")
            
            summary_parts.append(f"Detected {len(vehicles)} vehicle(s): {', '.join(vehicle_desc)}")
        
        if traffic_signals:
            signal_types = [t['class_name'] for t in traffic_signals]
            summary_parts.append(f"Found {len(traffic_signals)} traffic signal(s): {', '.join(set(signal_types))}")
        
        if people:
            summary_parts.append(f"Detected {len(people)} person(s)")
        
        if not summary_parts:
            summary_parts.append("No significant objects detected")
        
        return ". ".join(summary_parts) + "."
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict], 
                       show_labels: bool = True, show_confidence: bool = True) -> np.ndarray:
        """
        Draw bounding boxes and labels on the image.
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            show_labels: Whether to show class labels
            show_confidence: Whether to show confidence scores
            
        Returns:
            Image with drawn detections
        """
        img_copy = image.copy()
        
        # Define colors for different object types
        colors = {
            'car': (0, 255, 0),      # Green
            'truck': (0, 255, 255),   # Yellow
            'bus': (255, 0, 0),       # Blue
            'motorcycle': (255, 0, 255), # Magenta
            'bicycle': (128, 0, 128), # Purple
            'person': (0, 165, 255),  # Orange
            'traffic light': (255, 255, 0), # Cyan
            'stop sign': (0, 0, 255), # Red
            'default': (255, 255, 255) # White
        }
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Get color for this class
            color = colors.get(class_name, colors['default'])
            
            # Draw bounding box
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            label_parts = []
            if show_labels:
                label_parts.append(class_name)
            if show_confidence:
                label_parts.append(f"{confidence:.2f}")
            
            if label_parts:
                label = " ".join(label_parts)
                
                # Calculate label position and size
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                # Draw label background
                cv2.rectangle(img_copy, (x1, y1 - label_height - 10), 
                            (x1 + label_width, y1), color, -1)
                
                # Draw label text
                cv2.putText(img_copy, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return img_copy
    
    def process_video(self, video_path: str, output_path: str = None, 
                     show_analysis: bool = True) -> None:
        """
        Process a video file and detect objects in each frame.
        
        Args:
            video_path: Path to input video file
            output_path: Path to save output video (optional)
            show_analysis: Whether to display analysis information
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects in frame
            detections = self.detect_objects(frame)
            
            # Draw detections on frame
            annotated_frame = self.draw_detections(frame, detections)
            
            # Add frame information
            if show_analysis:
                analysis = self.analyze_scene(frame)
                info_text = [
                    f"Frame: {frame_count}",
                    f"Objects: {analysis['total_objects']}",
                    f"Vehicles: {analysis['vehicles']['count']}",
                    f"People: {analysis['people']['count']}"
                ]
                
                for i, text in enumerate(info_text):
                    cv2.putText(annotated_frame, text, (10, 30 + i * 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame to output video
            if writer:
                writer.write(annotated_frame)
            
            # Display frame
            cv2.imshow('Object Detection', annotated_frame)
            
            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        # Clean up
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Print processing statistics
        processing_time = time.time() - start_time
        print(f"Processed {frame_count} frames in {processing_time:.2f} seconds")
        print(f"Average FPS: {frame_count / processing_time:.2f}")
    
    def process_image(self, image_path: str, output_path: str = None, 
                     show_analysis: bool = True) -> Dict[str, Any]:
        """
        Process a single image and detect objects.
        
        Args:
            image_path: Path to input image
            output_path: Path to save annotated image (optional)
            show_analysis: Whether to print analysis information
            
        Returns:
            Analysis dictionary
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return {}
        
        # Detect objects
        detections = self.detect_objects(image)
        
        # Analyze scene
        analysis = self.analyze_scene(image)
        
        # Draw detections
        annotated_image = self.draw_detections(image, detections)
        
        # Save annotated image
        if output_path:
            cv2.imwrite(output_path, annotated_image)
            print(f"Annotated image saved to: {output_path}")
        
        # Display results
        if show_analysis:
            print("\n=== Scene Analysis ===")
            print(f"Total objects detected: {analysis['total_objects']}")
            print(f"Vehicles: {analysis['vehicles']['count']}")
            print(f"Traffic signals: {analysis['traffic_signals']['count']}")
            print(f"People: {analysis['people']['count']}")
            print(f"Other objects: {analysis['other_objects']['count']}")
            print(f"\nScene summary: {analysis['scene_summary']}")
        
        return analysis

def main():
    """Example usage of the CarObjectDetector."""
    
    # Initialize detector
    detector = CarObjectDetector(confidence_threshold=0.5)
    
    print("Car Object Detection Model")
    print("=" * 30)
    print("This model can detect:")
    print("- Vehicles (cars, trucks, buses, motorcycles, etc.)")
    print("- Traffic signals and signs")
    print("- People and pedestrians")
    print("- Other objects in the environment")
    print("\nUsage examples:")
    print("1. Process image: detector.process_image('path/to/image.jpg')")
    print("2. Process video: detector.process_video('path/to/video.mp4')")
    print("3. Real-time webcam: detector.process_video(0)")  # 0 for webcam
    
    # Example: Process a test image if available
    test_image_path = "test_image.jpg"
    if os.path.exists(test_image_path):
        print(f"\nProcessing test image: {test_image_path}")
        analysis = detector.process_image(test_image_path, "output_annotated.jpg")
    else:
        print(f"\nNo test image found at {test_image_path}")
        print("You can add your own images to test the model.")

if __name__ == "__main__":
    main() 