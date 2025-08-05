#!/usr/bin/env python3
"""
Car Detection Project Demo
==========================

This script demonstrates all the features of the car detection project:
- Real-time webcam detection
- Image processing
- Video processing
- Analytics and reporting
- Web interface
- Model training capabilities
"""

import os
import sys
import time
import argparse
from pathlib import Path

def print_banner():
    """Print project banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    Car Detection Project                  ║
    ║                                                              ║
    ║  A comprehensive computer vision system for detecting cars   ║
    ║  and surrounding objects using YOLO object detection.       ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def demo_basic_detection():
    """Demonstrate basic object detection."""
    print("\nDemo 1: Basic Object Detection")
    print("=" * 50)
    
    try:
        from object_detection_model import CarObjectDetector
        
        # Initialize detector
        print("Initializing detector...")
        detector = CarObjectDetector(confidence_threshold=0.5)
        print("Detector initialized successfully!")
        
        # Test with a sample image if available
        test_image = "test_image.jpg"
        if os.path.exists(test_image):
            print(f"Processing test image: {test_image}")
            analysis = detector.process_image(test_image, "demo_output.jpg")
            print(f"Analysis completed!")
            print(f"   - Total objects: {analysis['total_objects']}")
            print(f"   - Vehicles: {analysis['vehicles']['count']}")
            print(f"   - People: {analysis['people']['count']}")
            print(f"   - Summary: {analysis['scene_summary']}")
        else:
            print("No test image found. Create a 'test_image.jpg' to test detection.")
        
    except Exception as e:
        print(f"Error in basic detection demo: {e}")

def demo_analytics():
    """Demonstrate analytics features."""
    print("\nDemo 2: Analytics and Reporting")
    print("=" * 50)
    
    try:
        from analytics import DetectionAnalytics
        import cv2
        import numpy as np
        
        # Initialize analytics
        analytics = DetectionAnalytics()
        
        # Simulate some detection data
        print("Simulating detection data...")
        for i in range(10):
            # Simulate frame data
            frame_data = {
                'frame_number': i,
                'detections': [
                    {'class_name': 'car', 'confidence': 0.8 + i * 0.02},
                    {'class_name': 'person', 'confidence': 0.7 + i * 0.01},
                    {'class_name': 'truck', 'confidence': 0.6 + i * 0.03}
                ],
                'processing_time': 0.05 + i * 0.01
            }
            analytics.record_detection(frame_data)
        
        # Generate summary
        summary = analytics.generate_detection_summary()
        print("Analytics summary generated!")
        print(f"   - Total frames: {summary['total_frames_processed']}")
        print(f"   - Total detections: {summary['total_detections']}")
        print(f"   - Average processing time: {summary['performance_stats']['mean_processing_time']:.4f}s")
        
        # Generate visualizations
        print("Generating visualizations...")
        analytics.generate_visualizations()
        print("Visualizations created!")
        
        # Export data
        print("Exporting data...")
        analytics.export_data('json')
        analytics.export_data('csv')
        print("Data exported!")
        
        # Generate report
        print("Generating HTML report...")
        report_path = analytics.generate_report()
        print(f"Report generated: {report_path}")
        
    except Exception as e:
        print(f"Error in analytics demo: {e}")

def demo_web_interface():
    """Demonstrate web interface."""
    print("\nDemo 3: Web Interface")
    print("=" * 50)
    
    print("The web interface provides:")
    print("  - Real-time webcam detection")
    print("  - Image upload and processing")
    print("  - Live analytics dashboard")
    print("  - Data export capabilities")
    print("  - HTML report generation")
    print("\nTo start the web interface:")
    print("  python web_interface.py")
    print("Then open: http://localhost:8080")

def demo_model_training():
    """Demonstrate model training capabilities."""
    print("\nDemo 4: Model Training")
    print("=" * 50)
    
    try:
        from train_model import ModelTrainer
        
        # Create dataset template
        print("Creating dataset template...")
        trainer = ModelTrainer("dummy")
        trainer.create_dataset_template("demo_dataset")
        print("Dataset template created in 'demo_dataset/'")
        print("   - Use this template to organize your training data")
        print("   - Follow the YOLO format for labels")
        print("   - Update data.yaml with your class names")
        
        print("\nTraining workflow:")
        print("1. Organize your dataset using the template")
        print("2. Create labels in YOLO format")
        print("3. Update data.yaml configuration")
        print("4. Run training: python train_model.py --data_dir your_dataset")
        print("5. Evaluate and export your model")
        
    except Exception as e:
        print(f"Error in training demo: {e}")

def demo_configuration():
    """Demonstrate configuration management."""
    print("\nDemo 5: Configuration Management")
    print("=" * 50)
    
    try:
        from config import config
        
        print("Current configuration:")
        print(f"  - Confidence threshold: {config.detection.confidence_threshold}")
        print(f"  - Max detections: {config.detection.max_detections}")
        print(f"  - Video FPS: {config.video.fps}")
        print(f"  - Web interface port: {config.web.port}")
        
        # Demonstrate configuration modification
        print("\nModifying configuration...")
        original_threshold = config.detection.confidence_threshold
        config.detection.confidence_threshold = 0.7
        print(f"  - Updated confidence threshold: {config.detection.confidence_threshold}")
        
        # Save configuration
        config.save_config("demo_config.json")
        print("Configuration saved to 'demo_config.json'")
        
        # Restore original
        config.detection.confidence_threshold = original_threshold
        
    except Exception as e:
        print(f"Error in configuration demo: {e}")

def demo_performance():
    """Demonstrate performance features."""
    print("\n⚡ Demo 6: Performance Features")
    print("=" * 50)
    
    print("Performance optimization features:")
    print("  - GPU acceleration (CUDA support)")
    print("  - Configurable model sizes (YOLOv8n, YOLOv8s, YOLOv8m)")
    print("  - Batch processing capabilities")
    print("  - Memory management")
    print("  - Real-time processing optimization")
    
    print("\nPerformance tips:")
    print("  - Use GPU for faster inference")
    print("  - Choose appropriate model size for your needs")
    print("  - Adjust confidence threshold for speed/accuracy balance")
    print("  - Monitor memory usage during training")

def run_full_demo():
    """Run the complete demo."""
    print_banner()
    
    print("Starting Car Detection Project Demo")
    print("This demo will showcase all the features of the project.")
    print("\nPress Enter to continue...")
    input()
    
    # Run all demos
    demo_basic_detection()
    demo_analytics()
    demo_web_interface()
    demo_model_training()
    demo_configuration()
    demo_performance()
    
    print("\n" + "=" * 60)
    print("🎉 Demo completed successfully!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Start web interface: python web_interface.py")
    print("3. Train custom model: python train_model.py --create_template")
    print("4. Process your own images/videos")
    print("\nFor more information, see README.md")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Car Detection Project Demo')
    parser.add_argument('--basic', action='store_true', help='Run basic detection demo only')
    parser.add_argument('--analytics', action='store_true', help='Run analytics demo only')
    parser.add_argument('--web', action='store_true', help='Show web interface info only')
    parser.add_argument('--training', action='store_true', help='Show training info only')
    parser.add_argument('--config', action='store_true', help='Show configuration info only')
    parser.add_argument('--performance', action='store_true', help='Show performance info only')
    
    args = parser.parse_args()
    
    if args.basic:
        demo_basic_detection()
    elif args.analytics:
        demo_analytics()
    elif args.web:
        demo_web_interface()
    elif args.training:
        demo_model_training()
    elif args.config:
        demo_configuration()
    elif args.performance:
        demo_performance()
    else:
        run_full_demo()

if __name__ == "__main__":
    main() 