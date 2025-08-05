#!/usr/bin/env python3
"""
System Test for Car Detection Project
=====================================

This script tests all components of the car detection project to ensure
everything is working correctly.
"""

import os
import sys
import time
import traceback
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    modules = [
        'cv2',
        'numpy',
        'torch',
        'ultralytics',
        'matplotlib',
        'PIL',
        'flask',
        'flask_socketio',
        'pandas',
        'yaml'
    ]
    
    failed_imports = []
    
    for module in modules:
        try:
            __import__(module)
            print(f"  {module}")
        except ImportError as e:
            print(f"  {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\nFailed to import: {failed_imports}")
        return False
    else:
        print("All imports successful!")
        return True

def test_object_detection():
    """Test the main object detection model."""
    print("\nTesting object detection model...")
    
    try:
        from object_detection_model import CarObjectDetector
        
        # Initialize detector
        detector = CarObjectDetector(confidence_threshold=0.5)
        print("  Detector initialized")
        
        # Test with dummy image
        import numpy as np
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Test detection
        detections = detector.detect_objects(dummy_image)
        print(f"  Detection completed (found {len(detections)} objects)")
        
        # Test scene analysis
        analysis = detector.analyze_scene(dummy_image)
        print("  Scene analysis completed")
        
        return True
        
    except Exception as e:
        print(f"  Object detection test failed: {e}")
        traceback.print_exc()
        return False

def test_analytics():
    """Test the analytics module."""
    print("\n Testing analytics module...")
    
    try:
        from analytics import DetectionAnalytics
        
        # Initialize analytics
        analytics = DetectionAnalytics()
        print("  Analytics initialized")
        
        # Test recording data
        frame_data = {
            'frame_number': 0,
            'detections': [
                {'class_name': 'car', 'confidence': 0.8},
                {'class_name': 'person', 'confidence': 0.7}
            ],
            'processing_time': 0.1
        }
        
        analytics.record_detection(frame_data)
        print("  Data recording completed")
        
        # Test summary generation
        summary = analytics.generate_detection_summary()
        print("  Summary generation completed")
        
        # Test data export
        analytics.export_data('json')
        print("  Data export completed")
        
        return True
        
    except Exception as e:
        print(f"  Analytics test failed: {e}")
        traceback.print_exc()
        return False

def test_configuration():
    """Test the configuration system."""
    print("\n  Testing configuration system...")
    
    try:
        from config import config
        
        # Test configuration access
        threshold = config.detection.confidence_threshold
        fps = config.video.fps
        port = config.web.port
        
        print(f"  Configuration loaded (threshold: {threshold}, fps: {fps}, port: {port})")
        
        # Test configuration modification
        original_threshold = config.detection.confidence_threshold
        config.detection.confidence_threshold = 0.7
        
        # Test configuration save/load
        config.save_config("test_config.json")
        print("  Configuration save completed")
        
        # Restore original
        config.detection.confidence_threshold = original_threshold
        
        return True
        
    except Exception as e:
        print(f"  Configuration test failed: {e}")
        traceback.print_exc()
        return False

def test_model_training():
    """Test the model training module."""
    print("\nTesting model training module...")
    
    try:
        from train_model import ModelTrainer
        
        # Test trainer initialization
        trainer = ModelTrainer("dummy", "test_output")
        print("  Trainer initialized")
        
        # Test dataset template creation
        trainer.create_dataset_template("test_dataset")
        print("  Dataset template created")
        
        # Check if template files exist
        template_files = [
            "test_dataset/data.yaml",
            "test_dataset/README.md"
        ]
        
        for file_path in template_files:
            if os.path.exists(file_path):
                print(f"  {file_path} exists")
            else:
                print(f"  {file_path} missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"  Model training test failed: {e}")
        traceback.print_exc()
        return False

def test_web_interface():
    """Test web interface components."""
    print("\nTesting web interface components...")
    
    try:
        # Test Flask import
        from flask import Flask
        print("  Flask imported")
        
        # Test SocketIO import
        from flask_socketio import SocketIO
        print("  SocketIO imported")
        
        # Test template directory creation
        os.makedirs("templates", exist_ok=True)
        if os.path.exists("templates"):
            print("  Templates directory exists")
        
        return True
        
    except Exception as e:
        print(f"  Web interface test failed: {e}")
        traceback.print_exc()
        return False

def test_performance():
    """Test performance-related features."""
    print("\n Testing performance features...")
    
    try:
        import torch
        
        # Test CUDA availability
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"  CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("  CUDA not available (will use CPU)")
        
        # Test OpenCV
        import cv2
        print(f"  OpenCV version: {cv2.__version__}")
        
        # Test numpy
        import numpy as np
        print(f"  NumPy version: {np.__version__}")
        
        return True
        
    except Exception as e:
        print(f"  Performance test failed: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all system tests."""
    print("Car Detection Project - System Test")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Object Detection", test_object_detection),
        ("Analytics", test_analytics),
        ("Configuration", test_configuration),
        ("Model Training", test_model_training),
        ("Web Interface", test_web_interface),
        ("Performance", test_performance)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"{test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 50)
    print("Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        return True
    else:
        print("Some tests failed. Please check the errors above.")
        return False

def cleanup():
    """Clean up test files."""
    test_files = [
        "test_config.json",
        "test_dataset",
        "analytics_output"
    ]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            if os.path.isdir(file_path):
                import shutil
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)

if __name__ == "__main__":
    try:
        success = run_all_tests()
        if success:
            print("\nSystem test completed successfully!")
            print("You can now use the car detection project.")
        else:
            print("\nSystem test failed!")
            print("Please fix the issues before using the project.")
    finally:
        cleanup() 