#  Car Detection Project

A comprehensive computer vision system for detecting cars and surrounding objects in real-world scenarios using YOLO (You Only Look Once) object detection.

##  Features

### Core Detection
- **Real-time object detection** using YOLOv8
- **Multi-class detection**: cars, trucks, buses, motorcycles, people, traffic signals
- **High accuracy** with configurable confidence thresholds
- **GPU acceleration** support for faster processing

### Web Interface
- **Modern web UI** with real-time video streaming
- **Live webcam processing** with instant detection results
- **Image upload** with drag-and-drop support
- **Real-time analytics** and performance metrics
- **Responsive design** for desktop and mobile

### Analytics & Reporting
- **Comprehensive analytics** with detailed statistics
- **Performance monitoring** with processing time tracking
- **Data export** in multiple formats (JSON, CSV, Excel)
- **HTML reports** with visualizations and charts
- **Session tracking** with historical data

### Model Training
- **Custom model training** on your own datasets
- **Dataset validation** and structure checking
- **Training configuration** management
- **Model evaluation** with detailed metrics
- **Model export** to various formats (ONNX, TorchScript)

### Configuration Management
- **Flexible configuration** system for all components
- **Environment-specific** settings
- **Easy customization** of detection parameters
- **Performance optimization** options

##  Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, for acceleration)
- Webcam (for real-time detection)

##  Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd car_detection_project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python object_detection_model.py
   ```

##  Quick Start

### 1. Basic Usage

```python
from object_detection_model import CarObjectDetector

# Initialize detector
detector = CarObjectDetector(confidence_threshold=0.5)

# Process an image
analysis = detector.process_image('path/to/image.jpg', 'output.jpg')
print(analysis['scene_summary'])
```

### 2. Web Interface

```bash
# Start the web interface
python web_interface.py
```

Then open your browser to `http://localhost:8080`

### 3. Real-time Webcam

```python
# Process webcam feed
detector.process_video(0)  # 0 for default webcam
```

##  Usage Guide

### Web Interface

The web interface provides an intuitive way to interact with the car detection system:

1. **Live Webcam**: Start real-time detection from your webcam
2. **Image Upload**: Upload and process individual images
3. **Analytics**: View detailed statistics and export data
4. **Reports**: Generate comprehensive HTML reports

### Command Line Interface

#### Process Images
```bash
python object_detection_model.py --image path/to/image.jpg --output output.jpg
```

#### Process Videos
```bash
python object_detection_model.py --video path/to/video.mp4 --output output.mp4
```

#### Train Custom Model
```bash
# Create dataset template
python train_model.py --create_template

# Train model
python train_model.py --data_dir path/to/dataset --epochs 100 --batch_size 16
```

### Programmatic Usage

```python
from object_detection_model import CarObjectDetector
from analytics import DetectionAnalytics
from config import config

# Initialize components
detector = CarObjectDetector()
analytics = DetectionAnalytics()

# Process image with analytics
image = cv2.imread('image.jpg')
detections = detector.detect_objects(image)
analysis = detector.analyze_scene(image)

# Record analytics
analytics.record_detection({
    'frame_number': 0,
    'detections': detections,
    'processing_time': 0.1
})

# Generate report
report_path = analytics.generate_report()
```

## Analytics Features

### Real-time Metrics
- **Detection counts** per frame
- **Processing time** tracking
- **Object class distribution**
- **Performance statistics**

### Data Export
- **JSON format**: Complete detection history
- **CSV format**: Tabular data for analysis
- **Excel format**: Multi-sheet reports
- **HTML reports**: Visual charts and summaries

### Visualization
- **Detection trends** over time
- **Class distribution** pie charts
- **Processing time** histograms
- **Vehicle vs person** comparisons

## Model Training

### Dataset Preparation

1. **Create dataset structure**:
   ```
   dataset/
   ├── train/
   │   ├── images/
   │   └── labels/
   ├── val/
   │   ├── images/
   │   └── labels/
   └── data.yaml
   ```

2. **Label format** (YOLO):
   ```
   <class_id> <x_center> <y_center> <width> <height>
   ```

3. **Configuration file** (`data.yaml`):
   ```yaml
   path: ../dataset
   train: train/images
   val: val/images
   nc: 5
   names: ['car', 'truck', 'bus', 'motorcycle', 'person']
   ```

### Training Process

```python
from train_model import ModelTrainer

# Initialize trainer
trainer = ModelTrainer('path/to/dataset', 'output_dir')

# Train model
model_path = trainer.train_model()

# Evaluate model
metrics = trainer.evaluate_model(model_path)

# Export model
export_path = trainer.export_model(model_path, format='onnx')
```

## Configuration

The project uses a flexible configuration system:

```python
from config import config

# Modify detection settings
config.detection.confidence_threshold = 0.6
config.detection.max_detections = 50

# Modify video settings
config.video.fps = 30
config.video.frame_width = 1280
config.video.frame_height = 720

# Save configuration
config.save_config('custom_config.json')
```

##  Project Structure

```
car_detection_project/
├── object_detection_model.py  # Main detection model
├── web_interface.py           # Flask web application
├── analytics.py               # Analytics and reporting
├── train_model.py             # Model training utilities
├── config.py                  # Configuration management
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── templates/                 # Web interface templates
├── analytics_output/          # Generated reports
├── trained_models/           # Trained model outputs
├── uploads/                  # Uploaded files
└── outputs/                  # Processed outputs
```

##  Customization

### Adding New Object Classes

1. **Update class names** in `object_detection_model.py`:
   ```python
   self.class_names = ['car', 'truck', 'bus', 'motorcycle', 'person', 'new_class']
   ```

2. **Add color mapping**:
   ```python
   colors = {
       'new_class': (255, 0, 0),  # Red
       # ... other colors
   }
   ```

### Performance Optimization

1. **GPU acceleration**:
   ```python
   detector = CarObjectDetector()
   # Automatically uses CUDA if available
   ```

2. **Batch processing**:
   ```python
   # Process multiple images efficiently
   for image in images:
       detections = detector.detect_objects(image)
   ```

3. **Model optimization**:
   ```python
   # Use smaller model for faster inference
   detector = CarObjectDetector(model_path='yolov8n.pt')
   ```

##  Troubleshooting

### Common Issues

1. **CUDA not available**:
   - Install CUDA toolkit
   - Verify GPU drivers
   - Use CPU fallback: `device='cpu'`

2. **Webcam not working**:
   - Check camera permissions
   - Verify camera index
   - Test with `cv2.VideoCapture(0)`

3. **Memory issues**:
   - Reduce batch size
   - Lower image resolution
   - Use smaller model

4. **Slow performance**:
   - Enable GPU acceleration
   - Reduce confidence threshold
   - Use smaller input size

### Performance Tips

- **GPU acceleration**: Use CUDA-compatible GPU
- **Model size**: Choose appropriate YOLO variant
- **Image resolution**: Balance accuracy vs speed
- **Batch processing**: Process multiple images together
- **Memory management**: Monitor GPU memory usage

##  Performance Benchmarks

| Model | Input Size | FPS (CPU) | FPS (GPU) | mAP50 |
|-------|------------|-----------|-----------|-------|
| YOLOv8n | 640x640 | 15 | 45 | 0.37 |
| YOLOv8s | 640x640 | 8 | 25 | 0.44 |
| YOLOv8m | 640x640 | 4 | 15 | 0.50 |

*Benchmarks on Intel i7-10700K CPU and RTX 3080 GPU*

##  Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Acknowledgments

- **Ultralytics** for YOLOv8 implementation
- **OpenCV** for computer vision utilities
- **Flask** for web framework
- **Matplotlib/Seaborn** for visualizations

##  Support

For questions, issues, or contributions:

1. Check the documentation
2. Search existing issues
3. Create a new issue with detailed information
4. Contact the maintainers

---

