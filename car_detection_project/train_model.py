import os
import yaml
import json
from typing import Dict, List, Any, Optional
import shutil
from datetime import datetime
import argparse

class ModelTrainer:
    """Trainer for custom car detection models."""
    
    def __init__(self, data_dir: str, output_dir: str = "trained_models"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.model_config = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Default training configuration
        self.default_config = {
            'model': 'yolov8n.pt',  # Base model
            'data': None,  # Will be set to data.yaml path
            'epochs': 100,
            'batch_size': 16,
            'imgsz': 640,
            'patience': 50,
            'save': True,
            'save_period': 10,
            'cache': False,
            'device': 'auto',
            'workers': 8,
            'project': output_dir,
            'name': 'car_detection_model',
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'auto',
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'pose': 12.0,
            'kobj': 1.0,
            'label_smoothing': 0.0,
            'nbs': 64,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'plots': True
        }
    
    def validate_dataset(self) -> bool:
        """Validate the dataset structure."""
        required_dirs = ['train', 'val', 'test']
        required_files = ['data.yaml']
        
        # Check if data.yaml exists
        data_yaml_path = os.path.join(self.data_dir, 'data.yaml')
        if not os.path.exists(data_yaml_path):
            print(f"Error: data.yaml not found in {self.data_dir}")
            return False
        
        # Check directories
        for dir_name in required_dirs:
            dir_path = os.path.join(self.data_dir, dir_name)
            if not os.path.exists(dir_path):
                print(f"Warning: {dir_name} directory not found in {self.data_dir}")
        
        # Load and validate data.yaml
        try:
            with open(data_yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)
            
            # Check required keys
            required_keys = ['train', 'val', 'nc', 'names']
            for key in required_keys:
                if key not in data_config:
                    print(f"Error: Missing '{key}' in data.yaml")
                    return False
            
            print(f"Dataset validation successful:")
            print(f"  - Classes: {data_config['nc']}")
            print(f"  - Class names: {data_config['names']}")
            print(f"  - Train path: {data_config['train']}")
            print(f"  - Val path: {data_config['val']}")
            
            return True
            
        except Exception as e:
            print(f"Error reading data.yaml: {e}")
            return False
    
    def create_training_config(self, custom_config: Dict[str, Any] = None) -> str:
        """Create training configuration file."""
        config = self.default_config.copy()
        
        # Set data path
        config['data'] = os.path.join(self.data_dir, 'data.yaml')
        
        # Update with custom config
        if custom_config:
            config.update(custom_config)
        
        # Create config file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_path = os.path.join(self.output_dir, f'training_config_{timestamp}.yaml')
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Training configuration saved to: {config_path}")
        return config_path
    
    def train_model(self, config_path: str = None, custom_config: Dict[str, Any] = None) -> str:
        """Train the model using YOLO."""
        from ultralytics import YOLO
        
        # Validate dataset first
        if not self.validate_dataset():
            raise ValueError("Dataset validation failed")
        
        # Create config if not provided
        if not config_path:
            config_path = self.create_training_config(custom_config)
        
        # Load base model
        model = YOLO('yolov8n.pt')
        
        # Start training
        print("Starting model training...")
        print(f"Configuration: {config_path}")
        
        try:
            # Train the model
            results = model.train(
                data=os.path.join(self.data_dir, 'data.yaml'),
                epochs=self.default_config['epochs'],
                imgsz=self.default_config['imgsz'],
                batch=self.default_config['batch_size'],
                name=f"car_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                project=self.output_dir,
                exist_ok=True,
                pretrained=True,
                optimizer='auto',
                verbose=True,
                save=True,
                save_period=10,
                plots=True
            )
            
            # Get best model path
            best_model_path = results.save_dir / 'weights' / 'best.pt'
            if best_model_path.exists():
                print(f"Training completed successfully!")
                print(f"Best model saved to: {best_model_path}")
                return str(best_model_path)
            else:
                print("Training completed but best model not found")
                return None
                
        except Exception as e:
            print(f"Training failed: {e}")
            raise
    
    def evaluate_model(self, model_path: str) -> Dict[str, Any]:
        """Evaluate the trained model."""
        from ultralytics import YOLO
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load model
        model = YOLO(model_path)
        
        # Evaluate on validation set
        data_yaml_path = os.path.join(self.data_dir, 'data.yaml')
        results = model.val(data=data_yaml_path)
        
        # Extract metrics
        metrics = {
            'mAP50': float(results.box.map50),
            'mAP50-95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
            'f1': float(results.box.map50 * 2 / (results.box.map50 + 1))
        }
        
        print("Model Evaluation Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def export_model(self, model_path: str, format: str = 'onnx') -> str:
        """Export model to different formats."""
        from ultralytics import YOLO
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = YOLO(model_path)
        
        # Export model
        export_path = model.export(format=format)
        
        print(f"Model exported to: {export_path}")
        return export_path
    
    def create_dataset_template(self, output_dir: str = "dataset_template"):
        """Create a template for dataset structure."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create directory structure
        dirs = ['train/images', 'train/labels', 'val/images', 'val/labels', 'test/images', 'test/labels']
        for dir_path in dirs:
            os.makedirs(os.path.join(output_dir, dir_path), exist_ok=True)
        
        # Create data.yaml template
        data_yaml_content = """# Dataset configuration
path: ../dataset  # Dataset root directory
train: train/images  # Train images (relative to 'path')
val: val/images  # Val images (relative to 'path')
test: test/images  # Test images (optional)

# Classes
nc: 5  # Number of classes
names: ['car', 'truck', 'bus', 'motorcycle', 'person']  # Class names
"""
        
        with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
            f.write(data_yaml_content)
        
        # Create README
        readme_content = """# Car Detection Dataset Template

This is a template for organizing your car detection dataset.

## Directory Structure
```
dataset/
├── train/
│   ├── images/     # Training images
│   └── labels/     # Training labels (YOLO format)
├── val/
│   ├── images/     # Validation images
│   └── labels/     # Validation labels (YOLO format)
├── test/
│   ├── images/     # Test images (optional)
│   └── labels/     # Test labels (optional)
└── data.yaml       # Dataset configuration
```

## Label Format
Labels should be in YOLO format:
- One line per object
- Format: <class_id> <x_center> <y_center> <width> <height>
- All values are normalized (0-1)

## Image Format
- Supported formats: JPG, PNG, BMP, TIFF
- Recommended size: 640x640 or larger
- Aspect ratio: Any

## Class IDs
0: car
1: truck
2: bus
3: motorcycle
4: person

## Usage
1. Place your images in the appropriate directories
2. Create corresponding label files
3. Update data.yaml with correct paths and class names
4. Use with the ModelTrainer class
"""
        
        with open(os.path.join(output_dir, 'README.md'), 'w') as f:
            f.write(readme_content)
        
        print(f"Dataset template created in: {output_dir}")
        print("Please organize your data according to the template structure.")

def main():
    """Main function for model training."""
    parser = argparse.ArgumentParser(description='Train custom car detection model')
    parser.add_argument('--data_dir', required=True, help='Path to dataset directory')
    parser.add_argument('--output_dir', default='trained_models', help='Output directory for trained models')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--create_template', action='store_true', help='Create dataset template')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model after training')
    parser.add_argument('--export', action='store_true', help='Export model after training')
    
    args = parser.parse_args()
    
    if args.create_template:
        trainer = ModelTrainer("dummy")
        trainer.create_dataset_template()
        return
    
    # Initialize trainer
    trainer = ModelTrainer(args.data_dir, args.output_dir)
    
    # Custom training configuration
    custom_config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'imgsz': args.imgsz
    }
    
    try:
        # Train model
        model_path = trainer.train_model(custom_config=custom_config)
        
        if model_path and args.evaluate:
            # Evaluate model
            metrics = trainer.evaluate_model(model_path)
            
            # Save evaluation results
            results_path = os.path.join(args.output_dir, 'evaluation_results.json')
            with open(results_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"Evaluation results saved to: {results_path}")
        
        if model_path and args.export:
            # Export model
            export_path = trainer.export_model(model_path, format='onnx')
            print(f"Model exported to: {export_path}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 