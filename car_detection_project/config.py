import os
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class DetectionConfig:
    """Configuration for object detection settings."""
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    max_detections: int = 100
    model_path: str = None
    device: str = 'auto'  # 'auto', 'cpu', 'cuda'

@dataclass
class VideoConfig:
    """Configuration for video processing."""
    fps: int = 30
    frame_width: int = 640
    frame_height: int = 480
    output_format: str = 'mp4v'
    save_frames: bool = False
    max_video_length: int = 300  # seconds

@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""
    show_labels: bool = True
    show_confidence: bool = True
    show_tracking: bool = True
    colors: Dict[str, tuple] = None
    font_scale: float = 0.5
    line_thickness: int = 2

@dataclass
class AnalyticsConfig:
    """Configuration for analytics and reporting."""
    save_detections: bool = True
    generate_reports: bool = True
    export_format: str = 'json'  # 'json', 'csv', 'xml'
    tracking_enabled: bool = True
    performance_monitoring: bool = True

@dataclass
class WebConfig:
    """Configuration for web interface."""
    host: str = '0.0.0.0'
    port: int = 8080
    debug: bool = False
    enable_streaming: bool = True
    max_connections: int = 10

class ProjectConfig:
    """Main configuration class for the car detection project."""
    
    def __init__(self):
        self.detection = DetectionConfig()
        self.video = VideoConfig()
        self.visualization = VisualizationConfig()
        self.analytics = AnalyticsConfig()
        self.web = WebConfig()
        
        # Initialize default colors
        self.visualization.colors = {
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'detection': self.detection.__dict__,
            'video': self.video.__dict__,
            'visualization': self.visualization.__dict__,
            'analytics': self.analytics.__dict__,
            'web': self.web.__dict__
        }
    
    def from_dict(self, config_dict: Dict[str, Any]):
        """Load configuration from dictionary."""
        if 'detection' in config_dict:
            for key, value in config_dict['detection'].items():
                setattr(self.detection, key, value)
        
        if 'video' in config_dict:
            for key, value in config_dict['video'].items():
                setattr(self.video, key, value)
        
        if 'visualization' in config_dict:
            for key, value in config_dict['visualization'].items():
                setattr(self.visualization, key, value)
        
        if 'analytics' in config_dict:
            for key, value in config_dict['analytics'].items():
                setattr(self.analytics, key, value)
        
        if 'web' in config_dict:
            for key, value in config_dict['web'].items():
                setattr(self.web, key, value)
    
    def save_config(self, file_path: str):
        """Save configuration to file."""
        import json
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def load_config(self, file_path: str):
        """Load configuration from file."""
        import json
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        self.from_dict(config_dict)

# Global configuration instance
config = ProjectConfig() 