import json
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import numpy as np
from collections import defaultdict, Counter
import os

class DetectionAnalytics:
    """Analytics and reporting system for car detection project."""
    
    def __init__(self, output_dir: str = "analytics_output"):
        self.output_dir = output_dir
        self.detection_history = []
        self.performance_metrics = []
        self.session_data = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize counters
        self.total_detections = 0
        self.total_frames_processed = 0
        self.session_start_time = datetime.now()
    
    def record_detection(self, frame_data: Dict[str, Any]):
        """Record detection data for analysis."""
        timestamp = datetime.now()
        
        detection_record = {
            'timestamp': timestamp.isoformat(),
            'frame_number': frame_data.get('frame_number', 0),
            'detections': frame_data.get('detections', []),
            'processing_time': frame_data.get('processing_time', 0),
            'total_objects': len(frame_data.get('detections', [])),
            'vehicle_count': len([d for d in frame_data.get('detections', []) 
                                if d.get('class_name') in ['car', 'truck', 'bus', 'motorcycle']]),
            'person_count': len([d for d in frame_data.get('detections', []) 
                               if d.get('class_name') == 'person']),
            'traffic_signal_count': len([d for d in frame_data.get('detections', []) 
                                       if d.get('class_name') in ['traffic light', 'stop sign']])
        }
        
        self.detection_history.append(detection_record)
        self.total_detections += detection_record['total_objects']
        self.total_frames_processed += 1
    
    def record_performance(self, metrics: Dict[str, Any]):
        """Record performance metrics."""
        self.performance_metrics.append({
            'timestamp': datetime.now().isoformat(),
            **metrics
        })
    
    def generate_detection_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive detection summary."""
        if not self.detection_history:
            return {}
        
        # Extract all detections
        all_detections = []
        for record in self.detection_history:
            all_detections.extend(record['detections'])
        
        # Count by class
        class_counts = Counter([d.get('class_name', 'unknown') for d in all_detections])
        
        # Calculate confidence statistics
        confidences = [d.get('confidence', 0) for d in all_detections]
        
        # Calculate processing time statistics
        processing_times = [r.get('processing_time', 0) for r in self.detection_history]
        
        summary = {
            'total_frames_processed': self.total_frames_processed,
            'total_detections': self.total_detections,
            'average_detections_per_frame': self.total_detections / max(self.total_frames_processed, 1),
            'class_distribution': dict(class_counts),
            'confidence_stats': {
                'mean': np.mean(confidences) if confidences else 0,
                'std': np.std(confidences) if confidences else 0,
                'min': min(confidences) if confidences else 0,
                'max': max(confidences) if confidences else 0
            },
            'performance_stats': {
                'mean_processing_time': np.mean(processing_times) if processing_times else 0,
                'max_processing_time': max(processing_times) if processing_times else 0,
                'min_processing_time': min(processing_times) if processing_times else 0
            },
            'session_duration': (datetime.now() - self.session_start_time).total_seconds(),
            'detection_rate': self.total_detections / max((datetime.now() - self.session_start_time).total_seconds(), 1)
        }
        
        return summary
    
    def generate_visualizations(self):
        """Generate various visualization charts."""
        if not self.detection_history:
            print("No detection data available for visualization")
            return
        
        # Set style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Detection counts over time
        timestamps = [datetime.fromisoformat(r['timestamp']) for r in self.detection_history]
        detection_counts = [r['total_objects'] for r in self.detection_history]
        
        axes[0, 0].plot(timestamps, detection_counts, marker='o', alpha=0.7)
        axes[0, 0].set_title('Detections Over Time')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Number of Detections')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Class distribution pie chart
        summary = self.generate_detection_summary()
        class_dist = summary.get('class_distribution', {})
        
        if class_dist:
            labels = list(class_dist.keys())
            sizes = list(class_dist.values())
            axes[0, 1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            axes[0, 1].set_title('Object Class Distribution')
        
        # 3. Processing time histogram
        processing_times = [r.get('processing_time', 0) for r in self.detection_history]
        if processing_times:
            axes[1, 0].hist(processing_times, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Processing Time Distribution')
            axes[1, 0].set_xlabel('Processing Time (seconds)')
            axes[1, 0].set_ylabel('Frequency')
        
        # 4. Vehicle vs Person detection comparison
        vehicle_counts = [r['vehicle_count'] for r in self.detection_history]
        person_counts = [r['person_count'] for r in self.detection_history]
        
        x = range(len(self.detection_history))
        axes[1, 1].plot(x, vehicle_counts, label='Vehicles', marker='o', alpha=0.7)
        axes[1, 1].plot(x, person_counts, label='People', marker='s', alpha=0.7)
        axes[1, 1].set_title('Vehicle vs Person Detection')
        axes[1, 1].set_xlabel('Frame Number')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'detection_analytics.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def export_data(self, format: str = 'json'):
        """Export detection data in various formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == 'json':
            filename = os.path.join(self.output_dir, f'detection_data_{timestamp}.json')
            with open(filename, 'w') as f:
                json.dump({
                    'detection_history': self.detection_history,
                    'performance_metrics': self.performance_metrics,
                    'summary': self.generate_detection_summary()
                }, f, indent=2)
        
        elif format == 'csv':
            filename = os.path.join(self.output_dir, f'detection_data_{timestamp}.csv')
            with open(filename, 'w', newline='') as f:
                if self.detection_history:
                    writer = csv.DictWriter(f, fieldnames=self.detection_history[0].keys())
                    writer.writeheader()
                    writer.writerows(self.detection_history)
        
        elif format == 'excel':
            filename = os.path.join(self.output_dir, f'detection_data_{timestamp}.xlsx')
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Detection history
                if self.detection_history:
                    df_history = pd.DataFrame(self.detection_history)
                    df_history.to_excel(writer, sheet_name='Detection_History', index=False)
                
                # Performance metrics
                if self.performance_metrics:
                    df_performance = pd.DataFrame(self.performance_metrics)
                    df_performance.to_excel(writer, sheet_name='Performance_Metrics', index=False)
                
                # Summary
                summary = self.generate_detection_summary()
                df_summary = pd.DataFrame([summary])
                df_summary.to_excel(writer, sheet_name='Summary', index=False)
        
        print(f"Data exported to: {filename}")
        return filename
    
    def generate_report(self) -> str:
        """Generate a comprehensive HTML report."""
        summary = self.generate_detection_summary()
        
        # Generate visualizations
        self.generate_visualizations()
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Car Detection Analytics Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4f8; border-radius: 3px; }}
                .chart {{ text-align: center; margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Car Detection Analytics Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Session Overview</h2>
                <div class="metric">
                    <strong>Total Frames Processed:</strong> {summary.get('total_frames_processed', 0)}
                </div>
                <div class="metric">
                    <strong>Total Detections:</strong> {summary.get('total_detections', 0)}
                </div>
                <div class="metric">
                    <strong>Session Duration:</strong> {summary.get('session_duration', 0):.2f} seconds
                </div>
                <div class="metric">
                    <strong>Detection Rate:</strong> {summary.get('detection_rate', 0):.2f} detections/second
                </div>
            </div>
            
            <div class="section">
                <h2>Performance Statistics</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Mean Processing Time</td>
                        <td>{summary.get('performance_stats', {}).get('mean_processing_time', 0):.4f} seconds</td>
                    </tr>
                    <tr>
                        <td>Max Processing Time</td>
                        <td>{summary.get('performance_stats', {}).get('max_processing_time', 0):.4f} seconds</td>
                    </tr>
                    <tr>
                        <td>Average Detections per Frame</td>
                        <td>{summary.get('average_detections_per_frame', 0):.2f}</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Object Class Distribution</h2>
                <table>
                    <tr>
                        <th>Class</th>
                        <th>Count</th>
                    </tr>
        """
        
        for class_name, count in summary.get('class_distribution', {}).items():
            html_content += f"""
                    <tr>
                        <td>{class_name}</td>
                        <td>{count}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
                <div class="chart">
                    <img src="detection_analytics.png" alt="Detection Analytics" style="max-width: 100%;">
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        report_filename = os.path.join(self.output_dir, f'analytics_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html')
        with open(report_filename, 'w') as f:
            f.write(html_content)
        
        print(f"Analytics report generated: {report_filename}")
        return report_filename
    
    def reset_session(self):
        """Reset session data."""
        self.detection_history = []
        self.performance_metrics = []
        self.total_detections = 0
        self.total_frames_processed = 0
        self.session_start_time = datetime.now()
        print("Session data reset") 