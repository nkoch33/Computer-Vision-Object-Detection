from flask import Flask, render_template, request, jsonify, Response, send_file
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import os
import json
import threading
import time
from datetime import datetime
from typing import Dict, Any, Optional
import io
from PIL import Image

from object_detection_model import CarObjectDetector
from analytics import DetectionAnalytics
from config import config

app = Flask(__name__)
app.config['SECRET_KEY'] = 'car_detection_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
detector = None
analytics = None
is_processing = False
current_frame = None
processing_thread = None

def initialize_detector():
    """Initialize the car detector."""
    global detector, analytics
    detector = CarObjectDetector(
        confidence_threshold=config.detection.confidence_threshold
    )
    analytics = DetectionAnalytics()

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """Get current system status."""
    return jsonify({
        'is_processing': is_processing,
        'detector_initialized': detector is not None,
        'analytics_available': analytics is not None
    })

@app.route('/api/start_webcam')
def start_webcam():
    """Start webcam processing."""
    global is_processing, processing_thread
    
    if is_processing:
        return jsonify({'success': False, 'message': 'Already processing'})
    
    is_processing = True
    processing_thread = threading.Thread(target=webcam_processing_loop)
    processing_thread.daemon = True
    processing_thread.start()
    
    return jsonify({'success': True, 'message': 'Webcam processing started'})

@app.route('/api/stop_webcam')
def stop_webcam():
    """Stop webcam processing."""
    global is_processing
    is_processing = False
    return jsonify({'success': True, 'message': 'Webcam processing stopped'})

@app.route('/api/upload_image', methods=['POST'])
def upload_image():
    """Process uploaded image."""
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image provided'})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'})
    
    # Read and process image
    image_bytes = file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return jsonify({'success': False, 'message': 'Invalid image format'})
    
    # Detect objects
    start_time = time.time()
    detections = detector.detect_objects(image)
    processing_time = time.time() - start_time
    
    # Analyze scene
    analysis = detector.analyze_scene(image)
    
    # Draw detections
    annotated_image = detector.draw_detections(image, detections)
    
    # Convert to base64 for web display
    _, buffer = cv2.imencode('.jpg', annotated_image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Record analytics
    if analytics:
        analytics.record_detection({
            'frame_number': 0,
            'detections': detections,
            'processing_time': processing_time
        })
    
    return jsonify({
        'success': True,
        'image': img_base64,
        'analysis': analysis,
        'processing_time': processing_time
    })

@app.route('/api/process_video', methods=['POST'])
def process_video():
    """Process uploaded video."""
    if 'video' not in request.files:
        return jsonify({'success': False, 'message': 'No video provided'})
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'})
    
    # Save uploaded video
    video_path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(video_path)
    
    # Process video
    output_path = os.path.join('outputs', f'processed_{file.filename}')
    os.makedirs('outputs', exist_ok=True)
    
    try:
        detector.process_video(video_path, output_path, show_analysis=False)
        return jsonify({
            'success': True,
            'message': 'Video processed successfully',
            'output_path': output_path
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error processing video: {str(e)}'})

@app.route('/api/analytics')
def get_analytics():
    """Get analytics data."""
    if not analytics:
        return jsonify({'success': False, 'message': 'Analytics not available'})
    
    summary = analytics.generate_detection_summary()
    return jsonify({
        'success': True,
        'summary': summary
    })

@app.route('/api/export_data')
def export_data():
    """Export analytics data."""
    if not analytics:
        return jsonify({'success': False, 'message': 'Analytics not available'})
    
    format_type = request.args.get('format', 'json')
    try:
        filename = analytics.export_data(format_type)
        return send_file(filename, as_attachment=True)
    except Exception as e:
        return jsonify({'success': False, 'message': f'Export failed: {str(e)}'})

@app.route('/api/generate_report')
def generate_report():
    """Generate analytics report."""
    if not analytics:
        return jsonify({'success': False, 'message': 'Analytics not available'})
    
    try:
        report_path = analytics.generate_report()
        return send_file(report_path, as_attachment=True)
    except Exception as e:
        return jsonify({'success': False, 'message': f'Report generation failed: {str(e)}'})

def webcam_processing_loop():
    """Main webcam processing loop."""
    global current_frame, is_processing
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    frame_count = 0
    
    while is_processing:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Process frame
        start_time = time.time()
        detections = detector.detect_objects(frame)
        processing_time = time.time() - start_time
        
        # Draw detections
        annotated_frame = detector.draw_detections(frame, detections)
        
        # Record analytics
        if analytics:
            analytics.record_detection({
                'frame_number': frame_count,
                'detections': detections,
                'processing_time': processing_time
            })
        
        # Convert frame to base64
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Send frame to web client
        socketio.emit('frame_update', {
            'image': img_base64,
            'detections': len(detections),
            'processing_time': processing_time,
            'frame_count': frame_count
        })
        
        frame_count += 1
        time.sleep(1/config.video.fps)  # Control frame rate
    
    cap.release()

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print('Client connected')
    emit('status', {'message': 'Connected to car detection system'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print('Client disconnected')

if __name__ == '__main__':
    # Initialize detector
    initialize_detector()
    
    # Create templates directory and HTML template
    os.makedirs('templates', exist_ok=True)
    
    # Create the HTML template
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Car Detection System</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 10px;
            }
            .section {
                margin: 20px 0;
                padding: 20px;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            .button {
                background: #007bff;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                margin: 5px;
            }
            .button:hover {
                background: #0056b3;
            }
            .button:disabled {
                background: #ccc;
                cursor: not-allowed;
            }
            .video-container {
                text-align: center;
                margin: 20px 0;
            }
            #videoFeed {
                max-width: 100%;
                border: 2px solid #ddd;
                border-radius: 5px;
            }
            .stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }
            .stat-card {
                background: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                text-align: center;
            }
            .upload-area {
                border: 2px dashed #ddd;
                padding: 40px;
                text-align: center;
                border-radius: 5px;
                margin: 20px 0;
            }
            .upload-area.dragover {
                border-color: #007bff;
                background-color: #f0f8ff;
            }
        </style>
    </head>
    <body>
        <div class="container">
                                    <div class="header">
                            <h1>Car Detection System</h1>
                            <p>Real-time object detection and analytics</p>
                        </div>
            
            <div class="section">
                <h2>📹 Live Webcam</h2>
                <div class="video-container">
                    <img id="videoFeed" src="" alt="Video feed" style="display: none;">
                    <div id="noVideo">Click "Start Webcam" to begin live detection</div>
                </div>
                <div style="text-align: center;">
                    <button class="button" onclick="startWebcam()">Start Webcam</button>
                    <button class="button" onclick="stopWebcam()">Stop Webcam</button>
                </div>
                <div class="stats">
                    <div class="stat-card">
                        <h3>Detections</h3>
                        <p id="detectionCount">0</p>
                    </div>
                    <div class="stat-card">
                        <h3>Processing Time</h3>
                        <p id="processingTime">0ms</p>
                    </div>
                    <div class="stat-card">
                        <h3>Frame Count</h3>
                        <p id="frameCount">0</p>
                    </div>
                </div>
            </div>
            
            <div class="section">
                                        <h2>Upload Image</h2>
                <div class="upload-area" id="imageUpload">
                    <p>Drag and drop an image here or click to select</p>
                    <input type="file" id="imageInput" accept="image/*" style="display: none;">
                    <button class="button" onclick="document.getElementById('imageInput').click()">Select Image</button>
                </div>
                <div id="imageResult" style="display: none;">
                    <img id="processedImage" style="max-width: 100%; border-radius: 5px;">
                    <div id="imageAnalysis"></div>
                </div>
            </div>
            
            <div class="section">
                                        <h2>Analytics</h2>
                <button class="button" onclick="getAnalytics()">Refresh Analytics</button>
                <button class="button" onclick="exportData()">Export Data</button>
                <button class="button" onclick="generateReport()">Generate Report</button>
                <div id="analyticsData"></div>
            </div>
        </div>
        
        <script>
            const socket = io();
            
            // Webcam controls
            function startWebcam() {
                fetch('/api/start_webcam')
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            document.getElementById('noVideo').style.display = 'none';
                            document.getElementById('videoFeed').style.display = 'block';
                        }
                    });
            }
            
            function stopWebcam() {
                fetch('/api/stop_webcam')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('videoFeed').style.display = 'none';
                        document.getElementById('noVideo').style.display = 'block';
                    });
            }
            
            // Socket events
            socket.on('frame_update', function(data) {
                document.getElementById('videoFeed').src = 'data:image/jpeg;base64,' + data.image;
                document.getElementById('detectionCount').textContent = data.detections;
                document.getElementById('processingTime').textContent = (data.processing_time * 1000).toFixed(1) + 'ms';
                document.getElementById('frameCount').textContent = data.frame_count;
            });
            
            // Image upload
            document.getElementById('imageInput').addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (file) {
                    uploadImage(file);
                }
            });
            
            function uploadImage(file) {
                const formData = new FormData();
                formData.append('image', file);
                
                fetch('/api/upload_image', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById('processedImage').src = 'data:image/jpeg;base64,' + data.image;
                        document.getElementById('imageResult').style.display = 'block';
                        
                        const analysis = data.analysis;
                        document.getElementById('imageAnalysis').innerHTML = `
                            <h3>Analysis Results:</h3>
                            <p><strong>Total Objects:</strong> ${analysis.total_objects}</p>
                            <p><strong>Vehicles:</strong> ${analysis.vehicles.count}</p>
                            <p><strong>People:</strong> ${analysis.people.count}</p>
                            <p><strong>Processing Time:</strong> ${(data.processing_time * 1000).toFixed(1)}ms</p>
                            <p><strong>Summary:</strong> ${analysis.scene_summary}</p>
                        `;
                    }
                });
            }
            
            // Analytics
            function getAnalytics() {
                fetch('/api/analytics')
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            const summary = data.summary;
                            document.getElementById('analyticsData').innerHTML = `
                                <h3>Session Summary:</h3>
                                <p><strong>Total Frames:</strong> ${summary.total_frames_processed}</p>
                                <p><strong>Total Detections:</strong> ${summary.total_detections}</p>
                                <p><strong>Session Duration:</strong> ${summary.session_duration.toFixed(1)}s</p>
                                <p><strong>Detection Rate:</strong> ${summary.detection_rate.toFixed(2)} detections/s</p>
                            `;
                        }
                    });
            }
            
            function exportData() {
                window.open('/api/export_data?format=json', '_blank');
            }
            
            function generateReport() {
                window.open('/api/generate_report', '_blank');
            }
            
            // Drag and drop for image upload
            const uploadArea = document.getElementById('imageUpload');
            
            uploadArea.addEventListener('dragover', function(e) {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });
            
            uploadArea.addEventListener('dragleave', function(e) {
                uploadArea.classList.remove('dragover');
            });
            
            uploadArea.addEventListener('drop', function(e) {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    uploadImage(files[0]);
                }
            });
        </script>
    </body>
    </html>
    """
    
    with open('templates/index.html', 'w') as f:
        f.write(html_template)
    
    print(f"Starting web interface on http://{config.web.host}:{config.web.port}")
    socketio.run(app, host=config.web.host, port=config.web.port, debug=config.web.debug) 