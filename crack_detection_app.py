import cv2
import numpy as np
import time
import os
import json
import platform
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, send_from_directory, request
from flask_sqlalchemy import SQLAlchemy
import threading
import queue
import base64
from io import BytesIO
from PIL import Image
import serial
import serial.tools.list_ports

# === CONFIGURATION ===
# Cross-platform path detection
if platform.system() == "Windows":
    base_path = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(base_path, "yolov4-tiny-custom_final.weights")
    config_path = os.path.join(base_path, "yolov4-tiny-custom.cfg")
    names_path = os.path.join(base_path, "obj.names")
    output_dir = os.path.join(base_path, "crack_screenshots")
    db_path = os.path.join(base_path, "crack_detection.db")
else:
    base_path = "/home/harshitji/vscode/python/main-pr"
    weights_path = os.path.join(base_path, "yolov4-tiny-custom_final.weights")
    config_path = os.path.join(base_path, "yolov4-tiny-custom.cfg")
    names_path = os.path.join(base_path, "obj.names")
    output_dir = os.path.join(base_path, "crack_screenshots")
    db_path = os.path.join(base_path, "crack_detection.db")

# Create directories
os.makedirs(output_dir, exist_ok=True)

# Detection parameters
conf_threshold = 0.2
nms_threshold = 0.4

# Global variables
current_frame = None
frame_lock = threading.Lock()
detection_lock = threading.Lock()
camera_error = None
model_error = None

# Arduino serial variables
arduino_serial = None
latest_arduino_data = None
arduino_lock = threading.Lock()
arduino_error = None
arduino_thread = None
arduino_running = False

# Flask app
app = Flask(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Database Model
class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.now)  # Use local time
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(500), nullable=False)
    confidence = db.Column(db.Float, nullable=True)
    total_detections = db.Column(db.Integer, nullable=False, default=0)
    bbox_data = db.Column(db.Text, nullable=True)  # JSON string for bounding box data
    class_name = db.Column(db.String(100), nullable=True)
    distance = db.Column(db.Float, nullable=True, default=0.0)  # Distance in meters

    def __repr__(self):
        return f'<Detection {self.id} at {self.timestamp}>'

    def to_dict(self):
        """Convert detection to dictionary for JSON response"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'filename': self.filename,
            'filepath': self.filepath,
            'confidence': self.confidence,
            'total_detections': self.total_detections,
            'bbox_data': json.loads(self.bbox_data) if self.bbox_data else [],
            'class_name': self.class_name,
            'distance': self.distance if hasattr(self, 'distance') else 100.0
        }

class CrackDetector:
    def __init__(self):
        self.net = None
        self.classes = []
        self.output_layers = []
        self.cap = None
        self.is_running = False
        
    def load_model(self):
        """Load YOLO model with error handling"""
        global model_error
        try:
            # Check if files exist
            if not os.path.exists(weights_path):
                model_error = f"Weights file not found: {weights_path}"
                return False
            if not os.path.exists(config_path):
                model_error = f"Config file not found: {config_path}"
                return False
            if not os.path.exists(names_path):
                model_error = f"Names file not found: {names_path}"
                return False
                
            # Load model
            self.net = cv2.dnn.readNet(weights_path, config_path)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            # Load class labels
            with open(names_path, "r") as f:
                self.classes = f.read().strip().split("\n")
            
            # Get output layer names
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]
            
            model_error = None
            return True
            
        except Exception as e:
            model_error = f"Error loading model: {str(e)}"
            return False
    
    def initialize_camera(self):
        """Initialize camera with error handling"""
        global camera_error
        try:
            # Try different camera indices for external webcam
            for camera_index in [0, 1, 2]:
                print(f"Trying camera index {camera_index}...")
                self.cap = cv2.VideoCapture(camera_index)
                if self.cap.isOpened():
                    # Test if we can actually read a frame
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None:
                        print(f"Camera initialized successfully with index {camera_index}")
                        camera_error = None
                        return True
                    else:
                        print(f"Camera index {camera_index} opened but cannot read frames")
                        self.cap.release()
                else:
                    print(f"Camera index {camera_index} failed to open")
                    self.cap.release()
            
            camera_error = "No camera found. Tried indices 0, 1, 2"
            return False
            
        except Exception as e:
            camera_error = f"Camera initialization error: {str(e)}"
            return False
    
    def preprocess_frame(self, frame):
        """Preprocess frame for better detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(gray_eq, cv2.MORPH_OPEN, kernel)
        blurred = cv2.GaussianBlur(opened, (3, 3), 0)
        preprocessed = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
        return preprocessed
    
    def detect_cracks(self, frame):
        """Detect cracks in the frame"""
        if self.net is None:
            return frame, []
        
        height, width = frame.shape[:2]
        processed_frame = self.preprocess_frame(frame)
        
        blob = cv2.dnn.blobFromImage(processed_frame, 1/255.0, (640, 640), swapRB=True, crop=False)
        self.net.setInput(blob)
        detections = self.net.forward(self.output_layers)
        
        boxes, confidences, class_ids = [], [], []
        
        for output in detections:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        
        detections_list = []
        if len(indices) > 0:
            for i in indices:
                i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
                x, y, w, h = boxes[i]
                label = f"{self.classes[class_id]}: {confidences[i]:.2f}"
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                detections_list.append({
                    'bbox': [x, y, w, h],
                    'confidence': confidences[i],
                    'class': self.classes[class_ids[i]]
                })
        
        # Always return the frame with detections (if any)
        # Add status text to show detection is active
        cv2.putText(frame, "Detection Active", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame, detections_list
    
    def save_screenshot(self, frame, detections):
        """Save screenshot with database record"""
        # Use consistent timestamp (local time for both filename and database)
        current_time = datetime.now()
        timestamp_str = current_time.strftime("%Y%m%d_%H%M%S")
        filename = f"crack_{timestamp_str}.jpg"
        filepath = os.path.join(output_dir, filename)
        
        # Save image
        cv2.imwrite(filepath, frame)
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        # Get detection data
        confidence = detections[0]['confidence'] if detections else None
        class_name = detections[0]['class'] if detections else None
        
        # Convert detections for JSON serialization
        serializable_detections = convert_numpy_types(detections) if detections else None
        bbox_data = json.dumps(serializable_detections) if serializable_detections else None
        
        # Get real-time distance from Arduino
        arduino_distance = get_arduino_distance()
        
        # Create database record with explicit timestamp
        with app.app_context():
            detection_record = Detection(
                timestamp=current_time,  # Use the same timestamp as filename
                filename=filename,
                filepath=filepath,
                confidence=confidence,
                total_detections=len(detections),
                bbox_data=bbox_data,
                class_name=class_name,
                distance=arduino_distance  # Real-time Arduino distance
            )
            
            db.session.add(detection_record)
            db.session.commit()
        
        return detection_record
    
    def run_detection(self):
        """Main detection loop"""
        self.is_running = True
        
        while self.is_running:
            if self.cap is None or not self.cap.isOpened():
                print("Camera not available, stopping detection loop")
                break
                
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame from camera")
                break
            
            # Detect cracks
            processed_frame, detections = self.detect_cracks(frame)
            
            # Save screenshot if detections found
            if detections:
                self.save_screenshot(processed_frame, detections)
            
            # Always update current frame (show live feed continuously)
            with frame_lock:
                global current_frame
                current_frame = processed_frame.copy()
            
            # Small delay
            cv2.waitKey(1)
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

# Arduino serial functions
def find_arduino_port():
    """Find Arduino port automatically"""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if 'Arduino' in port.description or 'CH340' in port.description or 'USB Serial' in port.description:
            return port.device
    return None

def initialize_arduino():
    """Initialize Arduino serial connection"""
    global arduino_serial, arduino_error
    
    try:
        # Try to find Arduino port automatically
        port = find_arduino_port()
        if not port:
            # Fallback to common ports
            if platform.system() == "Windows":
                common_ports = ['COM3', 'COM4', 'COM5', 'COM6']
            else:
                common_ports = ['/dev/ttyACM0', '/dev/ttyUSB0', '/dev/ttyUSB1']
            
            for p in common_ports:
                try:
                    test_serial = serial.Serial(p, 9600, timeout=1)
                    test_serial.close()
                    port = p
                    break
                except:
                    continue
        
        if port:
            arduino_serial = serial.Serial(port, 9600, timeout=1)
            arduino_error = None
            print(f"Arduino connected on {port}")
            return True
        else:
            arduino_error = "No Arduino found on common ports"
            return False
            
    except Exception as e:
        arduino_error = f"Arduino connection error: {str(e)}"
        return False

def arduino_reader():
    """Background thread to read Arduino data"""
    global latest_arduino_data, arduino_running, arduino_error
    
    while arduino_running:
        try:
            if arduino_serial and arduino_serial.is_open:
                if arduino_serial.in_waiting > 0:
                    line = arduino_serial.readline().decode('utf-8').strip()
                    if line:
                        with arduino_lock:
                            latest_arduino_data = line
                        print(f"Arduino data: {line}")
            time.sleep(0.1)  # Small delay to prevent blocking
        except Exception as e:
            arduino_error = f"Arduino read error: {str(e)}"
            time.sleep(1)  # Wait before retrying

def get_arduino_distance():
    """Get distance value from Arduino data"""
    global latest_arduino_data
    
    try:
        with arduino_lock:
            if latest_arduino_data:
                # Try to parse the Arduino data as a float (distance in meters)
                distance = float(latest_arduino_data)
                return distance
            else:
                return 100.0  # Default fallback
    except (ValueError, TypeError):
        # If Arduino data is not a valid number, return default
        return 100.0

def start_arduino_reader():
    """Start Arduino reading thread"""
    global arduino_thread, arduino_running
    
    if arduino_serial and arduino_serial.is_open:
        arduino_running = True
        arduino_thread = threading.Thread(target=arduino_reader, daemon=True)
        arduino_thread.start()
        print("Arduino reader thread started")

def stop_arduino_reader():
    """Stop Arduino reading thread"""
    global arduino_running
    
    arduino_running = False
    if arduino_thread:
        arduino_thread.join(timeout=1)

# Global detector instance
detector = CrackDetector()

def generate_frames():
    """Generate frames for video stream"""
    while True:
        with frame_lock:
            if current_frame is not None:
                # Convert frame to JPEG
                ret, buffer = cv2.imencode('.jpg', current_frame)
                if ret:
                    frame_data = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
            else:
                # Create a placeholder frame when no camera feed is available
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Camera Initializing...", (150, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', placeholder)
                if ret:
                    frame_data = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        time.sleep(0.1)

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/static/screenshots/<filename>')
def serve_screenshot(filename):
    """Serve screenshot files"""
    return send_from_directory(output_dir, filename)

@app.route('/api/detections')
def get_detections():
    """API endpoint for detection history"""
    try:
        # Get last 10 detections from database
        recent_detections = Detection.query.order_by(Detection.timestamp.desc()).limit(10).all()
        total_count = Detection.query.count()
        
        return jsonify({
            'detections': [detection.to_dict() for detection in recent_detections],
            'total': total_count,
            'page': 1,
            'per_page': 10
        })
    except Exception as e:
        print(f"Error fetching detections: {e}")
        return jsonify({
            'detections': [],
            'total': 0,
            'page': 1,
            'per_page': 10
        })

@app.route('/api/errors')
def get_errors():
    """API endpoint for error status"""
    return jsonify({
        'camera_error': camera_error,
        'model_error': model_error
    })

@app.route('/api/stats')
def get_stats():
    """API endpoint for system statistics"""
    try:
        # Get total detections
        total_detections = Detection.query.count()
        
        # Get today's detections
        today = datetime.now().date()
        today_detections = Detection.query.filter(
            Detection.timestamp >= today
        ).count()
        
        # Calculate detection rate (percentage of detections with cracks found)
        if total_detections > 0:
            successful_detections = Detection.query.filter(
                Detection.total_detections > 0
            ).count()
            detection_rate = round((successful_detections / total_detections) * 100, 1)
        else:
            detection_rate = 0.0
        
        return jsonify({
            'total_cracks': total_detections,
            'today_cracks': today_detections,
            'detection_rate': detection_rate
        })
        
    except Exception as e:
        print(f"Error fetching stats: {e}")
        # Return default values if database error occurs
        return jsonify({
            'total_cracks': 0,
            'today_cracks': 0,
            'detection_rate': 0.0
        })

@app.route('/arduino-data')
def get_arduino_data():
    """API endpoint for Arduino serial data"""
    try:
        with arduino_lock:
            data = latest_arduino_data if latest_arduino_data else "No data"
            error = arduino_error if arduino_error else None
            is_connected = arduino_serial and arduino_serial.is_open if arduino_serial else False
        
        return jsonify({
            'data': data,
            'error': error,
            'connected': is_connected,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'data': "Error reading data",
            'error': str(e),
            'connected': False,
            'timestamp': datetime.now().isoformat()
        })

@app.route('/api/arduino-distance')
def get_arduino_distance_api():
    """API endpoint for current Arduino distance"""
    try:
        distance = get_arduino_distance()
        with arduino_lock:
            is_connected = arduino_serial and arduino_serial.is_open if arduino_serial else False
            error = arduino_error if arduino_error else None
        
        return jsonify({
            'distance': distance,
            'connected': is_connected,
            'error': error,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'distance': 100.0,
            'connected': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        })

@app.route('/api/delete-range', methods=['POST'])
def delete_range():
    """API endpoint for deleting detections within an ID range"""
    try:
        data = request.get_json()
        start_id = data.get('start_id')
        end_id = data.get('end_id')
        
        if not start_id or not end_id:
            return jsonify({'success': False, 'error': 'Start and end IDs are required'})
        
        # Validate ID range
        if start_id > end_id:
            return jsonify({'success': False, 'error': 'Start ID cannot be greater than end ID'})
        
        if start_id < 1:
            return jsonify({'success': False, 'error': 'Start ID must be at least 1'})
        
        # Query detections within the ID range
        detections_to_delete = Detection.query.filter(
            Detection.id >= start_id,
            Detection.id <= end_id
        ).all()
        
        deleted_count = len(detections_to_delete)
        
        if deleted_count == 0:
            return jsonify({'success': False, 'error': f'No detections found with IDs between {start_id} and {end_id}'})
        
        # Delete the files associated with these detections
        for detection in detections_to_delete:
            try:
                file_path = os.path.join(output_dir, detection.filename)
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {detection.filename}: {e}")
        
        # Delete from database
        for detection in detections_to_delete:
            db.session.delete(detection)
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'deleted_count': deleted_count,
            'message': f'Successfully deleted {deleted_count} detections (IDs {start_id}-{end_id})'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})

def start_detection():
    """Start the detection process"""
    # Create database tables
    with app.app_context():
        # Drop all tables and recreate them to include the new distance column
        db.drop_all()
        db.create_all()
        print("Database tables recreated with distance column")
    
    # Initialize Arduino
    if initialize_arduino():
        start_arduino_reader()
        print("Arduino initialized and reader started")
    else:
        print(f"Arduino initialization failed: {arduino_error}")
    
    # Load model
    if not detector.load_model():
        print(f"Model loading failed: {model_error}")
        return
    
    # Initialize camera
    if not detector.initialize_camera():
        print(f"Camera initialization failed: {camera_error}")
        # Create a test pattern if no camera is available
        with frame_lock:
            global current_frame
            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(test_frame, "Camera Not Available", (150, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(test_frame, "Please check camera connection", (100, 250), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            current_frame = test_frame
        return
    
    # Start detection loop
    detector.run_detection()

if __name__ == '__main__':
    # Start detection in separate thread
    detection_thread = threading.Thread(target=start_detection, daemon=True)
    detection_thread.start()
    
    try:
        # Start Flask app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("Shutting down...")
        stop_arduino_reader()
        if arduino_serial:
            arduino_serial.close() 