# smart_cleaning_bot/app.py
from flask import Flask, Response, render_template, request, jsonify, url_for
from picamera2 import Picamera2
import cv2
import time
import threading

# Import our custom modules
from detector import TrashDetector
from bot_controller import BotController

# --- Flask App Initialization ---
app = Flask(__name__) # 'templates' and 'static' folders are automatically found by Flask

# --- Configuration ---
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# --- Initialize Components ---
# Picamera2
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (FRAME_WIDTH, FRAME_HEIGHT)} # Capture RGB
)
picam2.configure(preview_config)
picam2.start()
print("PiCamera2 started.")

# Trash Detector
trash_detector = TrashDetector() # Uses defaults from detector.py

# Bot Controller (handles Arduino communication and state)
bot = BotController()

# --- Video Streaming Logic ---
def generate_frames():
    while True:
        # 1. Capture frame (RGB from Picamera2)
        frame_rgb = picam2.capture_array()

        # 2. Perform detection (on RGB frame)
        predictions = trash_detector.detect(frame_rgb.copy()) # Send a copy

        # 3. If in smart mode, update bot targeting based on TRASH predictions
        # Filter for only 'trash' predictions to send to bot controller
        trash_predictions_for_bot = [p for p in predictions if p['class_name'] == 'trash']
        if bot.bot_state == "SMART_CLEANING":
            bot.update_smart_targeting(trash_predictions_for_bot, FRAME_WIDTH)

        # 4. Prepare frame for display: Draw ALL detections
        # For display, we'll convert RGB to BGR as OpenCV drawing and imencode often expect BGR
        frame_bgr_display = cv2.cvtColor(frame_rgb.copy(), cv2.COLOR_RGB2BGR)
        trash_detector.draw_detections(frame_bgr_display, predictions) # Modifies frame_bgr_display

        # 5. Encode the BGR frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame_bgr_display)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html', initial_bot_state=bot.bot_state)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control', methods=['POST']) # Renamed to control_bot_route in HTML for clarity if needed
def control_bot_route(): # Matched to HTML's url_for
    data = request.get_json()
    action = data.get('action')
    status_message = "Unknown action"

    if action == 'OFF':
        status_message = bot.set_mode_off()
    elif action == 'SIMPLE_CLEAN':
        status_message = bot.set_mode_simple_clean()
    elif action == 'SMART_CLEAN':
        status_message = bot.set_mode_smart_clean()
    elif action == 'TOGGLE_VACUUM':
        status_message = bot.toggle_vacuum()
    else:
        return jsonify({'status': 'error', 'message': 'Invalid action'}), 400

    print(f"Flask Route Action: {action}, Bot State: {bot.bot_state}")
    return jsonify({'status': 'success', 'status_message': status_message, 'current_bot_state': bot.bot_state})

if __name__ == '__main__':
    # Important for picamera2 and Flask development server stability
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False, use_reloader=False)