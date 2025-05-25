# smart_cleaning_bot/detector.py
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

# --- Configuration ---
MODEL_PATH = "/home/pi/Downloads/TF Models/best-fp16.tflite"  # Or pass as argument
CLASS_NAMES = ['dirt', 'liquid', 'marks', 'trash']  # Or pass as argument
TFLITE_INPUT_SIZE = (640, 640)  # Model specific
CONFIDENCE_THRESHOLD = 0.4  # Detection confidence


class TrashDetector:
    def __init__(self, model_path=MODEL_PATH, class_names=CLASS_NAMES, input_size=TFLITE_INPUT_SIZE,
                 confidence_threshold=CONFIDENCE_THRESHOLD):
        # ... (init from previous step, including trash_confidence_threshold) ...
        self.model_path = model_path
        self.class_names = class_names
        self.input_size = input_size
        self.default_confidence_threshold = confidence_threshold
        self.trash_confidence_threshold = 0.60

        self.interpreter = Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Define BGR colors for each class (ensure order matches CLASS_NAMES)
        # CLASS_NAMES = ['dirt', 'liquid', 'marks', 'trash']
        self.class_colors_bgr = {
            'dirt': (100, 100, 200),  # Light Reddish/Brown for dirt
            'liquid': (200, 150, 50),  # Light Blue for liquid
            'marks': (150, 200, 100),  # Light Greenish for marks
            'trash': (50, 50, 255)  # Bright Red for trash (to stand out)
        }
        # Fallback color if class name not in map
        self.default_box_color_bgr = (0, 200, 200)  # Yellow

        # For basic tracking stability (very simple persistence)
        self.last_known_predictions = []
        self.persistence_frames = 3  # How many frames to persist a box if not re-detected
        self.frames_since_last_strong_detection = {}  # class_id -> count

        print("TrashDetector initialized with custom colors and thresholds.")

    # ... (_preprocess and _postprocess from previous step) ...
    def _preprocess(self, frame_rgb):
        input_image_resized = cv2.resize(frame_rgb, self.input_size)
        input_data = np.expand_dims(input_image_resized, axis=0).astype(np.float32) / 255.0
        return input_data

    def _postprocess(self, output_data_copy, original_frame_shape):
        predictions = []
        num_detections = output_data_copy.shape[1]
        for i in range(num_detections):
            detection = output_data_copy[0][i]
            obj_confidence = detection[4]
            if obj_confidence >= self.default_confidence_threshold * 0.8:  # Pre-filter
                class_scores = detection[5:]
                class_id = np.argmax(class_scores)
                class_confidence_score = class_scores[class_id]
                current_class_name = self.class_names[int(class_id)]
                effective_threshold = self.default_confidence_threshold
                if current_class_name == 'trash':
                    effective_threshold = self.trash_confidence_threshold
                final_confidence = obj_confidence * class_confidence_score
                if final_confidence >= effective_threshold:
                    xc_norm, yc_norm, w_norm, h_norm = detection[:4]
                    predictions.append({
                        'box_normalized': [xc_norm, yc_norm, w_norm, h_norm],
                        'confidence': float(final_confidence),
                        'class_id': int(class_id),
                        'class_name': current_class_name,
                        'unique_id': None  # For more advanced tracking
                    })
        return predictions

    def detect(self, frame_rgb):
        input_data = self._preprocess(frame_rgb)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index']).copy()

        current_predictions = self._postprocess(output_data, frame_rgb.shape)

        # --- Basic Persistence Logic (Example) ---
        # This is a very naive approach. Real tracking is much more complex.
        # It doesn't handle occlusions, multiple instances well, or smooth box movement.
        # For now, let's just use current_predictions directly.
        # Implementing robust tracking is a significant task.
        # self.last_known_predictions = current_predictions # Simplest: just use current
        # --- End Basic Persistence ---

        return current_predictions

    def draw_detections(self, frame_bgr_display, predictions):
        frame_h, frame_w = frame_bgr_display.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        for pred in predictions:
            class_name = pred['class_name']
            confidence = pred['confidence']
            box_norm = pred['box_normalized']

            color = self.class_colors_bgr.get(class_name, self.default_box_color_bgr)  # BGR Color

            xc_norm, yc_norm, w_norm, h_norm = box_norm
            x1 = int((xc_norm - w_norm / 2) * frame_w)
            y1 = int((yc_norm - h_norm / 2) * frame_h)
            x2 = int((xc_norm + w_norm / 2) * frame_w)
            y2 = int((yc_norm + h_norm / 2) * frame_h)

            # Draw the bounding box
            cv2.rectangle(frame_bgr_display, (x1, y1), (x2, y2), color, thickness)

            # Prepare text: Class name and confidence
            label = f"{class_name.upper()}: {confidence:.0%}"  # e.g., TRASH: 65%

            # Get text size to create a filled background rectangle for the label
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale,
                                                                  thickness - 1)  # thinner font for label bg

            # Position the label background just above the top-left corner of the box
            label_bg_y1 = max(y1 - text_height - baseline - 2, 0)  # Ensure it's not off-screen
            label_bg_y2 = y1 - baseline + 2

            # Ensure label background doesn't go off top of screen
            if label_bg_y1 < 0:
                label_bg_y1 = y1 + baseline
                label_bg_y2 = y1 + text_height + baseline + 4

            cv2.rectangle(frame_bgr_display, (x1, label_bg_y1), (x1 + text_width, label_bg_y2), color, cv2.FILLED)

            # Put the text on the background
            # Text color: white for dark backgrounds, black for light backgrounds
            text_color = (255, 255, 255) if sum(color) < 384 else (0, 0, 0)  # Simple brightness check
            cv2.putText(frame_bgr_display, label, (x1, y1 - baseline - 2),  # Adjust y for baseline
                        font, font_scale, text_color, thickness - 1)  # Thinner text

        return frame_bgr_display


if __name__ == '__main__':
    # Simple test for detector.py
    # You'd need a sample image 'sample.jpg' in the same directory
    # or provide a path to an image.
    # This also assumes you have a PiCamera available if you try to use it directly.
    print("Testing Detector...")
    try:
        cap = cv2.VideoCapture(0)  # Or path to a video file
        if not cap.isOpened():
            print("Cannot open camera or video file for testing.")
            # Fallback to a dummy image if no camera
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(dummy_frame, "No Camera", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            test_frame_rgb = cv2.cvtColor(dummy_frame, cv2.COLOR_BGR2RGB)  # Assuming dummy is BGR
        else:
            ret, frame = cap.read()  # OpenCV reads in BGR
            if not ret:
                print("Cannot read frame for testing.")
                exit()
            test_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cap.release()

        detector = TrashDetector()
        predictions = detector.detect(test_frame_rgb)
        print(f"Detections: {predictions}")

        # To visualize, convert the RGB test frame to BGR for OpenCV display
        frame_bgr_display = cv2.cvtColor(test_frame_rgb.copy(), cv2.COLOR_RGB2BGR)
        detector.draw_detections(frame_bgr_display, predictions)

        # cv2.imshow("Detector Test", frame_bgr_display)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        print("Detector test finished (visualization part commented out).")

    except Exception as e:
        print(f"Error during detector test: {e}")