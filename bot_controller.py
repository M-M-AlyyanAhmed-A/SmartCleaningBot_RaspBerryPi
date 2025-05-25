# smart_cleaning_bot/bot_controller.py
import serial
import time
import threading

SERIAL_PORT = '/dev/ttyUSB0'  # Or /dev/ttyACM0
SERIAL_BAUDRATE = 9600

class BotController:
    def __init__(self, port=SERIAL_PORT, baudrate=SERIAL_BAUDRATE):
        self.port = port
        self.baudrate = baudrate
        self.arduino = None
        self.serial_lock = threading.Lock()
        self.bot_state = "IDLE" # IDLE, SIMPLE_CLEANING, SMART_CLEANING
        self.last_trash_detection_time = 0
        self.smart_target_command = None # Sent to Arduino: 'A', 'W', 'D', 'X'

        try:
            self.arduino = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2) # Wait for Arduino to reset
            print("Arduino connected via BotController.")
        except serial.SerialException as e:
            print(f"BotController: Error connecting to Arduino: {e}")
            self.arduino = None

    def send_command(self, command_char):
        if self.arduino and self.arduino.is_open:
            with self.serial_lock:
                print(f"BotController sending to Arduino: {command_char}")
                self.arduino.write(command_char.encode() + b'\n') # Send command with newline
                # Optional: Read response if Arduino sends one
                # response = self.arduino.readline().decode().strip()
                # if response: print(f"Arduino says: {response}")
        else:
            print("BotController: Arduino not connected or port not open.")

    def set_mode_off(self):
        self.send_command('0') # Command '0' for OFF/IDLE on Arduino
        self.bot_state = "IDLE"
        return "Bot Status: OFF (IDLE)"

    def set_mode_simple_clean(self):
        self.send_command('1') # Command '1' for Simple Clean
        self.bot_state = "SIMPLE_CLEANING"
        return "Bot Status: Simple Cleaning Activated"

    def set_mode_smart_clean(self):
        self.send_command('2') # Command '2' for Smart Clean
        self.bot_state = "SMART_CLEANING"
        self.last_trash_detection_time = time.time() # Reset timer
        self.smart_target_command = None # Clear previous target
        return "Bot Status: Smart Cleaning Activated"

    def toggle_vacuum(self):
        self.send_command('V') # Command 'V' to toggle vacuum
        return "Bot Status: Vacuum Toggled" # Arduino confirms actual state via its serial

    def get_current_state_message(self):
        if self.bot_state == "IDLE": return "Bot Status: OFF (IDLE)"
        if self.bot_state == "SIMPLE_CLEANING": return "Bot Status: Simple Cleaning Active"
        if self.bot_state == "SMART_CLEANING": return "Bot Status: Smart Cleaning Active"
        return f"Bot Status: {self.bot_state}"


    def update_smart_targeting(self, detected_trash_predictions, frame_width):
        """
        Processes trash detections and sends commands for smart targeting.
        'detected_trash_predictions' should be a list of prediction dicts
        where each dict has 'class_name' and 'box_normalized'.
        """
        if self.bot_state != "SMART_CLEANING":
            return

        trash_found_this_frame = False
        target_direction_for_arduino = None # 'A', 'W', 'D'

        for pred in detected_trash_predictions:
            if pred['class_name'] == 'trash':
                trash_found_this_frame = True
                xc_norm, _, w_norm, _ = pred['box_normalized']
                # Denormalize x_center to pixel coordinates of the original frame
                box_center_x_pixel = xc_norm * frame_width

                if box_center_x_pixel < frame_width / 3:
                    target_direction_for_arduino = 'A' # Turn Left
                elif box_center_x_pixel > 2 * frame_width / 3:
                    target_direction_for_arduino = 'D' # Turn Right
                else:
                    target_direction_for_arduino = 'W' # Go Center/Forward
                break # Process only the first 'trash' item for simplicity

        if trash_found_this_frame and target_direction_for_arduino:
            self.last_trash_detection_time = time.time()
            if target_direction_for_arduino != self.smart_target_command:
                self.smart_target_command = target_direction_for_arduino
                self.send_command(self.smart_target_command)
                print(f"Smart Command: {self.smart_target_command}")
        elif time.time() - self.last_trash_detection_time > 3.5: # Slightly longer timeout
            if self.smart_target_command is not None and self.smart_target_command != 'X': # Avoid sending X repeatedly
                self.send_command('X') # Stop
                self.smart_target_command = 'X' # Mark as stopped to avoid resending
                print("Smart mode: No trash detected for 3.5s, stopping.")
        # If already stopped (smart_target_command == 'X') and no trash, do nothing further.