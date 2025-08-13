import cv2
import mediapipe as mp
import numpy as np
import torch
import joblib
import time
import os
from vlc_controller import VLCController

class GestureRecognizer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Load model and labels
        self.model = self.load_model()
        self.gestures = self.load_labels()
        
        # VLC Controller
        self.vlc = VLCController()
        
        # Enhanced gesture state management
        self.reset_gesture_state()
        
    def reset_gesture_state(self):
        """Reset all gesture tracking state"""
        self.palm_detected = False
        self.palm_start_time = 0
        self.last_gesture = None
        self.action_triggered = False
        self.continuous_action = None
        self.continuous_start_time = 0
        self.no_hand_frames = 0
        self.max_no_hand_frames = 15  # ~0.5 seconds at 30fps
        self.gesture_confidence_frames = 0
        self.min_confidence_frames = 3  # Need 3 consistent frames
        self.last_confident_gesture = None
        self.last_action_time = 0
        self.action_cooldown = 0.5  # 500ms cooldown between actions
        
    def load_model(self):
        """Load the trained gesture recognition model"""
        class GestureNet(torch.nn.Module):
            def __init__(self, input_size, hidden_size, num_classes):
                super(GestureNet, self).__init__()
                self.fc1 = torch.nn.Linear(input_size, hidden_size)
                self.relu = torch.nn.ReLU()
                self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
                self.relu2 = torch.nn.ReLU()
                self.fc3 = torch.nn.Linear(hidden_size, num_classes)
            
            def forward(self, x):
                out = self.fc1(x)
                out = self.relu(out)
                out = self.fc2(out)
                out = self.relu2(out)
                out = self.fc3(out)
                return out
        
        # Find model file
        possible_paths = [
            "models/gesture_model.pth",
            "../models/gesture_model.pth",
            os.path.join(os.path.dirname(__file__), "models", "gesture_model.pth"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "gesture_model.pth")
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            raise FileNotFoundError(f"Model file not found. Searched: {possible_paths}")
        
        try:
            model = GestureNet(input_size=63, hidden_size=128, num_classes=5).to(self.device)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            print(f"✅ Model loaded from: {model_path}")
            return model
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")
    
    def load_labels(self):
        """Load gesture labels"""
        possible_paths = [
            "models/gesture_labels.pkl",
            "../models/gesture_labels.pkl",
            os.path.join(os.path.dirname(__file__), "models", "gesture_labels.pkl"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "gesture_labels.pkl")
        ]
        
        labels_path = None
        for path in possible_paths:
            if os.path.exists(path):
                labels_path = path
                break
        
        if labels_path is None:
            raise FileNotFoundError(f"Labels file not found. Searched: {possible_paths}")
        
        try:
            gestures = joblib.load(labels_path)
            print(f"✅ Loaded {len(gestures)} gestures")
            return gestures
        except Exception as e:
            raise RuntimeError(f"Error loading labels: {e}")
    
    def predict_gesture(self, landmarks):
        """Predict gesture from hand landmarks with confidence"""
        try:
            # Normalize landmarks relative to wrist
            landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
            if landmarks_array[0] is not None:
                landmarks_array -= landmarks_array[0]  # Use wrist as origin
            landmarks_flat = landmarks_array.flatten().astype(np.float32)
            
            # Predict using model
            with torch.no_grad():
                inputs = torch.tensor(landmarks_flat, dtype=torch.float32).to(self.device)
                outputs = self.model(inputs.unsqueeze(0))
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                pred_gesture = self.gestures[predicted.item()]
                conf_score = confidence.item()
                
            return pred_gesture, conf_score
        except Exception as e:
            print(f"Prediction error: {e}")
            return "unknown", 0.0
    
    def map_gesture_name(self, predicted_gesture):
        """Map model predictions to standard gesture names"""
        gesture_mapping = {
            "palm": "palm",
            "fist": "fist", 
            "finger_up": "finger_up",
            "finger_down": "finger_down",
            "two_fingers": "two_fingers",
            "open": "palm",
            "closed": "fist",
            "thumbs_up": "finger_up",
            "thumbs_down": "finger_down", 
            "peace": "two_fingers",
            "v": "two_fingers"
        }
        return gesture_mapping.get(predicted_gesture.lower(), predicted_gesture.lower())
    
    def is_gesture_confident(self, gesture, confidence):
        """Check if gesture prediction is confident enough"""
        if gesture == self.last_confident_gesture and confidence > 0.6:
            self.gesture_confidence_frames += 1
        else:
            self.gesture_confidence_frames = 1
            self.last_confident_gesture = gesture
        
        return self.gesture_confidence_frames >= self.min_confidence_frames
    
    def process_gesture_sequence(self, raw_gesture, confidence):
        """Process gesture sequences starting with palm"""
        current_time = time.time()
        gesture = self.map_gesture_name(raw_gesture)
        
        # Only process if we have confident gesture recognition
        if not self.is_gesture_confident(gesture, confidence):
            return f"Detecting... ({gesture} {confidence:.2f})"
        
        # Action cooldown check
        if current_time - self.last_action_time < self.action_cooldown:
            return "⏳ Action cooldown..."
        
        # Palm detection logic
        if gesture == "palm":
            if not self.palm_detected:
                # First time showing palm - activate gesture mode
                self.palm_detected = True
                self.palm_start_time = current_time
                self.action_triggered = False
                self.continuous_action = None
                print("👋 Palm detected - Gesture mode ACTIVE")
                return "👋 Palm Active - Ready for command"
            else:
                # Palm shown again - stop continuous actions
                if self.continuous_action:
                    print(f"✋ Palm shown - stopping continuous action: {self.continuous_action}")
                    self.continuous_action = None
                # Reset action state for new commands
                self.action_triggered = False
                return "👋 Palm Active - Ready for new command"
        
        # Process action gestures only if palm was detected first
        if self.palm_detected and gesture != "palm":
            # Execute action only once per palm session
            if not self.action_triggered:
                self.execute_gesture_action(gesture)
                self.action_triggered = True
                self.last_action_time = current_time
                
                # For continuous actions, start the continuous loop
                if gesture in ["finger_up", "finger_down"]:
                    self.continuous_action = gesture
                    self.continuous_start_time = current_time
                    return f"🔄 Started continuous: {gesture}"
                return f"✅ Executed: {gesture}"
            
            # Handle continuous actions
            elif gesture in ["finger_up", "finger_down"] and self.continuous_action == gesture:
                # Continue action at fixed intervals
                if current_time - self.continuous_start_time >= 0.15:  # Every 150ms
                    self.execute_continuous_action(gesture)
                    self.continuous_start_time = current_time
                    self.last_action_time = current_time
                return f"🔄 Continuous: {gesture}"
            else:
                return f"⏸️ Action already done - show palm to reset"
        
        # If no palm detected, wait for palm
        return f"⏳ Show PALM first (detected: {gesture})"
    
    def execute_gesture_action(self, gesture):
        """Execute one-time gesture actions"""
        if gesture == "fist":
            self.vlc.play_pause()
        elif gesture == "two_fingers":
            self.vlc.mute_toggle()
        elif gesture == "finger_up":
            self.vlc.volume_up(continuous=True)
        elif gesture == "finger_down":
            self.vlc.volume_down(continuous=True)
    
    def execute_continuous_action(self, gesture):
        """Execute continuous actions for volume control"""
        if gesture == "finger_up":
            self.vlc.volume_up(continuous=True)
        elif gesture == "finger_down":
            self.vlc.volume_down(continuous=True)
    
    def run(self):
        """Main recognition loop with gesture state machine"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("🎥 Webcam started. VLC Gesture Control is ACTIVE.")
        prev_time = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Failed to capture frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(image_rgb)
                
                status_text = "No hand detected"
                gesture_info = ""
                confidence_info = ""
                
                if results.multi_hand_landmarks:
                    self.no_hand_frames = 0
                    
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Predict gesture with confidence
                        predicted_gesture, confidence = self.predict_gesture(hand_landmarks)
                        status_text = self.process_gesture_sequence(predicted_gesture, confidence)
                        
                        # Display information
                        gesture_info = f"Gesture: {predicted_gesture}"
                        confidence_info = f"Confidence: {confidence:.2f}"
                        
                        # Draw hand landmarks
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                        )
                        break
                else:
                    self.no_hand_frames += 1
                    
                    # Reset state if no hand detected for too long
                    if self.no_hand_frames >= self.max_no_hand_frames:
                        if self.continuous_action:
                            print("🖐️ Hand removed - stopping continuous action")
                            self.continuous_action = None
                        if self.palm_detected:
                            print("🔄 No hand detected - gesture mode reset")
                            self.reset_gesture_state()
                
                # Create info panel
                info_height = 200
                info_panel = np.zeros((info_height, frame.shape[1], 3), dtype=np.uint8)
                info_panel[:] = (40, 40, 40)  # Dark gray background
                
                # Display status information
                cv2.putText(info_panel, status_text, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                if gesture_info:
                    cv2.putText(info_panel, gesture_info, (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                if confidence_info:
                    cv2.putText(info_panel, confidence_info, (10, 85),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Display palm status
                palm_status = "Palm: ACTIVE ✅" if self.palm_detected else "Palm: Show palm to start ⏳"
                color = (0, 255, 0) if self.palm_detected else (100, 100, 255)
                cv2.putText(info_panel, palm_status, (10, 115),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Display continuous action status
                if self.continuous_action:
                    cv2.putText(info_panel, f"Continuous: {self.continuous_action} 🔄", (10, 145),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                
                # Calculate and display FPS
                current_time = time.time()
                fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
                prev_time = current_time
                cv2.putText(info_panel, f"FPS: {fps:.1f}", (10, 175),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Add ESC instruction
                cv2.putText(frame, "Press ESC to exit", (frame.shape[1] - 200, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Combine frame with info panel
                combined_frame = np.vstack([frame, info_panel])
                
                cv2.imshow("VLC Gesture Control", combined_frame)
                
                # Check for ESC key
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print("🛑 ESC pressed - exiting...")
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("📷 Webcam closed")

def main():
    print("VLC Gesture Control Starting...")
    try:
        recognizer = GestureRecognizer()
        recognizer.run()
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()