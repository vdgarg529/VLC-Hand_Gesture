import cv2
import mediapipe as mp
import numpy as np
import torch
import joblib
import time  # Added missing import for FPS calculation

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # For drawing landmarks

# Initialize hands processor
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Load model and labels
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

# Initialize model
print("Loading model...")
try:
    model = GestureNet(input_size=63, hidden_size=128, num_classes=5).to(device)
    model.load_state_dict(torch.load(r"Neural Network\models\gesture_model.pth", map_location=device))
    model.eval()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

print("Loading labels...")
try:
    gestures = joblib.load(r"Neural Network\models\gesture_labels.pkl")
    print(f"Loaded {len(gestures)} gesture labels: {gestures}")
except Exception as e:
    print(f"Error loading labels: {e}")
    exit()

# Video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture")
    exit()

print("Starting recognition. Press ESC to exit...")

# For FPS calculation
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break
    
    # Process frame with MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract and normalize landmarks
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            
            # Handle division by zero if wrist landmark is missing
            if landmarks[0] is not None:
                landmarks -= landmarks[0]  # Use wrist as origin
            landmarks = landmarks.flatten().astype(np.float32)
            
            # Convert to tensor and predict
            try:
                with torch.no_grad():
                    inputs = torch.tensor(landmarks, dtype=torch.float32).to(device)
                    outputs = model(inputs.unsqueeze(0))  # Add batch dimension
                    _, predicted = torch.max(outputs, 1)
                    pred_gesture = gestures[predicted.item()]
            except Exception as e:
                print(f"Prediction error: {e}")
                pred_gesture = "Error"
            
            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))
            
            # Display prediction
            cv2.putText(frame, f"Gesture: {pred_gesture}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
    prev_time = current_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.imshow("Gesture Recognition", frame)
    
    # Check for ESC key press
    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
print("Program ended")