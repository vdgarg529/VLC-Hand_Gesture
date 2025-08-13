import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import random
from torch.utils.data import DataLoader, TensorDataset

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
gestures = ["palm", "fist", "finger_up", "finger_down", "two_fingers"]

# Data collection parameters
NUM_SAMPLES_PER_GESTURE = 50  # Reduced for live capture
AUGMENTATION_FACTOR = 5       # Generate 5 augmented samples per real sample

def collect_gesture_data():
    """Capture gesture data from webcam with MediaPipe"""
    X, y = [], []
    
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7
    ) as hands:
        
        print("Press 'c' to capture a sample for the current gesture")
        print("Press 'n' to move to the next gesture")
        print("Press 'ESC' to exit")

        for gesture_idx, gesture in enumerate(gestures):
            print(f"\nShow gesture: {gesture} ({NUM_SAMPLES_PER_GESTURE} samples needed)")
            sample_count = 0
            
            while sample_count < NUM_SAMPLES_PER_GESTURE:
                ret, frame = cap.read()
                if not ret:
                    continue

                # Process frame with MediaPipe
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Extract and normalize landmarks
                        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                        landmarks -= landmarks[0]  # Use wrist as origin
                        landmarks = landmarks.flatten()
                        
                        # Display instructions
                        cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f"Samples: {sample_count}/{NUM_SAMPLES_PER_GESTURE}", (10, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('c'):
                            X.append(landmarks)
                            y.append(gesture_idx)
                            sample_count += 1
                            print(f"Sample {sample_count}/{NUM_SAMPLES_PER_GESTURE} captured for {gesture}")
                        elif key == ord('n'):
                            # Skip to next gesture if not enough samples
                            print(f"Moving to next gesture (only {sample_count} samples collected)")
                            sample_count = NUM_SAMPLES_PER_GESTURE
                else:
                    cv2.putText(frame, "No hand detected", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv2.imshow("Collecting Data", frame)
                if cv2.waitKey(1) == 27:  # ESC key
                    cap.release()
                    cv2.destroyAllWindows()
                    return None, None

    cap.release()
    cv2.destroyAllWindows()
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def augment_landmarks(landmarks):
    """Apply random transformations to landmarks for data augmentation"""
    # Reshape to (21, 3)
    points = landmarks.reshape(21, 3)
    
    # Random rotation (-15 to 15 degrees)
    angle = np.radians(random.uniform(-15, 15))
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    points = np.dot(points, rotation_matrix)
    
    # Random scaling (0.9 to 1.1)
    scale = random.uniform(0.9, 1.1)
    points *= scale
    
    # Random noise
    noise = np.random.normal(0, 0.01, points.shape)
    points += noise
    
    return points.flatten()

def create_augmented_dataset(X, y):
    """Generate augmented dataset"""
    X_aug, y_aug = [], []
    
    for i in range(len(X)):
        # Original sample
        X_aug.append(X[i])
        y_aug.append(y[i])
        
        # Augmented samples
        for _ in range(AUGMENTATION_FACTOR):
            X_aug.append(augment_landmarks(X[i]))
            y_aug.append(y[i])
    
    return np.array(X_aug), np.array(y_aug)

# Define neural network
class GestureNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(GestureNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

# Main execution
if __name__ == "__main__":
    # Collect data
    X, y = collect_gesture_data()
    if X is None:
        exit()
    
    print(f"\nCollected {len(X)} samples")
    
    # Data augmentation
    X_aug, y_aug = create_augmented_dataset(X, y)
    print(f"Created augmented dataset with {len(X_aug)} samples")
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_aug, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_aug, dtype=torch.long).to(device)
    
    # Create DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    model = GestureNet(
        input_size=63,  # 21 landmarks * 3 coordinates
        hidden_size=128,
        num_classes=len(gestures)
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print("\nTraining...")
    model.train()
    for epoch in range(50):  # Reduced epochs for quick training
        total_loss = 0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/50], Loss: {total_loss/len(dataloader):.4f}")
    
    # Save model and labels
    torch.save(model.state_dict(), "gesture_model.pth")
    joblib.dump(gestures, "gesture_labels.pkl")
    print("\nModel saved as gesture_model.pth")