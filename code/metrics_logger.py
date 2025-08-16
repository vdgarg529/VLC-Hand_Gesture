import time
import json
import csv
import os
from datetime import datetime

class MetricsLogger:
    def __init__(self, log_dir="logs", session_name=None):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name = session_name or f"session_{timestamp}"
        self.json_path = os.path.join(log_dir, f"{self.session_name}.json")
        self.csv_path = os.path.join(log_dir, f"{self.session_name}.csv")

        self.metrics = {
            "training": [],
            "inference": [],
            "gestures": [],
            "user": {},
            "vlc": {}
        }

    # ---------------- Training ----------------
    def log_training_epoch(self, epoch, loss, accuracy, duration):
        entry = {"epoch": epoch, "loss": loss, "accuracy": accuracy, "duration": duration}
        self.metrics["training"].append(entry)
        print(f"[TRAIN] Epoch {epoch}: Loss={loss:.4f}, Acc={accuracy:.2f}, Time={duration:.2f}s")

    # ---------------- Inference ----------------
    def log_inference(self, latency_ms, fps, confidence, gesture):
        entry = {"latency_ms": latency_ms, "fps": fps,
                 "confidence": confidence, "gesture": gesture}
        self.metrics["inference"].append(entry)
        print(f"[INFER] {gesture} | {confidence:.2f} | {latency_ms:.1f}ms | {fps:.1f} FPS")

    # ---------------- Gesture ----------------
    def log_gesture_event(self, gesture, success=True, continuous=False):
        entry = {"gesture": gesture, "success": success, "continuous": continuous,
                 "time": time.time()}
        self.metrics["gestures"].append(entry)
        print(f"[GESTURE] {gesture} | {'Success' if success else 'Fail'} | Continuous={continuous}")

    # ---------------- User ----------------
    def log_user_session(self, duration, gesture_count, adoption_rate=None):
        self.metrics["user"] = {
            "duration": duration,
            "gesture_count": gesture_count,
            "adoption_rate": adoption_rate
        }
        print(f"[USER] Session {duration:.1f}s | Gestures={gesture_count} | Adoption={adoption_rate}")

    # ---------------- VLC ----------------
    def log_vlc_status(self, api_success_rate, fallback_rate):
        self.metrics["vlc"] = {
            "api_success_rate": api_success_rate,
            "fallback_rate": fallback_rate
        }
        print(f"[VLC] API Success={api_success_rate:.1f}% | Fallback={fallback_rate:.1f}%")

    # ---------------- Save ----------------
    def save(self):
        # Save JSON
        with open(self.json_path, "w") as f:
            json.dump(self.metrics, f, indent=4)

        # Save CSV (only inference for simplicity)
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["gesture", "confidence", "latency_ms", "fps"])
            writer.writeheader()
            for entry in self.metrics["inference"]:
                writer.writerow(entry)

        print(f"[LOG] Metrics saved to {self.json_path} and {self.csv_path}")
