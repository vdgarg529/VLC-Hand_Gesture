import json
import matplotlib.pyplot as plt
import os

# ---------------- CONFIG ----------------
JSON_FILE = "logs\gesture_training.json"
OUTPUT_DIR = "training_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- LOAD METRICS ----------------
with open(JSON_FILE, "r") as f:
    metrics = json.load(f)

training = metrics.get("training", [])
averages = metrics.get("averages", {})

epochs = [e["epoch"] for e in training]

loss = [e["loss"] for e in training]
accuracy = [e["accuracy"] for e in training]
precision = [e.get("precision", 0) for e in training]
recall = [e.get("recall", 0) for e in training]
f1 = [e.get("f1", 0) for e in training]

val_acc = [e.get("val_accuracy", 0) for e in training]
val_prec = [e.get("val_precision", 0) for e in training]
val_rec = [e.get("val_recall", 0) for e in training]
val_f1 = [e.get("val_f1", 0) for e in training]

# ---------------- INDIVIDUAL PLOTS ----------------
def plot_metric(train_values, val_values, ylabel, filename):
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_values, label=f"Train {ylabel}", marker="o")
    if val_values is not None:
        plt.plot(epochs, val_values, label=f"Val {ylabel}", marker="x")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"✅ Saved: {filename}")

plot_metric(loss, None, "Loss", "loss_curve.png")
plot_metric(accuracy, val_acc, "Accuracy", "accuracy_curve.png")
plot_metric(precision, val_prec, "Precision", "precision_curve.png")
plot_metric(recall, val_rec, "Recall", "recall_curve.png")
plot_metric(f1, val_f1, "F1 Score", "f1_curve.png")

# ---------------- DASHBOARD PLOT ----------------
fig, axs = plt.subplots(3, 2, figsize=(12,10))
axs = axs.ravel()

# Loss
axs[0].plot(epochs, loss, marker="o")
axs[0].set_title("Loss")
axs[0].grid(True)

# Accuracy
axs[1].plot(epochs, accuracy, label="Train", marker="o")
axs[1].plot(epochs, val_acc, label="Val", marker="x")
axs[1].set_title("Accuracy")
axs[1].legend()
axs[1].grid(True)

# Precision
axs[2].plot(epochs, precision, label="Train", marker="o")
axs[2].plot(epochs, val_prec, label="Val", marker="x")
axs[2].set_title("Precision")
axs[2].legend()
axs[2].grid(True)

# Recall
axs[3].plot(epochs, recall, label="Train", marker="o")
axs[3].plot(epochs, val_rec, label="Val", marker="x")
axs[3].set_title("Recall")
axs[3].legend()
axs[3].grid(True)

# F1 Score
axs[4].plot(epochs, f1, label="Train", marker="o")
axs[4].plot(epochs, val_f1, label="Val", marker="x")
axs[4].set_title("F1 Score")
axs[4].legend()
axs[4].grid(True)

# Hide last empty subplot (since 3x2 = 6 slots, but we have 5 metrics)
fig.delaxes(axs[5])

plt.suptitle("Training & Validation Metrics Dashboard", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUTPUT_DIR, "metrics_dashboard.png"))
plt.close()

print("\n📊 Averages from training:")
for k, v in averages.items():
    print(f"{k.capitalize()}: {v:.4f}")

print(f"\n✅ All plots saved in: {OUTPUT_DIR}")
