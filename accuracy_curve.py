
import matplotlib.pyplot as plt

train_acc = [0.38, 0.46, 0.52, 0.57, 0.61, 0.64, 0.67, 0.69, 0.71, 0.72, 0.73, 0.74, 0.75, 0.75, 0.76]
val_acc   = [0.41, 0.44, 0.50, 0.53, 0.55, 0.57, 0.58, 0.59, 0.60, 0.60, 0.61, 0.61, 0.62, 0.62, 0.63]

plt.figure(figsize=(7,4))
plt.plot(train_acc, label="Train Acc")
plt.plot(val_acc, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training & Validation Accuracy")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("accuracy_curve.png", dpi=150)
plt.show()
