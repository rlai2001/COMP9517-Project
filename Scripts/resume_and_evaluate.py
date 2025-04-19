import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# ---------- Settings ----------
model_path = "C:/Users/Admin/.kaggle/models/resnet_epoch1.pth"
data_dir = "C:/Users/Admin/.kaggle/skyview_dataset/split"
batch_size = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ---------- Transforms ----------
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# ---------- Data ----------
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ---------- Load model ----------
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(test_dataset.classes))
model = model.to(device)

# Load saved weights
model.load_state_dict(torch.load(model_path))
model.eval()
print(f"📦 Loaded model from {model_path}")

# ---------- Evaluate ----------
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# ---------- Save classification report ----------
report_str = classification_report(y_true, y_pred, target_names=test_dataset.classes)
print("\n📊 Classification Report:\n", report_str)

os.makedirs("outputs", exist_ok=True)
epoch_num = 10  # or whichever epoch you're evaluating
report_path = f"outputs/classification_report_epoch{epoch_num}.txt"

with open(report_path, "w") as f:
    f.write(report_str)

print(f"📄 Saved classification report to {report_path}")

# ---------- Save confusion matrix ----------
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_dataset.classes)

plt.figure(figsize=(10, 10))
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
plt.close()
print("📷 Saved confusion matrix to outputs/confusion_matrix.png")
