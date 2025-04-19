import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse

# ---- Command-line Arguments ----
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./skyview_dataset/split', help='Path with train/ and test/ folders')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.0005)
args = parser.parse_args()

# ---- Device ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---- Transforms ----
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# ---- Load Datasets ----
train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'train'), transform=train_transform)
test_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'test'), transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# ---- Load Pretrained Model ----
model = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# ---- Training ----
best_accuracy = 0.0
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
train_f1s, val_f1s = [], []
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

for epoch in range(args.epochs):
    model.train()
    running_loss = 0.0
    y_train_true, y_train_pred = [], []
    pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]")

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        y_train_true.extend(labels.cpu().numpy())
        y_train_pred.extend(predicted.cpu().numpy())
        pbar.set_postfix(loss=loss.item())

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    train_accuracies.append(accuracy_score(y_train_true, y_train_pred))
    train_f1s.append(f1_score(y_train_true, y_train_pred, average='macro'))

    # Save model for this epoch
    model_path = f"models/resnet_epoch{epoch+1}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # ---- Evaluation ----
    model.eval()
    y_true, y_pred = [], []
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    avg_val_loss = val_loss / len(test_loader)
    val_losses.append(avg_val_loss)
    val_accuracies.append(accuracy_score(y_true, y_pred))
    val_f1s.append(f1_score(y_true, y_pred, average='macro'))

    correct = sum(np.array(y_true) == np.array(y_pred))
    total = len(y_true)
    accuracy = correct / total

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), "models/best_model.pth")
        print(f"Best model updated. Accuracy: {accuracy:.4f}")

    report_str = classification_report(y_true, y_pred, target_names=test_dataset.classes)
    report_path = f"outputs/classification_report_epoch{epoch+1}.txt"
    with open(report_path, "w") as f:
        f.write(report_str)
    print(f"Saved classification report to {report_path}")

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_dataset.classes)
    plt.figure(figsize=(10, 10))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title(f"Confusion Matrix (Epoch {epoch+1})")
    cm_path = f"outputs/confusion_matrix_epoch{epoch+1}.png"
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")

# ---- Plot Loss, Accuracy, F1 Curves ----
epochs_range = range(1, args.epochs + 1)
plt.figure(figsize=(18, 5))

# Loss
plt.subplot(1, 3, 1)
plt.plot(epochs_range, train_losses, label='Train Loss')
plt.plot(epochs_range, val_losses, label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy
plt.subplot(1, 3, 2)
plt.plot(epochs_range, train_accuracies, label='Train Acc')
plt.plot(epochs_range, val_accuracies, label='Val Acc')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# F1
plt.subplot(1, 3, 3)
plt.plot(epochs_range, train_f1s, label='Train F1')
plt.plot(epochs_range, val_f1s, label='Val F1')
plt.title('F1 Score Curve')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()

plt.tight_layout()
plt.savefig("outputs/training_curves.png")
plt.close()
print("Saved training curves to outputs/training_curves.png")

# ---- Final Summary ----
print(f"Best Accuracy Achieved: {best_accuracy:.4f}")
