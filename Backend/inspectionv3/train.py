import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

# ✅ Tip güvenliği ile InceptionV3 sınıfı
class InceptionV3Classifier(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.backbone = models.inception_v3(pretrained=True, aux_logits=False)
        in_features: int = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

def smooth_curve(points, factor=0.8):
    smoothed = []
    for point in points:
        if smoothed:
            smoothed.append(smoothed[-1] * factor + point * (1 - factor))
        else:
            smoothed.append(point)
    return smoothed

# 🔧 Yollar
base_path = r"/Backend/dataset/augmented_train"
output_dir = r"/Backend/inspectionv3/outputs/train"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "confusion_matrices"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "roc_curves"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "loss_plots"), exist_ok=True)

class_map_path = os.path.join(output_dir, "class_map.pth")
best_model_path = os.path.join(output_dir, "best_model.pth")
fold_metrics_path = os.path.join(output_dir, "fold_metrics.csv")

# 🔁 Transform
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(base_path, transform=transform)
class_map = dataset.class_to_idx
torch.save(class_map, class_map_path)

k_folds = 5
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
indices = list(range(len(dataset)))
targets = dataset.targets

fold_metrics = {key: [] for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "specificity", "sensitivity"]}
csv_data = []
best_f1 = 0
best_model_state = None

# 🔄 K-Fold Başlat
for fold, (train_idx, val_idx) in enumerate(skf.split(indices, targets)):
    print(f"\n🌀 Fold {fold + 1}/{k_folds}")
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=8, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InceptionV3Classifier(num_classes=len(class_map)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_val_f1 = 0
    patience_counter = 0
    patience = 10
    fold_train_losses = []
    fold_val_losses = []

    all_preds, all_labels, all_probs = [], [], []
    val_f1 = 0

    for epoch in range(100):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_train_loss = running_loss / len(train_idx)
        fold_train_losses.append(epoch_train_loss)

        model.eval()
        val_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1).cpu().numpy()
                all_probs.extend(probs[:, 1].cpu().numpy() if probs.shape[1] == 2 else probs.max(dim=1)[0].cpu().numpy())
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        epoch_val_loss = val_loss / len(val_idx)
        fold_val_losses.append(epoch_val_loss)
        scheduler.step(epoch_val_loss)

        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        print(f"Epoch {epoch + 1}: Train Loss = {epoch_train_loss:.4f}, Val Loss = {epoch_val_loss:.4f}, Val F1 = {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏹️ Early stopping.")
                break

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = val_f1

    # ✅ ROC güvenli hale getirildi
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        roc_auc = 0

    # ✅ Confusion Matrix güvenli hale getirildi
    try:
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    except ValueError:
        specificity = sensitivity = 0

    for k, v in zip(fold_metrics.keys(), [acc, prec, rec, f1, roc_auc, specificity, sensitivity]):
        fold_metrics[k].append(v)
    csv_data.append([fold + 1, acc, prec, rec, f1, roc_auc, specificity, sensitivity])

    if f1 > best_f1:
        best_f1 = f1
        best_model_state = model.state_dict()

    # 🔻 Görsel kayıt
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_map.keys(), yticklabels=class_map.keys())
    plt.title(f"Fold {fold+1} - Confusion Matrix")
    plt.savefig(f"{output_dir}/confusion_matrices/cm_fold{fold+1}.png")
    plt.clf()

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(f"ROC Curve - Fold {fold+1}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.savefig(f"{output_dir}/roc_curves/roc_fold{fold+1}.png")
    plt.clf()

    plt.plot(smooth_curve(fold_train_losses), label="Train Loss")
    plt.plot(smooth_curve(fold_val_losses), label="Val Loss")
    plt.title(f"Loss Curve - Fold {fold+1}")
    plt.legend()
    plt.savefig(f"{output_dir}/loss_plots/loss_fold{fold+1}.png")
    plt.clf()

# ✅ Best model kayıt
torch.save(best_model_state, best_model_path)
print(f"\n💾 Best model saved to: {best_model_path} (F1: {best_f1:.4f})")

with open(fold_metrics_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Fold", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "Specificity", "Sensitivity"])
    writer.writerows(csv_data)

print("\n📋 Average Results:")
for k, v in fold_metrics.items():
    print(f"{k.capitalize()}: {np.mean(v):.4f}")
