import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

class UNetEncoderClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(UNetEncoderClassifier, self).__init__()
        self.encoder = nn.Sequential(
            self.conv_block(3, 64),
            nn.MaxPool2d(2),
            self.conv_block(64, 128),
            nn.MaxPool2d(2),
            self.conv_block(128, 256),
            nn.MaxPool2d(2),
            self.conv_block(256, 512)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    @staticmethod
    def conv_block(in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

def smooth_curve(points, factor=0.8):
    smoothed = []
    for point in points:
        if smoothed:
            smoothed.append(smoothed[-1] * factor + point * (1 - factor))
        else:
            smoothed.append(point)
    return smoothed

# Paths
base_path = r"C:/Users/ceren/PycharmProjects/Feng498/dataset/augmented_train"
output_dir = r"C:/Users/ceren/PycharmProjects/Feng498/unet/outputs"
class_map_path = os.path.join(output_dir, "class_map.pth")
best_model_path = os.path.join(output_dir, "best_model.pth")
fold_metrics_path = os.path.join(output_dir, "fold_metrics.csv")

os.makedirs(os.path.join(output_dir, "confusion_matrices"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "roc_curves"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "loss_plots"), exist_ok=True)

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
dataset = datasets.ImageFolder(base_path, transform=transform)
class_map = dataset.class_to_idx
torch.save(class_map, class_map_path)

k_folds = 5
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
targets = [label for _, label in dataset.samples]

fold_metrics = {key: [] for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "specificity", "sensitivity"]}
csv_data = []
best_f1 = 0
best_model_state = None

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
    print(f"\n🌀 Fold {fold + 1}/{k_folds}")
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=32, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetEncoderClassifier(num_classes=len(class_map)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)

    fold_train_losses = []
    fold_val_losses = []

    best_val_loss = float('inf')
    patience = 50
    patience_counter = 0

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
        epoch_train_loss = running_loss / len(train_loader.dataset)
        fold_train_losses.append(epoch_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
        epoch_val_loss = val_loss / len(val_loader.dataset)
        fold_val_losses.append(epoch_val_loss)

        print(f"Epoch {epoch + 1}: Train Loss = {epoch_train_loss:.4f}, Val Loss = {epoch_val_loss:.4f}")
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️ Early stopping at epoch {epoch + 1}")
                break

    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            if probs.shape[1] == 2:
                all_probs.extend(probs[:, 1].cpu().numpy())
            else:
                all_probs.extend(probs.max(dim=1)[0].cpu().numpy())
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_probs)

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

    for key, val in zip(fold_metrics.keys(), [acc, prec, rec, f1, roc_auc, specificity, sensitivity]):
        fold_metrics[key].append(val)
    csv_data.append([fold + 1, acc, prec, rec, f1, roc_auc, specificity, sensitivity])

    if f1 > best_f1:
        best_f1 = f1
        best_model_state = model.state_dict()

    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=class_map.keys(), yticklabels=class_map.keys())
    plt.title(f"Fold {fold+1} - Confusion Matrix")
    plt.savefig(os.path.join(output_dir, "confusion_matrices", f"confusion_matrix_fold{fold+1}.png"))
    plt.clf()

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.plot(fpr, tpr, label=f"Fold {fold+1} AUC={roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color='gray')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "roc_curves", f"roc_curve_fold{fold+1}.png"))
    plt.clf()

    plt.plot(smooth_curve(fold_train_losses), label="Train Loss")
    plt.plot(smooth_curve(fold_val_losses), label="Validation Loss", linestyle='--')
    plt.title(f"Loss Curve - Fold {fold+1} (Smoothed)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_plots", f"loss_curve_fold{fold+1}.png"))
    plt.clf()

torch.save(best_model_state, best_model_path)
print(f"\n💾 En iyi model başarıyla '{best_model_path}' olarak kaydedildi (En iyi F1: {best_f1:.4f})")

with open(fold_metrics_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Fold", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "Specificity", "Sensitivity"])
    writer.writerows(csv_data)

print(f"\n📄 Fold metrikleri '{fold_metrics_path}' olarak kaydedildi.")
print("\n📋 Ortalama Sonuçlar:")
for metric, values in fold_metrics.items():
    print(f"{metric.capitalize()}: {np.mean(values):.4f}")