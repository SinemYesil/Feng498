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

# 🔧 U-Net Encoder Classifier
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

# 📁 Dataset ayarları
base_path = "C:/Users/ceren/PycharmProjects/Feng498/dataset/augmented_train"
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
dataset = datasets.ImageFolder(base_path, transform=transform)
class_map = dataset.class_to_idx
torch.save(class_map, "C:/Users/ceren/PycharmProjects/Feng498/unet/class_map.pth")  # ✅ class_map aynı klasöre kaydedildi

# 📁 Çıktı klasörlerini oluştur
output_dir = "C:/Users/ceren/PycharmProjects/Feng498/unet/outputs"
os.makedirs(f"{output_dir}/confusion_matrices", exist_ok=True)
os.makedirs(f"{output_dir}/roc_curves", exist_ok=True)
os.makedirs(f"{output_dir}/loss_plots", exist_ok=True)

# 📊 Metrikler ve ayarlar
k_folds = 5
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
targets = [label for _, label in dataset.samples]
fold_metrics = { "accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": [], "sensitivity": [], "specificity": [] }
csv_data = []

best_f1 = 0
best_model_state = None

train_loss_history = []
val_loss_history = []

# 🔁 Fold eğitim döngüsü
for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
    print(f"\n🌀 Fold {fold + 1}/{k_folds}")
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=16, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetEncoderClassifier(num_classes=len(class_map)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    fold_train_losses = []
    fold_val_losses = []

    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    for epoch in range(50):
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

        print(f"Epoch {epoch+1}: Train Loss = {epoch_train_loss:.4f}, Val Loss = {epoch_val_loss:.4f}")
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️ Early stopping at epoch {epoch + 1}")
                break

    train_loss_history.append(fold_train_losses)
    val_loss_history.append(fold_val_losses)

    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    fold_metrics["accuracy"].append(acc)
    fold_metrics["precision"].append(prec)
    fold_metrics["recall"].append(rec)
    fold_metrics["f1"].append(f1)
    fold_metrics["roc_auc"].append(roc_auc)
    fold_metrics["sensitivity"].append(sensitivity)
    fold_metrics["specificity"].append(specificity)

    csv_data.append([fold + 1, acc, prec, rec, f1, roc_auc, sensitivity, specificity])

    if f1 > best_f1:
        best_f1 = f1
        best_model_state = model.state_dict()

    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=class_map.keys(), yticklabels=class_map.keys())
    plt.title(f"Fold {fold+1} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"{output_dir}/confusion_matrices/confusion_matrix_fold{fold+1}.png")
    plt.clf()

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.plot(fpr, tpr, label=f"Fold {fold+1} AUC={roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(f"{output_dir}/roc_curves/roc_curve_fold{fold+1}.png")
    plt.clf()

    plt.plot(fold_train_losses, label="Train Loss")
    plt.plot(fold_val_losses, label="Validation Loss", linestyle='--')
    plt.title(f"Loss Curve - Fold {fold+1}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{output_dir}/loss_plots/loss_curve_fold{fold+1}.png")
    plt.clf()

torch.save(best_model_state, "C:/Users/ceren/PycharmProjects/Feng498/unet/best_model.pth")  # ✅ best_model aynı klasöre kaydedildi
print(f"\n💾 En iyi model başarıyla 'C:/Users/ceren/PycharmProjects/Feng498/unet/best_model.pth' olarak kaydedildi (En iyi F1: {best_f1:.4f})")

with open(f"{output_dir}/fold_metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Fold", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "Sensitivity", "Specificity"])
    writer.writerows(csv_data)
print(f"📄 Fold metrikleri '{output_dir}/fold_metrics.csv' olarak kaydedildi.")

print("\n📋 Ortalama Sonuçlar:")
for metric, values in fold_metrics.items():
    print(f"{metric.capitalize()}: {np.mean(values):.4f}")
