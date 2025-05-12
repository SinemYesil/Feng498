import os
import torch
import pandas as pd
import numpy as np
from ultralytics import YOLO
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# 📁 Path ayarları
test_path = 'C:/Users/ceren/PycharmProjects/Feng498/dataset/test'
model_path = 'C:/Users/ceren/PycharmProjects/Feng498/yolov8/outputs/train_logs/yolov8_classifier/weights/best.pt'
output_dir = 'C:/Users/ceren/PycharmProjects/Feng498/yolov8/outputs/predict_logs'
os.makedirs(output_dir, exist_ok=True)

# 🧠 Model yükle
model = YOLO(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔍 Test verisini yükle
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
test_dataset = datasets.ImageFolder(test_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
class_map = test_dataset.class_to_idx
inv_map = {v: k for k, v in class_map.items()}

# 🔎 Tahmin yap
all_preds, all_labels, all_probs = [], [], []
for images, labels in test_loader:
    images = images.to(device)
    results = model(images, verbose=False)

    for result, label in zip(results, labels):
        probs = result.probs
        if probs is not None:
            probs_tensor = probs.data  # 🔧 DÜZELTME: Tensor olarak al
            pred = torch.argmax(probs_tensor).item()
            prob = probs_tensor[1].item() if len(probs_tensor) == 2 else torch.max(probs_tensor).item()
        else:
            pred = -1
            prob = 0

        all_preds.append(pred)
        all_labels.append(label.item())
        all_probs.append(prob)

# 🧾 Değerlendirme raporu
report = classification_report(all_labels, all_preds, target_names=list(class_map.keys()), output_dict=True)
pd.DataFrame(report).transpose().to_csv(os.path.join(output_dir, "classification_report.csv"), index=True)

# 🔢 Ek metrikler
cm = confusion_matrix(all_labels, all_preds)
tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
specificity = tn / (tn + fp) if (tn + fp) else 0
sensitivity = tp / (tp + fn) if (tp + fn) else 0
accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

summary_data = {
    "Accuracy": [accuracy],
    "Sensitivity": [sensitivity],
    "Specificity": [specificity]
}
if len(class_map) == 2:
    auc_score = roc_auc_score(all_labels, all_probs)
    summary_data["ROC_AUC"] = [auc_score]

pd.DataFrame(summary_data).to_csv(os.path.join(output_dir, "metrics_summary.csv"), index=False)

# 🎨 Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_map.keys(), yticklabels=class_map.keys())
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.close()

# 📈 ROC Curve (sadece 2 sınıf için)
if len(class_map) == 2:
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

# ===============================
# 📢 Konsola Özet Yazdır
# ===============================
print(f"\n📊 Toplam test görüntüsü: {len(all_labels)}")
print(f"✅ Doğru tahmin: {sum(np.array(all_preds) == np.array(all_labels))}")
print(f"❌ Yanlış tahmin: {len(all_labels) - sum(np.array(all_preds) == np.array(all_labels))}")
print(f"🎯 Doğruluk oranı: {accuracy * 100:.2f}%")
