import csv
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    roc_curve
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ultralytics import YOLO

# 📁 Path Ayarları
yolo_dataset_path = "C:/Users/ceren/PycharmProjects/Feng498/yolov8/yolo_dataset"
yaml_path = os.path.join(yolo_dataset_path, "data.yaml")
output_dir = "C:/Users/ceren/PycharmProjects/Feng498/yolov8/outputs"
metrics_csv = os.path.join(output_dir, "fold_metrics.csv")
best_model_path = os.path.join(output_dir, "train_run", "weights", "best.pt")

os.makedirs(output_dir, exist_ok=True)

# 🔍 Model başlat
model = YOLO("yolov8n-cls.pt")

# 🏋️ Eğitimi başlat (varsayılan ayarlarla)
model.train(
    data=yaml_path,
    epochs=100,
    imgsz=224,
    batch=32,
    patience=10,
    project=output_dir,
    name="train_run"
)

# 📁 En iyi modeli yükle
best_model = YOLO(best_model_path)

# 📊 Validation verisi
val_path = os.path.join(yolo_dataset_path, "images", "val")
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
val_dataset = datasets.ImageFolder(val_path, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
class_names = val_dataset.classes
num_classes = len(class_names)

true_labels, pred_labels, probs = [], [], []

for imgs, labels in val_loader:
    img_np = imgs[0].permute(1, 2, 0).numpy()
    result = best_model(img_np, verbose=False, imgsz=224)[0]
    pred = int(result.probs.top1)
    prob = float(result.probs.data[labels.item()])
    true_labels.append(labels.item())
    pred_labels.append(pred)
    probs.append(prob)

# 📈 Metrik Hesaplama
acc = accuracy_score(true_labels, pred_labels)
prec = precision_score(true_labels, pred_labels, average='macro')
rec = recall_score(true_labels, pred_labels, average='macro')
f1 = f1_score(true_labels, pred_labels, average='macro')
roc_auc = roc_auc_score(true_labels, probs) if num_classes == 2 else 0

# 📊 Confusion Matrix
cm = confusion_matrix(true_labels, pred_labels)
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(os.path.join(output_dir, "confusion_matrix_val.png"))
plt.close()

# 📉 ROC Curve
if num_classes == 2:
    fpr, tpr, _ = roc_curve(true_labels, probs)
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "roc_curve_val.png"))
    plt.close()

# 📄 CSV Kaydı
with open(metrics_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"])
    writer.writerow([acc, prec, rec, f1, roc_auc])

print("\n✅ Eğitim tamamlandı ve validation sonuçları hesaplandı.")
print(f"📁 Model: {best_model_path}")
print(f"📊 Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC_AUC: {roc_auc:.4f}")
