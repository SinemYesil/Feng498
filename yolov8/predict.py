import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ultralytics import YOLO

# 📁 Pathler
test_path = "C:/Users/ceren/PycharmProjects/Feng498/dataset/test"
model_path = "C:/Users/ceren/PycharmProjects/Feng498/yolov8/outputs/train_run/weights/best.pt"
output_dir = "C:/Users/ceren/PycharmProjects/Feng498/yolov8/outputs"

# 🔍 Model
model = YOLO(model_path)

# 🔢 Test verisi
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
test_data = datasets.ImageFolder(test_path, transform=transform)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
class_names = test_data.classes

true_labels, pred_labels = [], []

for imgs, labels in test_loader:
    img_np = imgs[0].permute(1, 2, 0).numpy()
    result = model(img_np, verbose=False, imgsz=224)[0]
    pred = int(result.probs.top1)
    true_labels.append(labels.item())
    pred_labels.append(pred)

# 📋 Rapor
print("\n📊 Classification Report:")
print(classification_report(true_labels, pred_labels, target_names=class_names))

total = len(true_labels)
correct = sum([p == l for p, l in zip(pred_labels, true_labels)])
accuracy = correct / total

print(f"\n📊 Toplam test görüntüsü: {total}")
print(f"✅ Doğru tahmin: {correct}")
print(f"❌ Yanlış tahmin: {total - correct}")
print(f"🎯 Doğruluk oranı: {accuracy * 100:.2f}%")

# 🔍 Confusion Matrix
cm = confusion_matrix(true_labels, pred_labels)
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix - Test")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix_test.png"))
plt.show()
