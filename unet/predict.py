import os
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 🔒 Tüm rastgelelikleri sabitle
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

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

# 📁 Yollar
test_path = "C:/Users/ceren/PycharmProjects/Feng498/dataset/test"
output_dir = "C:/Users/ceren/PycharmProjects/Feng498/unet/outputs"
class_map_path = os.path.join(output_dir, "class_map.pth")
best_model_path = os.path.join(output_dir, "best_model.pth")

# 📦 Cihaz ve veriler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_map = torch.load(class_map_path)
inv_map = {v: k for k, v in class_map.items()}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
test_data = datasets.ImageFolder(test_path, transform=transform)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

# 🔍 Modeli yükle
model = UNetEncoderClassifier(num_classes=len(class_map)).to(device)
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

# 🧠 Tahmin
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# 📊 Rapor
print("\n📊 Sınıflandırma Raporu:")
print(classification_report(all_labels, all_preds, target_names=list(class_map.keys())))

total = len(all_labels)
correct = sum(np.array(all_preds) == np.array(all_labels))
accuracy = correct / total
print(f"\n📊 Toplam test görüntüsü: {total}")
print(f"✅ Doğru tahmin: {correct}")
print(f"❌ Yanlış tahmin: {total - correct}")
print(f"🎯 Doğruluk oranı: {accuracy * 100:.2f}%")

# 🌀 Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=class_map.keys(), yticklabels=class_map.keys())
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix_test.png"))
plt.show()
