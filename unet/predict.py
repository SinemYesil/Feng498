import torch
import os
from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# 🔧 U-Net tabanlı encoder classifier
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import accuracy_score
from tqdm import tqdm

class UNetEncoderClassifier(torch.nn.Module):
    def __init__(self, num_classes=2):
        super(UNetEncoderClassifier, self).__init__()
        self.encoder = torch.nn.Sequential(
            self.conv_block(3, 64),
            torch.nn.MaxPool2d(2),
            self.conv_block(64, 128),
            torch.nn.MaxPool2d(2),
            self.conv_block(128, 256),
            torch.nn.MaxPool2d(2),
            self.conv_block(256, 512)
        )
        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, num_classes)
        )

    @staticmethod
    def conv_block(in_c, out_c):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_c, out_c, 3, padding=1),
            torch.nn.BatchNorm2d(out_c),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_c, out_c, 3, padding=1),
            torch.nn.BatchNorm2d(out_c),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

# 📁 Ayarlar
test_root = "C:/Users/ceren/PycharmProjects/Feng498/dataset/test"
model_path = "C:/Users/ceren/PycharmProjects/Feng498/unet/best_model.pth"
class_map_path = "C:/Users/ceren/PycharmProjects/Feng498/unet/class_map.pth"
class_map = torch.load(class_map_path)
idx_to_class = {v: k for k, v in class_map.items()}
class_to_idx = {k: v for v, k in idx_to_class.items()}

# 🧪 Görsel ön işleme
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 🔮 Tüm test verisi üzerinde tahmin
def evaluate_all_test_images():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetEncoderClassifier(num_classes=len(class_map))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    y_true, y_pred = [], []

    for cls_name in os.listdir(test_root):
        cls_dir = os.path.join(test_root, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        true_label = class_to_idx[cls_name]

        for img_name in tqdm(os.listdir(cls_dir), desc=f"🔍 {cls_name} sınıfı test ediliyor"):
            img_path = os.path.join(cls_dir, img_name)
            image = Image.open(img_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                pred = torch.argmax(output, dim=1).item()

            y_true.append(true_label)
            y_pred.append(pred)

    correct = sum([1 for t, p in zip(y_true, y_pred) if t == p])
    total = len(y_true)
    accuracy = accuracy_score(y_true, y_pred)

    print(f"\n📊 Toplam test görüntüsü: {total}")
    print(f"✅ Doğru tahmin: {correct}")
    print(f"❌ Yanlış tahmin: {total - correct}")
    print(f"🎯 Doğruluk oranı: {accuracy * 100:.2f}%")

# 🚀 Çalıştır
if __name__ == "__main__":
    evaluate_all_test_images()


