import os
import torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from ultralytics import YOLO

class YOLOv8Classifier:
    def __init__(self, model_path, class_map_path, test_root, img_size=640):
        self.model_path = model_path
        self.class_map_path = class_map_path
        self.test_root = test_root
        self.img_size = img_size
        self.class_map = torch.load(self.class_map_path)
        self.idx_to_class = {v: k for k, v in self.class_map.items()}
        self.class_to_idx = {k: v for v, k in self.idx_to_class.items()}
        self.model = YOLO(self.model_path)  # Load YOLOv8 model

    def evaluate_all_test_images(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        y_true, y_pred = [], []

        for cls_name in os.listdir(self.test_root):
            cls_dir = os.path.join(self.test_root, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            true_label = self.class_to_idx[cls_name]

            for img_name in tqdm(os.listdir(cls_dir), desc=f"🔍 {cls_name} sınıfı test ediliyor"):
                img_path = os.path.join(cls_dir, img_name)
                image = Image.open(img_path).convert("RGB")
                input_tensor = self.transform_image(image)

                # Perform inference with YOLOv8
                with torch.no_grad():
                    results = self.model(input_tensor)
                    pred = results.pred[0]  # Get predictions for the first image

                    if len(pred) > 0:
                        # Assuming single-class detection
                        pred_class = pred[:, -1].cpu().numpy().astype(int)
                        pred_label = pred_class[0]  # Choose the most confident prediction
                    else:
                        pred_label = -1  # No detection, set to invalid class

                y_true.append(true_label)
                y_pred.append(pred_label)

        accuracy = accuracy_score(y_true, y_pred)

        correct = sum([1 for t, p in zip(y_true, y_pred) if t == p])
        total = len(y_true)
        print(f"\n📊 Toplam test görüntüsü: {total}")
        print(f"✅ Doğru tahmin: {correct}")
        print(f"❌ Yanlış tahmin: {total - correct}")
        print(f"🎯 Doğruluk oranı: {accuracy * 100:.2f}%")

    def transform_image(self, image):
        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])
        return transform(image).unsqueeze(0)  # Add batch dimension

# 📁 Ayarlar
test_root = "C:/Users/ceren/PycharmProjects/Feng498/dataset/test"
model_path = "../Feng498/yolov8/best_model.pt"  # ✅ YOLO model path
class_map_path = "../Feng498/yolov8/class_map.pth"  # ✅ class_map path

# 🚀 Çalıştır
if __name__ == "__main__":
    yolo_classifier = YOLOv8Classifier(model_path=model_path, class_map_path=class_map_path, test_root=test_root)
    yolo_classifier.evaluate_all_test_images()
