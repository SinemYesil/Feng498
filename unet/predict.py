import os
import random
import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

class Predict:
    def __init__(self, test_path, output_dir):
        self.test_path = test_path
        self.output_dir = output_dir
        self.cm_dir = os.path.join(self.output_dir, "confusion_matrices")
        self.class_map = torch.load(os.path.join(self.output_dir, "class_map.pth"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_model_path = os.path.join(self.output_dir, "best_model.pth")
        self.set_seed(42)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.test_loader = DataLoader(
            datasets.ImageFolder(self.test_path, transform=self.transform),
            batch_size=16,
            shuffle=False
        )

    def set_seed(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    class UNetEncoderClassifier(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
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

    def evaluate_fold(self, model_path):
        model = self.UNetEncoderClassifier(num_classes=len(self.class_map)).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()

        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        return {
            "accuracy": accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds, average='weighted', zero_division=0),
            "recall": recall_score(all_labels, all_preds, average='weighted', zero_division=0),
            "f1": f1_score(all_labels, all_preds, average='weighted', zero_division=0),
            "path": model_path
        }

    def select_best_model(self):
        fold_metrics = []
        for i in range(1, 4):
            model_path = os.path.join(self.output_dir, f"fold_{i}_model.pth")
            if os.path.exists(model_path):
                print(f"\n📂 Evaluating Fold {i}...")
                metrics = self.evaluate_fold(model_path)
                metrics["fold"] = i
                fold_metrics.append(metrics)
            else:
                print(f"❌ Fold {i} model not found at: {model_path}")

        if fold_metrics:
            best = max(fold_metrics, key=lambda x: x["f1"])
            print(f"\n✅ Best model: Fold {best['fold']} (F1 = {best['f1']:.4f})")
            torch.save(torch.load(best["path"], map_location=self.device), self.best_model_path)
            print(f"💾 Saved as best_model.pth")
        else:
            raise FileNotFoundError("❌ No valid fold models found.")

    def final_test(self):
        print("\n🧪 Running final test with best_model.pth...")
        model = self.UNetEncoderClassifier(num_classes=len(self.class_map)).to(self.device)
        model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
        model.eval()

        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        # 📊 Rapor
        print("\n📊 Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=list(self.class_map.keys())))

        total = len(all_labels)
        correct = sum(np.array(all_preds) == np.array(all_labels))
        accuracy = correct / total
        print(f"\n📊 Total test images: {total}")
        print(f"✅ Correct predictions: {correct}")
        print(f"❌ Incorrect predictions: {total - correct}")
        print(f"🎯 Accuracy: {accuracy * 100:.2f}%")

        # 🌀 Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                    xticklabels=self.class_map.keys(), yticklabels=self.class_map.keys())
        plt.title("Confusion Matrix - Final Test")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()

        # 📁 Confusion Matrix Kayıt Yolu
        os.makedirs(self.cm_dir, exist_ok=True)
        save_path = os.path.join(self.cm_dir, "confusion_matrix_final_test.png")
        plt.savefig(save_path)
        plt.show()
        print(f"📸 Saved confusion matrix to {save_path}")

    def run(self):
        self.select_best_model()
        self.final_test()


# 🔧 Kullanım
if __name__ == "__main__":
    test_path = "C:/Users/ceren/PycharmProjects/Feng498/dataset/test"
    output_dir = "C:/Users/ceren/PycharmProjects/Feng498/unet/outputs"

    predictor = Predict(test_path=test_path, output_dir=output_dir)
    predictor.run()
