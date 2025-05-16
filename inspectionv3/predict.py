import os
import csv
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve
)
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

class Predictor:
    def __init__(self, test_data_path, model_path, class_map_path, output_dir, batch_size=16, device=None):
        self.test_path = test_data_path
        self.model_path = model_path
        self.class_map_path = class_map_path
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 📁 outputs ve predictions klasörleri
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir = self.output_dir.parent / "predictions"
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

        # 🗂️ Sınıf eşlemesi
        self.class_map = torch.load(self.class_map_path)
        self.idx_to_class = {v: k for k, v in self.class_map.items()}

        # 🔁 Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # 📦 Test verisi
        self.test_dataset = datasets.ImageFolder(self.test_path, transform=self.transform)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        # 🧠 inspectionv3 mimarisi (InceptionV3)
        self.model = models.inception_v3(weights=None, aux_logits=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.class_map))
        self.model.AuxLogits.fc = nn.Linear(self.model.AuxLogits.fc.in_features, len(self.class_map))
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self):
        all_preds, all_labels, all_probs, all_filenames = [], [], [], []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())

            for path, _ in self.test_dataset.samples:
                all_filenames.append(Path(path).name)

        return np.array(all_labels), np.array(all_preds), np.array(all_probs), all_filenames

    def evaluate(self):
        y_true, y_pred, y_probs, filenames = self.predict()

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        roc_auc = roc_auc_score(y_true, y_probs[:, 1]) if y_probs.shape[1] == 2 else roc_auc_score(y_true, y_probs, multi_class="ovo")

        # 🔍 Sensitivity & Specificity
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        except ValueError:
            specificity = 0
            sensitivity = 0

        # 📊 Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                    xticklabels=list(self.idx_to_class.values()),
                    yticklabels=list(self.idx_to_class.values()))
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(self.output_dir / "confusion_matrix.png")
        plt.clf()

        # 📈 ROC curve
        if y_probs.shape[1] == 2:
            fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
            plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()
            plt.savefig(self.output_dir / "roc_curve.png")
            plt.clf()

        # 💾 predictions.csv → predictions/ klasörüne
        with open(self.predictions_dir / "predictions.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Filename", "TrueLabel", "PredictedLabel", "Confidence"])
            for fname, true_idx, pred_idx, probs in zip(filenames, y_true, y_pred, y_probs):
                confidence = probs[pred_idx]
                writer.writerow([
                    fname,
                    self.idx_to_class[true_idx],
                    self.idx_to_class[pred_idx],
                    f"{confidence:.4f}"
                ])

        # 💾 test_metrics.csv → predictions/ klasörüne
        with open(self.predictions_dir / "test_metrics.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "Specificity", "Sensitivity"])
            writer.writerow([acc, prec, rec, f1, roc_auc, specificity, sensitivity])

        # 📢 Console Output
        correct = np.sum(y_true == y_pred)
        total = len(y_true)
        print(f"\n📊 Total Images: {total}")
        print(f"✅ Correct Predictions: {correct}")
        print(f"❌ Incorrect Predictions: {total - correct}")
        print(f"🎯 Accuracy: {acc * 100:.2f}%")


# ▶️ Ana çalıştırma bloğu
if __name__ == "__main__":
    test_dir_path = "/Users/ceren/PycharmProjects/Feng498/dataset/test"
    results_output_dir = "/Users/ceren/PycharmProjects/Feng498/inspectionv3/outputs"
    model_file_path = os.path.join(results_output_dir, "best_model.pth")
    class_index_map_path = os.path.join(results_output_dir, "class_map.pth")

    predictor = Predictor(
        test_data_path=test_dir_path,
        model_path=model_file_path,
        class_map_path=class_index_map_path,
        output_dir=results_output_dir
    )
    predictor.evaluate()
