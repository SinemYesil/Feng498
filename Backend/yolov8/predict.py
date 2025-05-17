import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import random
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import pandas as pd
from ultralytics import YOLO


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🧬 Seed set to: {seed}")


set_seed(42)


class Predict:
    def __init__(self, model_path, test_dir, output_dir):
        self.model_path = Path(model_path)
        self.test_dir = Path(test_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = YOLO(str(self.model_path))
        self.class_names = self.model.names  # {0: 'AMD', 1: 'NO'}

        self._rename_test_class_folders()

        self.all_labels = []
        self.all_preds = []
        self.all_probs = []

    def _rename_test_class_folders(self):
        print("🔤 Test klasör isimleri yeniden adlandırılıyor...")
        for class_idx, class_name in self.class_names.items():
            old_path = self.test_dir / class_name
            new_path = self.test_dir / str(class_idx)
            if old_path.exists() and not new_path.exists():
                old_path.rename(new_path)
                print(f"🔁 {old_path} → {new_path}")

    def evaluate(self):
        print("🔍 Starting prediction on test images...")
        for class_idx, class_name in self.class_names.items():
            class_folder = self.test_dir / str(class_idx)
            if not class_folder.exists():
                print(f"⚠️ Skipping missing class folder: {class_folder}")
                continue

            images = list(class_folder.glob("*"))
            if not images:
                print(f"⚠️ No images found in: {class_folder}")
                continue

            for img_path in images:
                try:
                    result = self.model(img_path, imgsz=299, verbose=False)[0]
                    pred_idx = int(result.probs.top1)
                    prob_tensor = result.probs.data

                    self.all_preds.append(pred_idx)
                    self.all_labels.append(class_idx)

                    if len(self.class_names) == 2:
                        self.all_probs.append(float(prob_tensor[1]))
                    else:
                        self.all_probs.append(float(torch.max(prob_tensor).item()))
                except Exception as e:
                    print(f"❌ Error processing {img_path}: {e}")

        if self.all_labels:
            self._save_classification_report()
            self._save_confusion_matrix()
            self._save_roc_curve()
        else:
            print("⚠️ Test verisi bulunamadı veya işlenemedi. Hiçbir çıktı oluşturulmadı.")

    def _save_classification_report(self):
        acc = accuracy_score(self.all_labels, self.all_preds)
        prec = precision_score(self.all_labels, self.all_preds, average="macro", zero_division=0)
        rec = recall_score(self.all_labels, self.all_preds, average="macro", zero_division=0)
        f1 = f1_score(self.all_labels, self.all_preds, average="macro", zero_division=0)

        results_df = pd.DataFrame({
            "Accuracy": [acc],
            "Precision": [prec],
            "Recall": [rec],
            "F1 Score": [f1]
        })
        results_csv_path = self.output_dir / "results_summary.csv"
        results_df.to_csv(results_csv_path, index=False)

        report = classification_report(
            self.all_labels,
            self.all_preds,
            target_names=list(self.class_names.values()),
            output_dict=False,
            zero_division=0,
        )
        report_path = self.output_dir / "classification_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        print(f"📄 Classification report saved to: {report_path}")

        print(f"\n📊 Toplam test görüntüsü: {len(self.all_labels)}")
        print(f"✅ Doğru tahmin: {sum(p == l for p, l in zip(self.all_preds, self.all_labels))}")
        print(f"❌ Yanlış tahmin: {sum(p != l for p, l in zip(self.all_preds, self.all_labels))}")
        print(f"🎯 Doğruluk oranı: {acc * 100:.2f}%")
        print(f"📁 Metrics and visuals saved to: {self.output_dir}")

    def _save_confusion_matrix(self):
        cm = confusion_matrix(self.all_labels, self.all_preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                    xticklabels=list(self.class_names.values()),
                    yticklabels=list(self.class_names.values()))
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        cm_path = self.output_dir / "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        print(f"📊 Confusion matrix saved to: {cm_path}")

    def _save_roc_curve(self):
        if len(self.class_names) == 2:
            try:
                fpr, tpr, _ = roc_curve(self.all_labels, self.all_probs)
                roc_auc = roc_auc_score(self.all_labels, self.all_probs)

                plt.figure()
                plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC Curve")
                plt.legend()
                roc_path = self.output_dir / "roc_curve.png"
                plt.savefig(roc_path)
                plt.close()
                print(f"📈 ROC curve saved to: {roc_path}")
            except Exception as e:
                print(f"⚠️ ROC curve could not be generated: {e}")
        else:
            print("ℹ️ ROC curve only supported for binary classification.")


# 📌 Örnek kullanım
if __name__ == "__main__":
    predictor = Predict(
        model_path="C:/Users/ceren/PycharmProjects/Feng498/Backend/yolov8/outputs/yolov8_cls_run/weights/best.pt",
        test_dir="C:/Users/ceren/PycharmProjects/Feng498/Backend/dataset/test",
        output_dir="C:/Users/ceren/PycharmProjects/Feng498/Backend/yolov8/outputs/predict"
    )
    predictor.evaluate()
