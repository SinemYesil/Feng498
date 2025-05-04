import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, roc_curve)
from ultralytics import YOLO
from tqdm import tqdm

class YOLOv8KFoldClassifier:
    def __init__(self, num_folds=5, patience=3, epochs=50, batch_size=16, img_size=640):
        self.dataset_path = r"C:/Users/ceren/PycharmProjects/Feng498/dataset/augmented_train"
        self.output_dir = r"C:/Users/ceren/PycharmProjects/Feng498/yolov8"
        self.num_folds = num_folds
        self.patience = patience
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/confusion_matrices", exist_ok=True)
        os.makedirs(f"{self.output_dir}/roc_curves", exist_ok=True)
        os.makedirs(f"{self.output_dir}/loss_plots", exist_ok=True)

    def run(self):
        labels_dir = os.path.join(self.dataset_path, 'labels')
        all_labels = []
        for file in os.listdir(labels_dir):
            with open(os.path.join(labels_dir, file), 'r') as f:
                labels = [int(line.split()[0]) for line in f.readlines()]
                label = max(set(labels), key=labels.count) if labels else 0
                all_labels.append(label)

        indices = np.arange(len(all_labels))
        skf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=42)

        fold_metrics = { "accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": [], "sensitivity": [], "specificity": [] }
        csv_data = []
        best_f1 = 0
        best_model = None

        for fold, (train_idx, val_idx) in enumerate(skf.split(indices, all_labels)):
            print(f"\n🌀 Fold {fold+1}/{self.num_folds}")

            fold_dir = os.path.join(self.output_dir, f"fold_{fold+1}")
            os.makedirs(fold_dir, exist_ok=True)

            train_list = [os.path.join(self.dataset_path, 'images', f"{idx:06d}.jpg") for idx in train_idx]
            val_list = [os.path.join(self.dataset_path, 'images', f"{idx:06d}.jpg") for idx in val_idx]
            train_txt = os.path.join(fold_dir, "train.txt")
            val_txt = os.path.join(fold_dir, "val.txt")

            with open(train_txt, "w") as f:
                f.writelines([f"{path}\n" for path in train_list])
            with open(val_txt, "w") as f:
                f.writelines([f"{path}\n" for path in val_list])

            yaml_content = f"""
path: {self.dataset_path}
train: {train_txt}
val: {val_txt}
nc: 2
names: ['class0', 'class1']
"""
            yaml_file = os.path.join(fold_dir, "data.yaml")
            with open(yaml_file, "w") as f:
                f.write(yaml_content)

            model = YOLO('yolov8n-cls.pt')
            results = model.train(
                data=yaml_file,
                epochs=self.epochs,
                patience=self.patience,
                imgsz=self.img_size,
                batch=self.batch_size,
                project=fold_dir,
                name='yolov8_fold',
                verbose=True
            )

            metrics = model.val(data=yaml_file, save_json=True, plots=True)

            preds = metrics.results_dict['preds']
            acc_manual = accuracy_score(all_labels[val_idx], preds)
            prec = precision_score(all_labels[val_idx], preds, zero_division=0)
            rec = recall_score(all_labels[val_idx], preds, zero_division=0)
            f1 = f1_score(all_labels[val_idx], preds, zero_division=0)
            probs = metrics.results_dict['probs']
            roc_auc = roc_auc_score(all_labels[val_idx], probs[:,1])

            cm = confusion_matrix(all_labels[val_idx], preds)
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            fold_metrics["accuracy"].append(acc_manual)
            fold_metrics["precision"].append(prec)
            fold_metrics["recall"].append(rec)
            fold_metrics["f1"].append(f1)
            fold_metrics["roc_auc"].append(roc_auc)
            fold_metrics["sensitivity"].append(sensitivity)
            fold_metrics["specificity"].append(specificity)

            csv_data.append([fold+1, acc_manual, prec, rec, f1, roc_auc, sensitivity, specificity])

            sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
            plt.title(f"Fold {fold+1} - Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.savefig(f"{self.output_dir}/confusion_matrices/confusion_matrix_fold{fold+1}.png")
            plt.clf()

            fpr, tpr, _ = roc_curve(all_labels[val_idx], probs[:,1])
            plt.plot(fpr, tpr, label=f"Fold {fold+1} AUC={roc_auc:.2f}")
            plt.plot([0, 1], [0, 1], linestyle="--", color='gray')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()
            plt.savefig(f"{self.output_dir}/roc_curves/roc_curve_fold{fold+1}.png")
            plt.clf()

            plt.plot(results.metrics.box.loss, label="Train Loss")
            plt.title(f"Loss Curve - Fold {fold+1}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(f"{self.output_dir}/loss_plots/loss_curve_fold{fold+1}.png")
            plt.clf()

            if f1 > best_f1:
                best_f1 = f1
                best_model = model

        if best_model:
            best_model_path = os.path.join(self.output_dir, "best_model.pt")
            best_model.save(best_model_path)
            print(f"💾 Best model saved at '{best_model_path}' (F1={best_f1:.4f})")

        with open(os.path.join(self.output_dir, "fold_metrics.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Fold", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "Sensitivity", "Specificity"])
            writer.writerows(csv_data)

        print("\n📋 Average Metrics:")
        for metric, values in fold_metrics.items():
            print(f"{metric.capitalize()}: {np.mean(values):.4f}")
