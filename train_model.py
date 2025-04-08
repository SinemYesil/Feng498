# train_model.py
import os
from ultralytics import YOLO

# 📁 Dataset klasörünün bulunduğu ana yol
base_path = "/Users/sinemyesil/Desktop/Feng498"

def rename_folders():
    sets = ['train', 'test']
    rename_map = {'NO': '0_NO', 'AMD': '1_AMD'}
    for set_name in sets:
        set_path = os.path.join(base_path, set_name)
        for old_name, new_name in rename_map.items():
            old_dir = os.path.join(set_path, old_name)
            new_dir = os.path.join(set_path, new_name)
            if os.path.exists(old_dir):
                os.rename(old_dir, new_dir)
                print(f"✅ Yeniden adlandırıldı: {old_dir} → {new_dir}")
            else:
                print(f"⚠️ Klasör bulunamadı: {old_dir}")

def train_model():
    model = YOLO("yolov8s-cls.pt")
    model.train(
        data=base_path,
        epochs=50,
        imgsz=224,
        batch=16,
        project="classification_runs",
        name="no_amd_classifier"
    )
    print("✅ Model eğitimi tamamlandı.")

if __name__ == "__main__":
    rename_folders()
    train_model()
