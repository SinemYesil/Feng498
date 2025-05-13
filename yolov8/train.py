import shutil
import random
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO


class Train:
    def __init__(self, source_dir: str, yolov8_dir: str, split_ratio: float = 0.8, seed: int = 42):
        self.source_dir = Path(source_dir)
        self.base_output_dir = Path(yolov8_dir)
        self.renamed_dir = self.base_output_dir / "renamed"
        self.split_dir = self.base_output_dir / "split"
        self.train_dir = self.split_dir / "train"
        self.val_dir = self.split_dir / "val"
        self.output_dir = self.base_output_dir.parent / "outputs"
        self.split_ratio = split_ratio
        self.seed = seed
        self.supported_exts = [".jpg", ".jpeg", ".png"]

        self.set_seed()

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"🧬 Seed ayarlandı: {self.seed}")

    def standardize_filenames(self):
        print("🔤 Dosya adları düzenleniyor ve 'renamed' klasörüne kopyalanıyor...")
        for cls in self.source_dir.iterdir():
            if not cls.is_dir():
                continue
            images = [f for f in cls.glob("*") if f.suffix.lower() in self.supported_exts]
            new_cls_dir = self.renamed_dir / cls.name
            new_cls_dir.mkdir(parents=True, exist_ok=True)
            for i, img in enumerate(sorted(images), start=1):
                new_name = f"{cls.name}_{i:04d}{img.suffix.lower()}"
                new_path = new_cls_dir / new_name
                if new_path.exists():
                    print(f"⚠️ Hedef dosya zaten mevcut: {new_path}. Atlanıyor.")
                    continue
                shutil.copy2(img, new_path)
        print(f"✅ Dosyalar '{self.renamed_dir}' klasörüne standart isimlerle kopyalandı.")

    def split_dataset(self):
        print("🔀 Eğitim/Doğrulama veri kümesi ayrılıyor...")
        for cls in self.renamed_dir.iterdir():
            if not cls.is_dir():
                continue
            images = [f for f in cls.glob("*") if f.suffix.lower() in self.supported_exts]
            if not images:
                print(f"⚠️ {cls.name} sınıfı boş, atlanıyor.")
                continue
            random.shuffle(images)
            split_idx = int(len(images) * self.split_ratio)
            train_imgs = images[:split_idx]
            val_imgs = images[split_idx:]

            for subset, subset_imgs in [("train", train_imgs), ("val", val_imgs)]:
                out_dir = self.split_dir / subset / cls.name
                out_dir.mkdir(parents=True, exist_ok=True)
                for img in tqdm(subset_imgs, desc=f"{cls.name} - {subset}"):
                    shutil.copy2(img, out_dir / img.name)
        print(f"✅ '{self.split_dir}' altında train/val klasörleri oluşturuldu.")

    def train_model(self):
        print("🏁 YOLOv8 sınıflandırma eğitimi başlıyor...")
        model = YOLO("yolov8n-cls.pt")  # Hafif model

        results = model.train(
            data=str(self.split_dir),
            epochs=100,
            imgsz=224,
            batch=8,
            lr0=0.001,
            patience=10,
            project=str(self.output_dir),
            name="yolov8_cls_run",
            verbose=True
        )

        # ✅ save_dir güvenli kontrol
        save_dir = results.get("save_dir") if isinstance(results, dict) else getattr(results, "save_dir", None)
        if save_dir is None:
            raise RuntimeError("❌ Eğitim sonrası save_dir bulunamadı.")

        best_model_path = Path(save_dir) / "best.pt"
        backup_path = self.output_dir / "best_backup.pt"
        shutil.copy2(best_model_path, backup_path)

        print(f"\n✅ En iyi model: {best_model_path}")
        print(f"🗃️ Yedek kopya: {backup_path}")

    def run(self):
        self.standardize_filenames()
        self.split_dataset()
        self.train_model()


# 📌 Çalıştırmak için
if __name__ == "__main__":
    trainer = Train(
        source_dir="C:/Users/Ceren/PycharmProjects/Feng498/dataset/augmented_train",
        yolov8_dir="C:/Users/Ceren/PycharmProjects/Feng498/yolov8/dataset",
        split_ratio=0.8,
        seed=42
    )
    trainer.run()
