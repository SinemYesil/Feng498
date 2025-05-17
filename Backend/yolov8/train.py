import shutil
from pathlib import Path
from ultralytics import YOLO


class Train:
    def __init__(self, source_dir: str, yolov8_dir: str, split_ratio: float = 0.8, seed: int = 42):
        self.source_dir = Path(source_dir)
        self.base_output_dir = Path(yolov8_dir)
        self.renamed_dir = self.base_output_dir / "renamed"
        self.split_dir = self.base_output_dir / "split"
        self.output_dir = self.base_output_dir.parent / "outputs"
        self.split_ratio = split_ratio
        self.seed = seed
        self.supported_exts = [".jpg", ".jpeg", ".png"]

    def train_model(self):
        print("🏁 YOLOv8 sınıflandırma eğitimi başlıyor...")

        model = YOLO("yolov8n-cls.pt")

        results = model.train(
            data=str(self.split_dir),
            epochs=100,
            imgsz=299,
            batch=4,
            lr0=0.001,
            patience=10,
            project=str(self.output_dir),
            name="yolov8_cls_run",
            verbose=True
        )

        # save_dir kontrolü
        save_dir = results.get("save_dir") if isinstance(results, dict) else getattr(results, "save_dir", None)
        if save_dir is None:
            raise RuntimeError("❌ Eğitim sonrası save_dir bulunamadı.")

        best_model_path = Path(save_dir) / "weights" / "best.pt"
        backup_path = self.output_dir / "best_backup.pt"

        if best_model_path.exists():
            shutil.copy2(best_model_path, backup_path)
            print(f"\n✅ En iyi model: {best_model_path}")
            print(f"🗃️ Yedek kopya: {backup_path}")
        else:
            print(f"⚠️ Uyarı: 'best.pt' dosyası oluşturulmadı.")
            print(f"❌ Aranan dosya: {best_model_path}")

        if best_model_path.exists():
            shutil.copy2(best_model_path, backup_path)
            print(f"\n✅ En iyi model: {best_model_path}")
            print(f"🗃️ Yedek kopya: {backup_path}")
        else:
            print(f"⚠️ Uyarı: 'best.pt' dosyası oluşturulmadı.")
            print(f"💡 Muhtemelen early stopping devreye girdi ya da model hiç iyileşmedi.")
            print(f"❌ Aranan dosya: {best_model_path}")

    def run(self):
        self.train_model()


# 📌 Çalıştırmak için
if __name__ == "__main__":
    trainer = Train(
        source_dir="C:/Users/ceren/PycharmProjects/Feng498/Backend/dataset/augmented_train",
        yolov8_dir="C:/Users/ceren/PycharmProjects/Feng498/Backend/yolov8/dataset",
        split_ratio=0.8,
        seed=42
    )
    trainer.run()
