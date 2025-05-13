import shutil
import random
from pathlib import Path
from tqdm import tqdm


class DatasetProcessorContinueForYolo:

    def __init__(self, source_dir: str, yolov8_dataset_dir: str, split_ratio: float = 0.8):
        self.source_dir = Path(source_dir)
        self.base_output_dir = Path(yolov8_dataset_dir)
        self.renamed_dir = self.base_output_dir / "renamed"
        self.split_dir = self.base_output_dir / "split"
        self.train_dir = self.split_dir / "train"
        self.val_dir = self.split_dir / "val"
        self.split_ratio = split_ratio
        self.supported_exts = [".jpg", ".jpeg", ".png"]

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
        print(f"\n✅ Dosya adları düzenlendi ve '{self.renamed_dir}' klasörüne kopyalandı.")

    def split_dataset(self):
        print("🔀 Veri kümesi eğitim ve doğrulama alt kümelerine ayrılıyor...")
        for cls in self.renamed_dir.iterdir():
            if not cls.is_dir():
                continue
            images = [f for f in cls.glob("*") if f.suffix.lower() in self.supported_exts]
            if not images:
                print(f"⚠️ {cls.name} sınıfı boş, atlandı.")
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
        print(f"\n✅ Veri kümesi '{self.split_dir}' klasöründe eğitim ve doğrulama alt kümelerine ayrıldı.")

    def prepare(self):
        self.standardize_filenames()
        self.split_dataset()


# 📌 Örnek kullanım
if __name__ == "__main__":
    processor = DatasetProcessorContinueForYolo(
        source_dir="C:/Users/Ceren/PycharmProjects/Feng498/dataset/augmented_train",
        yolov8_dataset_dir="C:/Users/Ceren/PycharmProjects/Feng498/yolov8/dataset",
        split_ratio=0.8
    )
    processor.prepare()
