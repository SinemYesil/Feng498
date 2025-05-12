import shutil
import random
from pathlib import Path
import yaml
from tqdm import tqdm

class DatasetProcessorContinueForYoloV8:
    def __init__(self, source_dir, yolo_output_dir, split_ratio=0.8, image_extension=".jpg"):
        self.source_dir = Path(source_dir)
        self.yolo_base = Path(yolo_output_dir)
        self.train_dir = self.yolo_base / "images" / "train"
        self.val_dir = self.yolo_base / "images" / "val"
        self.yaml_path = self.yolo_base / "data.yaml"
        self.split_ratio = split_ratio
        self.image_extension = image_extension

    def _reset_directories(self):
        for folder in [self.train_dir, self.val_dir]:
            if folder.exists():
                shutil.rmtree(folder)
            folder.mkdir(parents=True, exist_ok=True)

    def process(self):
        self._reset_directories()
        classes = [d.name for d in self.source_dir.iterdir() if d.is_dir()]

        for cls in classes:
            cls_path = self.source_dir / cls
            images = list(cls_path.glob(f"*{self.image_extension}"))
            random.shuffle(images)
            split_idx = int(len(images) * self.split_ratio)
            train_images = images[:split_idx]
            val_images = images[split_idx:]

            train_cls_dir = self.train_dir / cls
            val_cls_dir = self.val_dir / cls
            train_cls_dir.mkdir(parents=True, exist_ok=True)
            val_cls_dir.mkdir(parents=True, exist_ok=True)

            for img in tqdm(train_images, desc=f"{cls} - Train"):
                shutil.copy(img, train_cls_dir / img.name)
            for img in tqdm(val_images, desc=f"{cls} - Val"):
                shutil.copy(img, val_cls_dir / img.name)

        data_config = {
            "path": str(self.yolo_base).replace("\\", "/"),
            "train": "images/train",
            "val": "images/val",
            "nc": len(classes),
            "names": classes
        }

        with open(self.yaml_path, "w") as f:
            yaml.dump(data_config, f)

        print(f"\n✅ YOLOv8 için veriler bölündü ve kaydedildi.")
        print(f"📄 data.yaml oluşturuldu: {self.yaml_path}")

if __name__ == "__main__":
    processor = DatasetProcessorContinueForYoloV8(
        source_dir="C:/Users/ceren/PycharmProjects/Feng498/dataset/augmented_train",
        yolo_output_dir="C:/Users/ceren/PycharmProjects/Feng498/yolov8/yolo_dataset"
    )
    processor.process()
