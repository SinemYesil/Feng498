import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

class DatasetProcessor:
    def __init__(self, original_dir, split_ratio=0.7, num_augmented_versions=5):
        self.original_dir = Path(original_dir)
        self.split_ratio = split_ratio
        self.num_augmented_versions = num_augmented_versions

        self.classes = ['AMD', 'NO']
        self.train_dir = self.original_dir / 'train'
        self.test_dir = self.original_dir / 'test'
        self.augmented_train_dir = self.original_dir / 'augmented_train'
        self.preprocess_dir = self.original_dir / 'preprocessed'

        self._reset_directories([self.train_dir, self.test_dir, self.augmented_train_dir, self.preprocess_dir])

    @staticmethod
    def _reset_directories(dirs):
        for folder in dirs:
            if folder.exists():
                shutil.rmtree(folder)
            folder.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def features_sat(image_block):
        n, m, d = image_block.shape
        SR = np.zeros((n, m))
        for row in range(n):
            for col in range(m):
                R, G, B = image_block[row, col]
                SR[row, col] = np.std([R, G, B])
        SIm = np.stack([SR] * 3, axis=2)
        max_val = SIm.max()
        return SIm / max_val if max_val != 0 else np.zeros_like(SIm)

    @staticmethod
    def features_b(image_block):
        n, m, d = image_block.shape
        BR = np.zeros((n, m))
        BG = np.zeros((n, m))
        BB = np.zeros((n, m))
        for row in range(n):
            for col in range(m):
                R, G, B = image_block[row, col]
                avg = (R + G + B) / 3
                BR[row, col] = abs(R - avg)
                BG[row, col] = abs(G - avg)
                BB[row, col] = abs(B - avg)
        BIm = np.stack([BR, BG, BB], axis=2)
        max_val = BIm.max()
        return BIm / max_val if max_val != 0 else np.zeros_like(BIm)

    def preprocess(self):
        print("🔀 Preprocessing başlıyor...")
        for cls in self.classes:
            cls_dir = self.original_dir / cls
            images = list(cls_dir.glob('*'))
            random.shuffle(images)

            preprocess_cls_dir = self.preprocess_dir / f'preprocess_{cls}'
            preprocess_cls_dir.mkdir(parents=True, exist_ok=True)

            for img_path in tqdm(images, desc=f"{cls} için preprocessing"):
                try:
                    with Image.open(img_path) as image:
                        image = image.convert("RGB")
                        image_np = np.asarray(image).astype(np.float64) / 255.0

                        mask_s = np.nan_to_num(self.features_sat(image_np))
                        mask_b = np.nan_to_num(self.features_b(image_np))

                        processed_image = np.clip(image_np * mask_s * mask_b, 0, 1)
                        output_img = Image.fromarray((processed_image * 255).astype(np.uint8))
                        output_img.save(preprocess_cls_dir / f"{img_path.stem}_IPS.jpg")
                except Exception as e:
                    print(f"⚠️ Hata ({img_path.name}): {e}")
        print("✅ Preprocessing tamamlandı.")

    def split_data(self):
        print("🔀 Split işlemi başlıyor...")
        for cls in self.classes:
            cls_dir = self.preprocess_dir / f'preprocess_{cls}'
            images = list(cls_dir.glob('*'))
            random.shuffle(images)

            split_point = int(len(images) * self.split_ratio)
            train_imgs = images[:split_point]
            test_imgs = images[split_point:]

            train_cls_dir = self.train_dir / cls
            test_cls_dir = self.test_dir / cls
            train_cls_dir.mkdir(parents=True, exist_ok=True)
            test_cls_dir.mkdir(parents=True, exist_ok=True)

            for img in train_imgs:
                shutil.copy2(img, train_cls_dir / img.name)
            for img in test_imgs:
                shutil.copy2(img, test_cls_dir / img.name)

        print("✅ Split işlemi tamamlandı.")

    def augment_data(self):
        augment_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(20),
            transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomPerspective(0.2, p=0.5),
        ])

        print("🎨 Augmentasyon başlıyor...")
        for cls in self.classes:
            input_dir = self.train_dir / cls
            output_dir = self.augmented_train_dir / cls
            output_dir.mkdir(parents=True, exist_ok=True)

            for img_path in tqdm(input_dir.glob('*'), desc=f"{cls} için augmentasyon"):
                with Image.open(img_path) as image:
                    image = image.convert("RGB")
                    image.resize((224, 224)).save(output_dir / img_path.name)

                    for i in range(self.num_augmented_versions):
                        aug_img = augment_transforms(image)
                        aug_img = aug_img.resize((224, 224))
                        aug_img.save(output_dir / f"{img_path.stem}_aug{i}.jpg")

        print("✅ Augmentasyon tamamlandı.")

    def process(self):
        self.preprocess()
        self.split_data()
        self.augment_data()

if __name__ == '__main__':
    processor = DatasetProcessor(original_dir='C:/Users/ceren/PycharmProjects/Feng498/dataset')
    processor.process()
