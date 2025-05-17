import random
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

class DatasetProcessor:
    def __init__(self, original_dir, split_ratio=0.7, num_augmented_versions=2):
        self.original_dir = Path(original_dir)
        self.split_ratio = split_ratio
        self.num_augmented_versions = num_augmented_versions
        self.classes = ['AMD', 'NO']
        self.train_dir = self.original_dir / 'train'
        self.test_dir = self.original_dir / 'test'
        self.augmented_train_dir = self.original_dir / 'augmented_train'
        self.preprocess_dir = self.original_dir / 'preprocessed'
        self._reset_directories([
            self.train_dir,
            self.test_dir,
            self.augmented_train_dir,
            self.preprocess_dir
        ])

    @staticmethod
    def _reset_directories(dirs):
        for folder in dirs:
            if folder.exists():
                shutil.rmtree(folder)
            folder.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def apply_contrast_mask(image):
        n, m, _ = image.shape
        BR = np.zeros((n, m))
        BG = np.zeros((n, m))
        BB = np.zeros((n, m))
        for row in range(n):
            for col in range(m):
                R, G, B = image[row, col]
                avg = (R + G + B) / 3
                BR[row, col] = abs(R - avg)
                BG[row, col] = abs(G - avg)
                BB[row, col] = abs(B - avg)
        BIm = np.stack([BR, BG, BB], axis=-1)
        max_val = BIm.max()
        BIm = BIm / max_val if max_val != 0 else np.zeros_like(BIm)
        return np.clip(BIm ** 0.7, 0, 1)

    def preprocess(self):
        print("🔀 Preprocessing başlıyor...")
        for cls in self.classes:
            cls_dir = self.original_dir / cls
            images = list(cls_dir.glob('*'))
            preprocess_cls_dir = self.preprocess_dir / f'preprocess_{cls}'
            preprocess_cls_dir.mkdir(parents=True, exist_ok=True)

            for img_path in tqdm(images, desc=f"{cls} için preprocessing"):
                try:
                    with Image.open(img_path) as img:
                        img = img.convert("RGB").resize((299, 299))  # ✅ Güncellendi
                        img_np = np.asarray(img, dtype=np.float32) / 255.0

                        Ib = self.apply_contrast_mask(img_np)
                        Ips = np.clip(img_np * Ib, 0, 1)

                        Image.fromarray((Ips * 255).astype(np.uint8)).save(
                            preprocess_cls_dir / f"{img_path.stem}_IPS.jpg")
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
        print("🎨 Augmentasyon başlıyor...")
        augment_transforms = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),  # Augmentasyonları tensor üstünde yapıyoruz
        ])
        to_pil = transforms.ToPILImage()  # Tensor → PIL dönüşümü için

        for cls in self.classes:
            input_dir = self.train_dir / cls
            output_dir = self.augmented_train_dir / cls
            output_dir.mkdir(parents=True, exist_ok=True)

            for img_path in tqdm(input_dir.glob('*'), desc=f"{cls} için augmentasyon"):
                with Image.open(img_path) as image:
                    image = image.convert("RGB").resize((299, 299))
                    # Orijinal resmi direkt kaydet
                    image.save(output_dir / img_path.name)

                    for i in range(self.num_augmented_versions):
                        aug_tensor = augment_transforms(image)  # Tensor olarak döner
                        aug_img_pil = to_pil(aug_tensor)  # Tensor'dan PIL'e çevir
                        aug_img_pil.save(output_dir / f"{img_path.stem}_aug{i}.jpg")
        print("✅ Augmentasyon tamamlandı.")

    def process(self):
        self.preprocess()
        self.split_data()
        self.augment_data()

if __name__ == '__main__':
    processor = DatasetProcessor(original_dir='dataset')
    processor.process()
