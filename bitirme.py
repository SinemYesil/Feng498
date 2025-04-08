import os
import shutil
import random

source_dir = '/Users/elifilkayozkan/Desktop/FENG498/DATASET'

os.makedirs(os.path.join(source_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(source_dir, 'test'), exist_ok=True)


classes = ['NO', 'AMD']

# %70 train, %30 test oranı
split_ratio = 0.7

# Her bir sınıf için verileri ayırıyoruz
for cls in classes:
    img_dir = os.path.join(source_dir, cls)
    images = os.listdir(img_dir)
    random.shuffle(images)

    split_point = int(len(images) * split_ratio)
    train_imgs = images[:split_point]
    test_imgs = images[split_point:]

    # Her sınıf için train ve test alt klasörlerini oluştur
    os.makedirs(os.path.join(source_dir, 'train', cls), exist_ok=True)
    os.makedirs(os.path.join(source_dir, 'test', cls), exist_ok=True)

    # Train verilerini kopyala
    for img in train_imgs:
        shutil.copy(os.path.join(img_dir, img), os.path.join(source_dir, 'train', cls, img))

    # Test verilerini kopyala
    for img in test_imgs:
        shutil.copy(os.path.join(img_dir, img), os.path.join(source_dir, 'test', cls, img))

print("Resimler başarıyla %70 train, %30 test olarak ayrıldı.")
