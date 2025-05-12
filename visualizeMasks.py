from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random

# 📁 Ana dataset yolu ve preview yolu
base_dataset_dir = Path("C:/Users/ceren/PycharmProjects/Feng498/dataset")
preview_dir = base_dataset_dir / "preview"
preview_dir.mkdir(exist_ok=True, parents=True)

classes = ['AMD', 'NO']

# 🧽 Yardımcı

def load_image_as_array(path):
    img = Image.open(path).convert("RGB").resize((224, 224))
    return np.asarray(img, dtype=np.float32) / 255.0

# 📦 Her sınıftan rastgele 2 örnek seç
examples = []
for cls in classes:
    class_dir = base_dataset_dir / cls
    images = list(class_dir.glob('*'))
    selected_images = random.sample(images, 2)

    for img_path in selected_images:
        original_img = load_image_as_array(img_path)

        # Brightness maskesi (maksimum kanal değeri ile)
        brightness_mask = np.max(original_img, axis=2, keepdims=True)

        # Brightness çarpımı
        brightness_img = np.clip(original_img * brightness_mask, 0, 1)

        examples.append({
            "class": cls,
            "original": original_img,
            "brightness": brightness_img
        })

# 🎨 Grafik oluştur
fig, axs = plt.subplots(len(examples), 2, figsize=(12, len(examples)*5))
titles = ["Original", "Original x Brightness Mask"]

for i, ex in enumerate(examples):
    label = ex["class"]
    variations = [
        ex["original"],
        ex["brightness"]
    ]

    for j in range(2):
        axs[i, j].imshow(variations[j])
        axs[i, j].axis('off')
        if i == 0:
            axs[i, j].set_title(titles[j], fontsize=14, fontweight='bold')
        if j == 0:
            axs[i, j].text(-0.1, 0.5, label, fontsize=14, fontweight='bold',
                           ha='right', va='center', transform=axs[i, j].transAxes, rotation=90)

# Genel başlık
plt.suptitle("Preprocessing Visualization (Original x Brightness Only)", fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Kaydet ve göster
output_path = preview_dir / "visualization_brightness_only.png"
plt.savefig(output_path, dpi=300)
plt.show()
