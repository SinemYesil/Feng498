from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random

# 📁 Ana dataset klasörü
base_dataset_dir = Path("C:/Users/ceren/PycharmProjects/Feng498/Backend/dataset")

# 📁 preview çıktısı dataset ile aynı seviyede olacak
preview_dir = base_dataset_dir.parent / "preprocessingOutputExample"  # ✅ .parent ile "Backend/" klasörüne çıktık
preview_dir.mkdir(parents=True, exist_ok=True)

classes = ['AMD', 'NO']

# 🧽 Yardımcı
def load_image_as_array(path):
    img = Image.open(path).convert("RGB").resize((299, 299))  # Inception için uygun
    return np.asarray(img, dtype=np.float32) / 255.0

# 📦 Her sınıftan rastgele 2 örnek seç
examples = []
for cls in classes:
    class_dir = base_dataset_dir / cls
    images = list(class_dir.glob('*'))
    selected_images = random.sample(images, 2)

    for img_path in selected_images:
        original_img = load_image_as_array(img_path)

        # Contrast maskesi
        contrast_mask = np.max(original_img, axis=2, keepdims=True)
        contrast_img = np.clip(original_img * contrast_mask, 0, 1)

        examples.append({
            "class": cls,
            "original": original_img,
            "contrast": contrast_img
        })

# 🎨 Görsel çizimi
fig, axs = plt.subplots(len(examples), 2, figsize=(12, len(examples)*5))
titles = ["Original Image", "After Contrast Mask Applied"]

for i, ex in enumerate(examples):
    label = ex["class"]
    variations = [ex["original"], ex["contrast"]]

    for j in range(2):
        axs[i, j].imshow(variations[j])
        axs[i, j].axis('off')
        if i == 0:
            axs[i, j].set_title(titles[j], fontsize=14, fontweight='bold')
        if j == 0:
            axs[i, j].text(-0.1, 0.5, label, fontsize=14, fontweight='bold',
                           ha='right', va='center', transform=axs[i, j].transAxes, rotation=90)

plt.suptitle("Preprocessing Visualization: Contrast Enhancement", fontsize=18, fontweight='bold')
plt.tight_layout(rect=(0, 0.03, 1, 0.95))

# ✅ Görseli kaydet
output_path = preview_dir / "preprocessing_contrast_visual_comparison.png"
plt.savefig(output_path, dpi=300)
plt.show()
