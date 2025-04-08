from ultralytics import YOLO
import os

# ❗ Eğitimi tamamladıysan, burada best.pt yolunu doğru verdiğinden emin ol
model_path = "/Users/sinemyesil/Desktop/Feng498/classification_runs/no_amd_classifier/weights/best.pt"
image_path = "/Users/sinemyesil/Desktop/Feng498/dataset/NO/some_image.jpg"

model = YOLO(model_path)
results = model.predict(image_path)

pred_index = results[0].probs.top1
raw_label = results[0].names[pred_index]
label = "normal" if "NO" in raw_label else "hastalıklı"

print(f"\n📸 Görsel: {os.path.basename(image_path)}")
print(f"✅ Tahmin sonucu: {label}")
