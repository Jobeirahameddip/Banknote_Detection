
import numpy as np
from pathlib import Path
from PIL import Image
from tensorflow.keras.models import load_model

BASE = Path(__file__).resolve().parent
DATASET = BASE / "dataset" / "not_banknotes"
MODEL = BASE / "best_efficientnetb0.keras"

THRESHOLD = 0.5916581749916077
BATCH_SIZE = 8
SIZE = (224, 224)

print("Loading existing model...", flush=True)
model = load_model(MODEL, compile=False)
print("Model loaded successfully!", flush=True)

print("Searching images...", flush=True)

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

files = [
    f for f in DATASET.rglob("*")
    if f.is_file() and f.suffix.lower() in extensions
]

total = len(files)
print("Total images found:", total, flush=True)

genuine = 0
fake = 0
errors = 0

for start in range(0, total, BATCH_SIZE):
    batch_files = files[start:start + BATCH_SIZE]
    images = []

    for file in batch_files:
        try:
            with Image.open(file) as img:
                img = img.convert("RGB")
                img = img.resize(SIZE)
                images.append(np.asarray(img, dtype=np.float32))
        except Exception:
            errors += 1

    if images:
        batch = np.stack(images)
        predictions = model(batch, training=False).numpy().reshape(-1)

        fake += int(np.sum(predictions >= THRESHOLD))
        genuine += int(np.sum(predictions < THRESHOLD))

    completed = min(start + BATCH_SIZE, total)
    print(f"Progress: {completed}/{total}", flush=True)

print("\n===== FINAL RESULTS =====")
print("Total:", total)
print("Genuine:", genuine)
print("Fake:", fake)
print("Errors:", errors)
