
from pathlib import Path
from PIL import Image
from collections import Counter

DATASET = Path("dataset/not_banknotes")

total = 0
categories = 0
invalid = 0

for folder in sorted(DATASET.iterdir()):
    if not folder.is_dir():
        continue

    count = 0

    for file in folder.iterdir():
        if file.suffix.lower() not in {
            ".jpg", ".jpeg", ".png", ".bmp", ".webp"
        }:
            continue

        try:
            with Image.open(file) as img:
                img.verify()
            count += 1
            total += 1
        except Exception:
            invalid += 1

    if count:
        categories += 1
        print(f"{folder.name}: {count}")

print("\n===== DATASET AUDIT =====")
print("Total categories:", categories)
print("Total valid images:", total)
print("Invalid images:", invalid)
