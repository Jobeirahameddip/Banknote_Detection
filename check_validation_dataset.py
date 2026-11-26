
from pathlib import Path
from PIL import Image

BASE = Path(__file__).resolve().parent
DATASET = BASE / "dataset"

folders = {
    "Real Notes": DATASET / "banknote_validation" / "real_notes",
    "Fake Notes": DATASET / "banknote_validation" / "fake_notes",
    "Not Banknotes": DATASET / "not_banknotes",
}

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

print("\n===== DATASET VALIDATION =====\n")

for name, folder in folders.items():
    total = 0
    valid = 0
    invalid = 0

    for file in folder.rglob("*"):
        if file.is_file() and file.suffix.lower() in extensions:
            total += 1
            try:
                with Image.open(file) as img:
                    img.verify()
                valid += 1
            except Exception:
                invalid += 1

    print(f"{name}:")
    print(f"  Total: {total}")
    print(f"  Valid: {valid}")
    print(f"  Invalid: {invalid}")
    print()

print("===== VALIDATION COMPLETED =====")
