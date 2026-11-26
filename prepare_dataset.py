
from pathlib import Path
from tkinter import Tk, filedialog
from PIL import Image, UnidentifiedImageError
import shutil

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "dataset" / "not_banknotes"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Select source dataset folder
root = Tk()
root.withdraw()

print("Select the 101_ObjectCategories folder...")

selected = filedialog.askdirectory(
    title="Select 101_ObjectCategories"
)

if not selected:
    print("No folder selected!")
    raise SystemExit

SOURCE_DIR = Path(selected)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

total_images = 0
total_categories = 0

for category in SOURCE_DIR.iterdir():

    if not category.is_dir() or category.name.startswith("__"):
        continue

    destination_folder = OUTPUT_DIR / category.name
    destination_folder.mkdir(parents=True, exist_ok=True)

    copied = 0

    for file in category.iterdir():

        if not file.is_file():
            continue

        if file.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        try:
            # Validate image
            with Image.open(file) as img:
                img.verify()

            destination = destination_folder / file.name

            if destination.exists():
                continue

            shutil.copy2(file, destination)

            copied += 1
            total_images += 1

        except (UnidentifiedImageError, OSError):
            continue

    if copied:
        total_categories += 1
        print(f"{category.name}: {copied} images")

print("\nDATASET PREPARATION COMPLETED")
print("Categories:", total_categories)
print("Images:", total_images)
print("Saved at:", OUTPUT_DIR)
