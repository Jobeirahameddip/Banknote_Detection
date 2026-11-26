
import re
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

# =====================================
# 1. CONFIGURATION
# =====================================

BASE = Path(__file__).resolve().parent

REAL_DIR = BASE / "dataset" / "banknote_validation" / "real_notes"
FAKE_DIR = BASE / "dataset" / "banknote_validation" / "fake_notes"
NON_DIR = BASE / "dataset" / "not_banknotes"

IMG_SIZE = (160, 160)
BATCH_SIZE = 4
EPOCHS = 5
SEED = 42

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Limit TensorFlow CPU threads
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(1)


# =====================================
# 2. CHECK FOLDERS
# =====================================

for folder in [REAL_DIR, FAKE_DIR, NON_DIR]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

print("Folders verified!", flush=True)


# =====================================
# 3. COLLECT BANKNOTE GROUPS
# =====================================

def collect_banknotes(folder, label, class_name):
    groups = {}

    for file in folder.rglob("*"):
        if not file.is_file() or file.suffix.lower() not in EXTENSIONS:
            continue

        match = re.match(r"^note_(\d+)_\d+$", file.stem)

        if not match:
            raise ValueError(f"Unexpected filename: {file.name}")

        note_id = match.group(1)
        groups.setdefault(note_id, []).append(str(file))

    print(f"{class_name} groups:", len(groups), flush=True)

    return groups


real_groups = collect_banknotes(REAL_DIR, 1, "Real")
fake_groups = collect_banknotes(FAKE_DIR, 1, "Fake")


# =====================================
# 4. COLLECT NON-BANKNOTE GROUPS
# =====================================

non_groups = {}

for file in NON_DIR.rglob("*"):
    if not file.is_file() or file.suffix.lower() not in EXTENSIONS:
        continue

    relative = file.relative_to(NON_DIR)

    if len(relative.parts) < 2:
        raise ValueError(
            "Non-banknote images must be inside category folders: "
            + str(file)
        )

    category = relative.parts[0]
    non_groups.setdefault(category, []).append(str(file))

print("Non-banknote category groups:", len(non_groups), flush=True)


# =====================================
# 5. GROUP-BASED SPLIT
# =====================================

def split_groups(groups, label, name):
    keys = sorted(groups.keys())
    random.shuffle(keys)

    if len(keys) < 2:
        raise ValueError(f"Not enough groups for {name}")

    split_index = int(len(keys) * 0.80)
    split_index = max(1, min(split_index, len(keys) - 1))

    train_keys = keys[:split_index]
    val_keys = keys[split_index:]

    train_records = []
    val_records = []

    for key in train_keys:
        for path in groups[key]:
            train_records.append((path, label))

    for key in val_keys:
        for path in groups[key]:
            val_records.append((path, label))

    print(
        f"{name}: train groups={len(train_keys)}, "
        f"validation groups={len(val_keys)}",
        flush=True
    )

    return train_records, val_records


train_records = []
val_records = []

for groups, label, name in [
    (real_groups, 1, "Real"),
    (fake_groups, 1, "Fake"),
    (non_groups, 0, "Non-banknote")
]:
    train_part, val_part = split_groups(groups, label, name)
    train_records.extend(train_part)
    val_records.extend(val_part)

random.shuffle(train_records)
random.shuffle(val_records)

print("\nTraining images:", len(train_records), flush=True)
print("Validation images:", len(val_records), flush=True)


# =====================================
# 6. CREATE TF DATASET
# =====================================

def load_image(path, label):
    image = tf.io.read_file(path)

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)

    return image, label


def make_dataset(records, training=False):
    paths = [record[0] for record in records]
    labels = np.array(
        [record[1] for record in records],
        dtype=np.float32
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            500,
            seed=SEED
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=1
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(1)

    return dataset


train_ds = make_dataset(train_records, training=True)
val_ds = make_dataset(val_records)


# =====================================
# 7. BUILD MOBILENETV2 MODEL
# =====================================

print("\nLoading MobileNetV2...", flush=True)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(160, 160, 3))

x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(
    1,
    activation="sigmoid"
)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# =====================================
# 8. TRAIN MODEL
# =====================================

print("\nStarting training...", flush=True)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)


# =====================================
# 9. SAVE MODEL
# =====================================

output_path = BASE / "banknote_validator.keras"

model.save(output_path)

print("\nTraining completed!", flush=True)
print("Saved model:", output_path, flush=True)
