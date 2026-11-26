# Bangladeshi Banknote Detection

A Flask-based web application for classifying images of Bangladeshi banknotes using pretrained **EfficientNetB0** models and **TensorFlow / Keras**. The repository includes a two-class genuine-versus-fake detector and a three-class variant that can also classify an image as **not a banknote**.

## Features

- Upload an image and view a classification result in the browser.
- **Three-class mode (recommended):** Genuine Banknote, Fake Banknote, or Not a Banknote.
- Capture a photo using a webcam or supported device camera in three-class mode.
- View per-class prediction percentages in three-class mode.
- **Two-class mode:** Genuine Banknote or Fake Banknote, with its fake-probability score.
- Submit images through the `POST /predict` endpoint.
- Run inference locally with the included trained `.keras` files; **retraining is not required** to try the website.

> **Important:** Predictions are estimates, not proof of a banknote's authenticity. The three-class model can still misclassify non-banknote images, and performance can vary with image quality, lighting, rotation, cropping, and previously unseen notes. Do not use this project as the sole basis for a financial decision.

## Technology stack

- **Language:** Python
- **Web framework:** Flask (HTML, CSS, and JavaScript are embedded in the Flask scripts)
- **Deep learning:** TensorFlow / Keras
- **Model family:** EfficientNetB0
- **Image processing:** Pillow, NumPy, and TensorFlow image utilities
- **Model format:** `.keras`

## Repository contents

```text
Banknote_Detection-main/
├── app.py                         # Two-class Flask application
├── app-LAPTOP-ADMVLS6C.py         # Three-class Flask application with webcam UI
├── best_efficientnetb0.keras      # Two-class model (1 sigmoid output)
├── best_efficientnetb0_3class.keras  # Three-class model (3 softmax outputs)
├── class_mapping.json            # real_notes / fake_notes
├── class_mapping_3class.json     # real_notes / fake_notes / not_banknotes
├── decision_threshold.json       # Saved two-class decision threshold
├── test_metrics.csv              # Recorded metrics for the two-class evaluation
├── prepare_dataset.py            # Prepare non-banknote images for training
├── check_dataset.py              # Audit non-banknote dataset
├── check_validation_dataset.py   # Validate dataset image files
├── train_banknote_validator.py   # Optional separate MobileNetV2 validator training
├── test_not_banknotes.py         # Evaluate the two-class model on non-banknote images
├── Banknote_Detection/           # Additional dataset-validation helper
├── .gitignore
└── README.md
```

**Note:** The dataset directories are not included in the provided project ZIP. They are unnecessary for running either pretrained Flask app, but are required for the dataset auditing and optional training scripts.

## Requirements

- **Python 3.11 (64-bit) recommended**, with `pip` and `venv` available.
- VS Code with the **Python** extension.
- Enough available memory to load the TensorFlow model.
- Internet access during the initial dependency installation.

### 1. Open the project in VS Code

1. Extract `Banknote_Detection-main.zip`.
2. Open **VS Code** → **File** → **Open Folder**.
3. Select the extracted `Banknote_Detection-main` folder (the folder that contains `app.py`).
4. Open **Terminal** → **New Terminal**.

All commands below assume the terminal is opened in the project root.

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run the following **in the current terminal**, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

In VS Code, select the `.venv` environment using **Ctrl+Shift+P** → **Python: Select Interpreter** (on macOS, use **Cmd+Shift+P**).

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install tensorflow flask pillow numpy
```

The project ZIP does not include a `requirements.txt`, so these are the packages imported by the Flask apps. Installing TensorFlow can take additional time and requires a Python version supported by the TensorFlow release installed.

### 4. Run the website

**Recommended: three-class detection (including “Not a Banknote”):**

```bash
python app-LAPTOP-ADMVLS6C.py
```

The script loads `best_efficientnetb0_3class.keras` and `class_mapping_3class.json` from the same directory. When the server starts, open:

**http://127.0.0.1:5000/**

Upload an image, or use **Start Camera** → **Capture Photo** → **Check Banknote**. Camera access depends on browser permissions and device support; localhost generally supports camera access.

**Alternative: two-class detection only:**

First stop the running server with **Ctrl+C**, then run:

```bash
python app.py
```

This script loads `best_efficientnetb0.keras` and labels predictions using the saved threshold **0.5916581749916077**. The two-class model has **no “Not a Banknote” output**; it will attempt to label unrelated images as genuine or fake, so use the three-class version when that distinction matters.

### 5. Stop the server

Press **Ctrl+C** in the VS Code terminal.

## How predictions work

### Three-class application

1. The browser sends an image to `POST /predict`.
2. Flask checks that an image was supplied and decodes it.
3. TensorFlow converts it to three-channel RGB and resizes it to **224 × 224**.
4. `best_efficientnetb0_3class.keras` produces three class probabilities.
5. The highest-scoring class is mapped using `class_mapping_3class.json`:

   | Model index | Class name | Displayed result |
   | --- | --- | --- |
   | `0` | `real_notes` | Genuine Banknote |
   | `1` | `fake_notes` | Fake Banknote |
   | `2` | `not_banknotes` | Not a Banknote |

6. Flask returns the result and per-class prediction percentages as JSON.

### Two-class application

The two-class app converts images to RGB, resizes them to **224 × 224** using LANCZOS resampling, keeps pixel values on the **0–255** scale, and uses a sigmoid output interpreted as fake probability. It predicts **Fake Banknote** when that probability is at least `0.5916581749916077`; otherwise it predicts **Genuine Banknote**.

## API example

Both applications accept a multipart form upload with field name `image`:

```bash
curl -X POST -F "image=@sample.jpg" http://127.0.0.1:5000/predict
```

On Windows PowerShell, use `curl.exe` instead of `curl` if your shell treats `curl` as a different command.

Example **three-class** response shape (illustrative only; percentages vary by image):

```json
{
  "result": "Genuine Banknote",
  "predicted_class": "real_notes",
  "probabilities": {
    "real_notes": 93.6,
    "fake_notes": 4.9,
    "not_banknotes": 1.5
  }
}
```

Example **two-class** response shape (illustrative only):

```json
{
  "result": "Fake Banknote",
  "fake_probability": 0.8732,
  "fake_probability_percent": 87.32
}
```

## Existing evaluation results

The repository includes `test_metrics.csv`, which records the following metrics for the **two-class** genuine/fake classifier:

| Metric | Recorded value |
| --- | ---: |
| Accuracy | 98.64% |
| Fake-class precision | 98.67% |
| Fake-class recall | 98.11% |
| F1 score | 98.39% |
| ROC AUC | 0.9935 |
| Decision threshold | 0.5917 |

These are **previously recorded evaluation values**, not a fresh local test. They must **not** be presented as verified three-class performance. The included ZIP does not contain the original evaluation dataset or enough information to independently reproduce these metrics.

## Optional training and dataset utilities

The following utilities are **not needed to start the website**:

- `prepare_dataset.py`: asks you to select a source category-image directory via Tkinter, then copies valid images to `dataset/not_banknotes/<category>/`.
- `check_dataset.py`: audits the `dataset/not_banknotes/` folder; execute it from the project root after adding the dataset.
- `check_validation_dataset.py`: checks images in `dataset/banknote_validation/real_notes/`, `dataset/banknote_validation/fake_notes/`, and `dataset/not_banknotes/`.
- `train_banknote_validator.py`: trains a **separate MobileNetV2-based binary banknote validator**, saves `banknote_validator.keras`, and requires all three dataset folders and expected real/fake filenames matching `note_<id>_<number>.*`. It **does not automatically replace** either EfficientNetB0 model in the web applications.
- `test_not_banknotes.py`: tests the **two-class** model on images in `dataset/not_banknotes/`; it does not measure the three-class app's rejection accuracy.

Suggested dataset structure if you plan to use these utilities:

```text
dataset/
├── banknote_validation/
│   ├── real_notes/
│   └── fake_notes/
└── not_banknotes/
    ├── category_1/
    └── category_2/
```

The dataset is ignored by the existing `.gitignore`.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| `python` / `py` not found | Install Python 3.11 (64-bit), then reopen the terminal. |
| `No module named flask` / `tensorflow` | Activate `.venv` and rerun the installation command. |
| PowerShell refuses to activate `.venv` | Use the process-scoped `Set-ExecutionPolicy` command shown above. |
| `Model file not found` | Open the project root and keep both `.keras` files beside the two Flask application scripts. |
| TensorFlow installation fails | Check your installed Python version, system architecture, and TensorFlow platform support. |
| Port `5000` is already in use | Stop the other server or adjust the `app.run(...)` port and open the matching URL. |
| Browser cannot access camera | Allow camera permission, use localhost with a supported browser, or upload an image instead. |
| Incorrect prediction | Try a clear, uncropped image with good lighting; model outputs are estimates and may still be wrong. |

## Commit this README to GitHub

If this project is already a Git repository and you only want to update documentation, run these commands **from the repository root**:

```bash
git status
git add README.md
git commit -m "docs: expand setup, usage, and model documentation"
git push
```

If `git push` reports no upstream branch, use `git push -u origin <your-branch-name>` with the branch name shown by `git branch --show-current`. If you downloaded a ZIP instead of cloning the repository, clone your GitHub repository first, copy this `README.md` into its root, and run the commands from that cloned folder.

## Project status

This repository contains the application code and pretrained model files. The archive was checked for Python syntax and model-file structure, but its running Flask applications and prediction accuracy have **not** been independently verified in a live TensorFlow environment.
