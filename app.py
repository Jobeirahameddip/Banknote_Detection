
from flask import Flask, request, jsonify, render_template_string
from tensorflow.keras.models import load_model
from PIL import Image, UnidentifiedImageError
import numpy as np
import os

app = Flask(__name__)

# =====================================
# PROJECT SETTINGS
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "best_efficientnetb0.keras"
)

THRESHOLD = 0.5916581749916077
IMAGE_SIZE = (224, 224)

# Same LANCZOS resizing method as Kaggle
_RESAMPLING = getattr(Image, "Resampling", Image).LANCZOS


# =====================================
# LOAD MODEL
# =====================================

print("Loading EfficientNetB0 model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model file not found: " + MODEL_PATH
    )

model = load_model(MODEL_PATH, compile=False)

print("Model loaded successfully!")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)


# =====================================
# WEBSITE HTML
# =====================================

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Bangladeshi Banknote Detection</title>

    <meta name="viewport" content="width=device-width, initial-scale=1">

    <style>
        body {
            font-family: Arial, sans-serif;
            background: #eef3f1;
            text-align: center;
            padding: 25px;
        }

        .container {
            background: white;
            max-width: 500px;
            margin: 30px auto;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 4px 15px #0002;
        }

        h1 {
            color: #087f5b;
        }

        input {
            margin: 20px;
            max-width: 100%;
        }

        button {
            background: #087f5b;
            color: white;
            border: none;
            padding: 13px 25px;
            border-radius: 7px;
            cursor: pointer;
            font-size: 16px;
        }

        button:disabled {
            background: gray;
        }

        #preview {
            display: none;
            max-width: 100%;
            max-height: 250px;
            margin: 15px auto;
            border-radius: 8px;
        }

        #result {
            margin-top: 20px;
            font-size: 20px;
            font-weight: bold;
        }

        #details {
            margin-top: 10px;
            color: #555;
            font-size: 14px;
        }
    </style>
</head>

<body>

<div class="container">

    <h1>Bangladeshi Banknote Detection</h1>

    <p>Upload or take a photo of a Bangladeshi banknote.</p>

    <input
        type="file"
        id="image"
        accept="image/*"
        capture="environment"
    >

    <br>

    <img id="preview">

    <br>

    <button id="checkButton" onclick="predict()">
        Check Banknote
    </button>

    <div id="result"></div>
    <div id="details"></div>

</div>

<script>

const imageInput = document.getElementById("image");

imageInput.addEventListener("change", function() {

    const file = this.files[0];

    if (file) {

        const preview = document.getElementById("preview");

        preview.src = URL.createObjectURL(file);
        preview.style.display = "block";

        document.getElementById("result").innerText = "";
        document.getElementById("details").innerText = "";
    }

});


async function predict() {

    const file = imageInput.files[0];

    const result = document.getElementById("result");
    const details = document.getElementById("details");
    const button = document.getElementById("checkButton");

    if (!file) {
        result.innerText = "Please select an image first.";
        return;
    }

    const formData = new FormData();

    formData.append("image", file);

    result.innerText = "Checking image...";
    details.innerText = "";

    button.disabled = true;

    try {

        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Prediction failed");
        }

        result.innerText = data.result;

        details.innerText =
            "Fake probability: " +
            data.fake_probability_percent + "%";

    }

    catch (error) {

        result.innerText = "Error: " + error.message;

    }

    finally {

        button.disabled = false;

    }

}

</script>

</body>
</html>
"""


# =====================================
# HOME PAGE
# =====================================

@app.route("/")
def home():

    return render_template_string(HTML)


# =====================================
# IMAGE PREDICTION
# =====================================

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:

        return jsonify({
            "error": "No image uploaded"
        }), 400

    file = request.files["image"]

    if not file.filename:

        return jsonify({
            "error": "Please select an image"
        }), 400

    try:

        # Read uploaded image
        image = Image.open(file)

        # Convert image to RGB
        image = image.convert("RGB")

        # Resize using the same LANCZOS method as Kaggle
        image = image.resize(
            IMAGE_SIZE,
            resample=_RESAMPLING
        )

        # Convert image to float32 NumPy array
        image_array = np.asarray(
            image,
            dtype=np.float32
        )

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # IMPORTANT:
        # Keep pixel values between 0 and 255.
        # EfficientNetB0 has its own Rescaling layer.

        # Model prediction
        prediction = model.predict(
            image_array,
            verbose=0
        )

        if prediction.shape != (1, 1):

            raise ValueError(
                "Unexpected model output shape: "
                + str(prediction.shape)
            )

        fake_probability = float(prediction[0][0])

        if not np.isfinite(fake_probability):

            raise ValueError(
                "Model returned an invalid prediction"
            )

        if not 0 <= fake_probability <= 1:

            raise ValueError(
                "Model output is outside probability range"
            )

        # Classification using validation threshold
        if fake_probability >= THRESHOLD:

            result = "Fake Banknote"

        else:

            result = "Genuine Banknote"

        # Terminal output
        print("\\n----- PREDICTION DETAILS -----")

        print("Raw model output:", prediction)

        print("Fake probability:", fake_probability)

        print("Threshold:", THRESHOLD)

        print("Final result:", result)

        print("------------------------------\\n")

        return jsonify({

            "result": result,

            "fake_probability": round(
                fake_probability, 4
            ),

            "fake_probability_percent": round(
                fake_probability * 100, 2
            )

        })

    except (UnidentifiedImageError, OSError):

        return jsonify({
            "error": "The uploaded file is not a valid image."
        }), 400

    except Exception as e:

        print("Prediction error:", str(e))

        return jsonify({
            "error": str(e)
        }), 400


# =====================================
# RUN WEBSITE
# =====================================

if __name__ == "__main__":

    app.run(
        debug=False
    )
