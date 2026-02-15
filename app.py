import os
import cv2
import numpy as np
import keras
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

# ---------------- CONFIG ----------------
keras.config.enable_unsafe_deserialization()

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "siamese_signature_model.keras")

UPLOAD_FOLDER = os.path.join(APP_DIR, "static", "uploads")
IMG_SIZE = 224

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- MODEL ----------------
def abs_diff(tensors):
    return K.abs(tensors[0] - tensors[1])

model = load_model(
    MODEL_PATH,
    safe_mode=False,
    custom_objects={"abs_diff": abs_diff}
)

# ---------------- FLASK ----------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------------- HELPERS ----------------
def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.repeat(img, 3, axis=-1)
    return np.expand_dims(img, axis=0)

def verify(ref_path, test_path, threshold=0.5):
    img1 = preprocess_image(ref_path)
    img2 = preprocess_image(test_path)

    score = model.predict([img1, img2])[0][0]
    result = "GENUINE" if score >= threshold else "FORGED"

    return float(score), result

# ---------------- ROUTES ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    score = None

    if request.method == "POST":
        ref = request.files.get("ref")
        test = request.files.get("test")

        if ref and test:
            ref_path = os.path.join(app.config["UPLOAD_FOLDER"], "ref.png")
            test_path = os.path.join(app.config["UPLOAD_FOLDER"], "test.png")

            ref.save(ref_path)
            test.save(test_path)

            score, result = verify(ref_path, test_path)

    return render_template(
        "index.html",
        result=result,
        score=score,
        identity=request.form.get("identity"),
        ref_img="uploads/ref.png",
        test_img="uploads/test.png"
        
    )


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
