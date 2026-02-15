import os
import cv2
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

def abs_diff(tensors):
    return K.abs(tensors[0] - tensors[1])



# allow lambda loading
keras.config.enable_unsafe_deserialization()

MODEL_PATH = "siamese_signature_model.keras"
IMG_SIZE = 224

# load model
model = load_model(
    MODEL_PATH,
    safe_mode=False,
    custom_objects={"abs_diff": abs_diff}
)



from PIL import Image

def preprocess(img_path):
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("OpenCV failed")
    except:
        img = Image.open(img_path).convert("L")
        img = np.array(img)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.repeat(img, 3, axis=-1)
    return img


def verify(sig1_path, sig2_path, threshold=0.5):
    img1 = preprocess(sig1_path)
    img2 = preprocess(sig2_path)

    img1 = np.expand_dims(img1, axis=0)
    img2 = np.expand_dims(img2, axis=0)

    score = model.predict([img1, img2])[0][0]

    result = "GENUINE" if score >= threshold else "FORGED"
    return score, result


# -------- TEST RUN --------
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    s1 = os.path.join(BASE_DIR, "sample1.png")
    s2 = os.path.join(BASE_DIR, "sample2.png")

    print("Loading:", s1)
    print("Loading:", s2)

    score, result = verify(s1, s2)
    print("Similarity score:", score)
    print("Result:", result)