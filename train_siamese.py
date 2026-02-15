import os

BASE_DIR = r"C:\Users\moren\Documents\sign_data"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

print("Train path:", TRAIN_DIR)
print("Test path:", TEST_DIR)

print("Train folders:", os.listdir(TRAIN_DIR))
print("Test folders:", os.listdir(TEST_DIR))

import cv2
import numpy as np

IMG_SIZE = 224  # MobileNetV2 input size

def load_preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    return img

def get_genuine_folders(base_dir):
    return [f for f in os.listdir(base_dir) if not f.endswith("_forg")]

def get_forged_folders(base_dir):
    return [f for f in os.listdir(base_dir) if f.endswith("_forg")]

genuine_folders = get_genuine_folders(TRAIN_DIR)
forged_folders = get_forged_folders(TRAIN_DIR)

print("Genuine folders sample:", genuine_folders[:5])
print("Forged folders sample:", forged_folders[:5])

import random

def to_rgb(img):
    return np.repeat(img, 3, axis=-1)

def create_pairs(base_dir):
    pairs = []
    labels = []

    genuine_folders = get_genuine_folders(base_dir)

    for folder in genuine_folders:
        genuine_path = os.path.join(base_dir, folder)
        forged_path = os.path.join(base_dir, folder + "_forg")

        genuine_images = os.listdir(genuine_path)
        forged_images = os.listdir(forged_path)

        # Positive pairs (genuine–genuine)
        for i in range(len(genuine_images) - 1):
            img1 = to_rgb(load_preprocess_image(os.path.join(genuine_path, genuine_images[i])))
            img2 = to_rgb(load_preprocess_image(os.path.join(genuine_path, genuine_images[i + 1])))

            pairs.append([img1, img2])
            labels.append(1)

        # Negative pairs (genuine–forged)
        for i in range(min(len(genuine_images), len(forged_images))):
           img1 = to_rgb(load_preprocess_image(os.path.join(genuine_path, genuine_images[i])))
           img2 = to_rgb(load_preprocess_image(os.path.join(forged_path, forged_images[i])))

           
           pairs.append([img1, img2])
           labels.append(0)

    return np.array(pairs), np.array(labels)

train_pairs, train_labels = create_pairs(TRAIN_DIR)

print("Total pairs:", train_pairs.shape)
print("Total labels:", train_labels.shape)


import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Lambda, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

def abs_diff(tensors):
    return K.abs(tensors[0] - tensors[1])


# Base CNN
base_cnn = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_cnn.trainable = False

def embedding_network():
    inp = Input(shape=(224, 224, 3))
    x = base_cnn(inp)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    return Model(inp, x)

embed_model = embedding_network()

# Siamese inputs
input_1 = Input(shape=(224, 224, 3))
input_2 = Input(shape=(224, 224, 3))

feat_1 = embed_model(input_1)
feat_2 = embed_model(input_2)

# Distance layer
from tensorflow.keras import backend as K

distance = Lambda(
    abs_diff,
    output_shape=lambda shapes: shapes[0],
    name="abs_diff"
)([feat_1, feat_2])


output = Dense(1, activation="sigmoid")(distance)

siamese_model = Model([input_1, input_2], output)

siamese_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

siamese_model.summary()

# Split pairs into two inputs
X1 = train_pairs[:, 0]
X2 = train_pairs[:, 1]

if __name__ == "__main__":

    history = siamese_model.fit(
        [X1, X2],
        train_labels,
        batch_size=16,
        epochs=10,
        validation_split=0.2
    )

    # Fine-tuning
    base_cnn.trainable = True

    for layer in base_cnn.layers[:-30]:
        layer.trainable = False

    siamese_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    history_ft = siamese_model.fit(
        [X1, X2],
        train_labels,
        batch_size=16,
        epochs=10,
        validation_split=0.2
    )

    # -------- TEST EVALUATION --------
    test_pairs, test_labels = create_pairs(TEST_DIR)

    X1_test = test_pairs[:, 0]
    X2_test = test_pairs[:, 1]

    test_loss, test_acc = siamese_model.evaluate(
        [X1_test, X2_test],
        test_labels,
        verbose=1
    )

    print("Test Accuracy:", test_acc)
    print("Test Loss:", test_loss)

    # -------- SAVE MODEL --------
    siamese_model.save("siamese_signature_model.keras")
    print("Model saved successfully")
