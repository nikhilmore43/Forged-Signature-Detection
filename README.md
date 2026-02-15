# Forged Signature Detection using Siamese Network

This project verifies whether a signature is genuine or forged
using a Siamese neural network and a Flask web application.

# Dataset

This project uses the CEDAR Off-Line Handwritten Signature Dataset.

Due to size and licensing constraints, the full dataset is not included in this repository.
Only a limited number of sample images are provided for demonstration purposes.

Dataset reference:
CEDAR Signature Dataset (University at Buffalo)


## Features
- Siamese network for signature comparison
- Structural similarity (SSIM) scoring
- Flask-based web interface

## Folder Structure
- `train_siamese.py` – model training
- `verify_signature.py` – verification logic
- `app.py` – Flask app
- `dataset/` – train and test data
- `static/` – CSS and uploads
- `templates/` – HTML files

## How to Run 
1. Install dependencies
2. Run `train_siamese.py` 
3. Start app using `python app.py`
4. In deployment phase reference image to be provided as an input should be original!

(.venv should be active while running)