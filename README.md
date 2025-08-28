Face Recognition using Python and OpenCV

Overview

This project is a Face Recognition System built using Python and OpenCV.
It allows users to capture images, extract facial features, train a recognition model, and recognize faces in real time using a webcam.

Technologies Used

- Python 3
- OpenCV
- NumPy
- Scikit-learn

Project Structure

face-recognition-python/
- │── scripts/
- │ ├── collect_images.py (Capture face dataset)
- │ ├── extract_features.py (Extract embeddings)
- │ ├── train_model.py (Train classifier)
- │ ├── recognize_face.py (Real-time recognition)
- │ ├── dataset/ (Preprocessed face dataset)
- │ ├── features/ (Extracted embeddings)
- │ └── model/ (Trained models)
- │── requirements.txt (Dependencies)
- │── README.md (Documentation)

Setup and Installation

- Clone the repository
- Install the required dependencies with the command:
  pip install -r requirements.txt

Usage

- Run collect_images.py to capture face images

- Run extract_features.py to extract embeddings

- Run train_model.py to train the recognition model

- Run recognize_face.py for real-time recognition

Features

- Real-time face detection and recognition

- Dataset creation from webcam

- Model training and testing

- Organized storage of images, features, and models
