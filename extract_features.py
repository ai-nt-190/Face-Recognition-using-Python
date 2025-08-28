import cv2
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

# Define the output directory and dataset path
dataset_dir = 'dataset'
output_dir = 'features'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize lists for features and labels
features = []
labels = []

# Desired face image size for consistency
image_size = (160, 160)  # Resize faces to 160x160

# Process all images in the dataset directory
for filename in os.listdir(dataset_dir):
    if filename.endswith('.jpg'):
        image_path = os.path.join(dataset_dir, filename)
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]

            # Resize the face image to a consistent size
            face_resized = cv2.resize(face, image_size)

            features.append(face_resized.flatten())  # Flatten the resized image
            label = filename.split('_')[0]  # Extract label (name) from the filename
            labels.append(label)

# Convert features and labels to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Encode the labels (names) into numerical values
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Save the features, labels, and label encoder
np.save(os.path.join(output_dir, 'features.npy'), features)
np.save(os.path.join(output_dir, 'labels.npy'), labels_encoded)
joblib.dump(label_encoder, os.path.join(output_dir, 'label_encoder.pkl'))

print("Features extracted and saved.")
