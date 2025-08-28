import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Load the extracted features and labels
features = np.load('features/features.npy')
labels = np.load('features/labels.npy')

# Check unique labels to debug
print("Unique labels in the dataset:", np.unique(labels))  # Add this line to check labels

# Initialize the SVM classifier
svm_model = SVC(kernel='linear', probability=True)

# Fit the model to the features and labels
svm_model.fit(features, labels)

# Save the trained model
joblib.dump(svm_model, 'model/face_recognition_model.pkl')

print("Model trained and saved successfully.")
