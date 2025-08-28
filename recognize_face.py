import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import joblib

# Load the trained SVM model and label encoder
model = joblib.load('model/face_recognition_model.pkl')
label_encoder = joblib.load('features/label_encoder.pkl')

# Initialize webcam and face detector
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize attendance log
attendance_log = pd.DataFrame(columns=["Name", "Timestamp"])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]

        # Resizing face to match the model's input
        face_resized = cv2.resize(face, (160, 160)).flatten().reshape(1, -1)

        # Predict the label using the SVM classifier
        label = model.predict(face_resized)
        name = label_encoder.inverse_transform([label])[0]

        # Draw rectangle and label on the image
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Log attendance
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_entry = pd.DataFrame({"Name": [name], "Timestamp": [timestamp]})
        attendance_log = pd.concat([attendance_log, new_entry], ignore_index=True)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the attendance log to an Excel file
attendance_log.to_excel("attendance.xlsx", index=False)
print("Attendance logged.")

cap.release()
cv2.destroyAllWindows()
