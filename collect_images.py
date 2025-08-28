import cv2
import os

# Create directory to store collected images
output_dir = 'dataset'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize the webcam and the face detector
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Collect images from webcam
user_name = input("Enter the name of the person: ")
num_images = 200  # Collect 200 images for each person

count = 0
while count < num_images:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        face_filename = os.path.join(output_dir, f"{user_name}_{count}.jpg")
        cv2.imwrite(face_filename, face)
        count += 1

    # Display the frame with detected face
    cv2.imshow("Collecting Faces", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"{num_images} images collected for {user_name}")
