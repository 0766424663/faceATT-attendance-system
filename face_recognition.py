import cv2
import numpy as np

# Step 1: Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Load OpenCV's built-in face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Check if the face cascade loaded properly
if face_cascade.empty():
    print("Error: Could not load face cascade.")
    exit()

print("Press 's' to save the face, and 'q' to quit")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Face Registration', frame)

    # Save face when 's' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and len(faces) > 0:
        for (x, y, w, h) in faces:
            face_image = frame[y:y+h, x:x+w]
            cv2.imwrite("registered_face.jpg", face_image)
            print("Face saved as 'registered_face.jpg'!")
        break  # Exit after saving a face

    # Quit with 'q'
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Load the saved face image
saved_face = cv2.imread("registered_face.jpg", cv2.IMREAD_GRAYSCALE)

if saved_face is None:
    print("Error: No registered face found. Please save a face first.")
    exit()

# Re-initialize webcam for recognition
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not re-open webcam.")
    exit()

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture image.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        current_face = gray_frame[y:y+h, x:x+w]
        current_face = cv2.resize(current_face, (saved_face.shape[1], saved_face.shape[0]))

        mse = np.mean((saved_face - current_face) ** 2)

        if mse < 1000:  # Lower is more similar; adjust if needed
            cv2.putText(frame, "Attendance Marked", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            print("Attendance marked!")
        else:
            cv2.putText(frame, "Face Not Recognized", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
