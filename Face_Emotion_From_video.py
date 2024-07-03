import cv2
from deepface import DeepFace
import numpy as np

# Load pre-trained emotion detection model
emotion_model = DeepFace.build_model('Emotion')

# Load face cascade
face_cascade = cv2.CascadeClassifier('Data/haarcascade_frontalface_default.xml')

# Load video
video_capture = cv2.VideoCapture('video/g297mg.mp4')

# Emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Check if the video has ended
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # For each detected face, predict emotion
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        # Convert face image to grayscale
        face_img_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        # Resize face image to match model input size
        face_img_resized = cv2.resize(face_img_gray, (48, 48))

        # Predict emotion on grayscale face image
        emotion_preds = emotion_model.predict(np.expand_dims(face_img_resized, axis=0))[0]

        # Get the dominant emotion label
        dominant_emotion_label = emotion_labels[np.argmax(emotion_preds)]

        # Draw rectangle around the face and label the emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, dominant_emotion_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
video_capture.release()
cv2.destroyAllWindows()
