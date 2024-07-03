import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained face detection model
FACE_DETECTION_MODEL = 'Data/haarcascade_frontalface_default.xml'
face_detect = cv2.CascadeClassifier(FACE_DETECTION_MODEL)

# Load the pre-trained emotion detection model
EMOTION_DETECTION_MODEL = 'Data/model_file_30epochs.h5'
model = load_model(EMOTION_DETECTION_MODEL)

# Create a dictionary to map the emotion labels
EMOTION_LABELS = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Surprise'
}

def detect_faces(gray_frame):
    """Detect faces in the grayscale frame"""
    return face_detect.detectMultiScale(gray_frame, 1.3, 3)

def extract_face_region(gray_frame, x, y, w, h):
    """Extract the face region from the grayscale frame"""
    return gray_frame[y:y+h, x:x+w]

def preprocess_face_region(face_region):
    """Resize and normalize the face region"""
    resized = cv2.resize(face_region, (48, 48))
    normalized = resized / 255.0
    return normalized

def predict_emotion(face_region):
    """Predict the emotion of the face region"""
    reshaped = np.reshape(face_region, (1, 48, 48, 1))
    result = model.predict(reshaped)
    label = np.argmax(result, axis=1)[0]
    return label

def draw_face_box(frame, x, y, w, h, label):
    """Draw a rectangle around the detected face and display the emotion label"""
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
    cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
    cv2.putText(frame, EMOTION_LABELS[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

def main():
    video = cv2.VideoCapture('video/9msyhy.mp4')
    while True:
        ret, frame = video.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray)
        for x, y, w, h in faces:
            face_region = extract_face_region(gray, x, y, w, h)
            face_region = preprocess_face_region(face_region)
            label = predict_emotion(face_region)
            draw_face_box(frame, x, y, w, h, label)
        cv2.imshow("Face Emotion Detection App", frame)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()