import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained face detection model
faceDetect=cv2.CascadeClassifier('Data/haarcascade_frontalface_default.xml')
# Load the pre-trained emotion detection model
model=load_model('Data/model_file_30epochs.h5')
# Create a dictionary to map the emotion labels
labels_dict={0:'Angry',1:'Disgust', 2:'Fear', 3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}
# Start the video capture from the default camera
video=cv2.VideoCapture(0)

while True:
    # Capture a frame from the video feed
    ret,frame=video.read()
    # Convert the frame to grayscaleq
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale frame
    faces= faceDetect.detectMultiScale(gray, 1.3, 3)
    # Loop through each detected face and predict the emotion
    for x,y,w,h in faces:	
        # Extract the face region from the grayscale frame
        sub_face_img=gray[y:y+h, x:x+w]
        # Resize the face region to 48x48 pixels
        resized=cv2.resize(sub_face_img,(48,48))
        # Normalize the face region pixel values to the range [0, 1]
        normalize=resized/255.0
        reshaped=np.reshape(normalize, (1, 48, 48, 1))
        result=model.predict(reshaped)
        # Get the index of the emotion with the highest predicted probability
        label=np.argmax(result, axis=1)[0]
        # print label
        print(label)
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
        # Draw the emotion label above the face
        cv2.putText(frame, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    # Display the resulting frame    
    cv2.imshow("Face Emotion Detection App",frame)
    # Exit the loop if the 'q' key is pressed
    k=cv2.waitKey(1)
    if k==ord('q'):
        break
# Release the video capture and close all windows
video.release()
cv2.destroyAllWindows()