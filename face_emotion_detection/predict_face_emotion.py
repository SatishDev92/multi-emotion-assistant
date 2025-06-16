import cv2 as cv
import numpy as np
import time
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

def predict_face_emotion(model_path=r"D:\multi_model_project\face_emotion_detection\ff.h5"):
    # Load model and face detector
    model = load_model(model_path)
    classifier = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
    emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

    cap = cv.VideoCapture(0)
    print("🎥 Capturing face emotion... Window will auto-close in 5 seconds or press 'q' to confirm.")

    predicted_emotion = "neutral"  # default
    start_time = time.time()       # Start timer

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = classifier.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray_frame[y:y+h, x:x+w]
            roi_gray = cv.resize(roi_gray, (48, 48), interpolation=cv.INTER_AREA)
            roi_gray = cv.equalizeHist(roi_gray)

            if np.sum(roi_gray) != 0:
                roi = roi_gray.astype('float32') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                preds = model.predict(roi, verbose=0)[0]
                confidence = np.max(preds)
                label = emotion_labels[np.argmax(preds)]

                if confidence > 0.5:
                    predicted_emotion = label

                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.putText(frame, f"{label} ({confidence:.2f})", (x, y - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv.putText(frame, "No face", (20, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv.imshow("Face Emotion Detection", frame)

        # Auto-close after 5 seconds or if 'q' is pressed
        if time.time() - start_time > 15 or (cv.waitKey(1) & 0xFF == ord('q')):
            break

    cap.release()
    cv.destroyAllWindows()
    print(f"🎯 Detected face emotion: {predicted_emotion}")
    return predicted_emotion

if __name__ == "__main__":
    predict_face_emotion()
