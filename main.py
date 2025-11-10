import cv2
import numpy as np
import tensorflow as tf

# Model load karo
model = tf.keras.models.load_model("mask_detector_model.h5")

# Labels
labels = ["Mask", "No Mask"]

# Haarcascade face detector load
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Webcam start
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face detect
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]

        # Preprocess face ROI
        img = cv2.resize(face_roi, (128, 128))
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        # Prediction
        preds = model.predict(img)
        print(preds)
        class_idx = np.argmax(preds[0])
        label = labels[class_idx]

        # Draw rectangle + label
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, color, 2)

    # Show frame
    cv2.imshow("Mask Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()