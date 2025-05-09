
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pyttsx3

model = load_model('models/sparshvaani_model.h5')
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ ")

engine = pyttsx3.init()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    roi = frame[100:300, 100:300]
    img = cv2.resize(roi, (64, 64))
    img = img / 255.0
    img = img.reshape(1, 64, 64, 3)

    prediction = model.predict(img)
    index = np.argmax(prediction)
    letter = labels[index]

    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
    cv2.putText(frame, letter, (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

    cv2.imshow("SparshVaani", frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        engine.say(letter)
        engine.runAndWait()
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
