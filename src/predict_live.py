import cv2
import numpy as np
from keras.models import load_model
import pyttsx3
import threading
import time

# Load the trained model
model = load_model(r'C:\Users\ADITYA VERMA\C++ Projects\SparshVaani\models\sparshvaani_model.h5')

# Define labels
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ['del', 'nothing', 'space']

# Initialize text-to-speech
engine = pyttsx3.init()

#Define the speak function
def speak(text):
    threading.Thread(target=lambda: engine.say(text) or engine.runAndWait()).start()

# Start webcam
cap = cv2.VideoCapture(0)

prev_prediction = ''
word = ''
last_spoken = time.time()
confidence_threshold = 0.75 #Only accept predictions above this confidence 

#Frame processing interval (every nth frame for performance)
frame_interval = 3
frame_count = 0

# Set the window to be resizable
cv2.namedWindow("SparshVaani - Live", cv2.WINDOW_NORMAL)

# Function to wrap text within a given width
def wrap_text(text, font, max_width):
    lines = []
    current_line = ' '
    
    for char in text:
        test_line = current_line + char
        (text_width, _), _ = cv2.getTextSize(test_line, font, 0.6, 1)
        
        if text_width <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = char
    
    if current_line:
     lines.append(current_line)  
    return lines

while True:
    ret, frame = cap.read()
    if not ret:
        break

   # Skip frames to improve performance
    frame_count += 1
    if frame_count % frame_interval != 0:
        continue  # Skip this frame if not divisible by frame_interval

    # Flip and crop the frame
    # frame = cv2.resize(frame, (640, 480))
    frame = cv2.flip(frame, 1)

    #Define ROI Box Position (left-center)
    x1, y1, x2, y2 = 100, frame.shape[0]//3, 324, frame.shape[0]//3 +224
    roi = frame[y1:y2, x1:x2]

    #Draw ROI rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    #Preprocess ROI
    roi = cv2.resize(roi, (64, 64))
    roi = roi.astype('float') / 255.0
    roi = np.expand_dims(roi, axis=0)

    #Predict
    preds = model.predict(roi)[0]
    if np.max(preds) > confidence_threshold:
        prediction = labels[np.argmax(preds)]
    else:
        prediction = 'nothing'
    
    #Process new prediction
    if prediction != prev_prediction:
        if prediction == 'space':
            word += ' '
        elif prediction == 'del':
            word = word[:-1]
        elif prediction != 'nothing':
            word += prediction
            speak(prediction) #Use speak function to pronounce the letter
        prev_prediction = prediction 

    # Display prediction and current word
    font = cv2.FONT_HERSHEY_SIMPLEX

    #Top right: letters --> Display the current predicted letter
    cv2. putText(frame, f"letters --> {prediction}", (x2 + 20, 50), font, 0.6, (0, 255, 0), 1)

    #Right middle: word --> Display the current word with wrapping
    cv2.putText(frame, "Word -->", (x2 + 20, 100), font, 0.6, (255, 0, 0), 1) 
    text_lines = wrap_text(word, font, frame.shape[1] - x2 - 40)   # Wrap text if it exceeds the screen width
    y_position = 130

    #Start at the middle of the right side
    for line in text_lines:
        cv2.putText(frame, line, (x2 + 20, y_position), font, 0.6, (255, 0, 0), 1) 
        y_position += 30  # Adjust line spacing for smaller text

    #Show frame(resize window if needed)
    frame_width = 1920
    frame_height = 1080
    cv2.resizeWindow("SparshVaani - Live", frame_width, frame_height)
    cv2.imshow("SparshVaani - Live", frame)

    #Exit on presing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    #time.sleep(0.03)

#Release resources    
cap.release()
cv2.destroyAllWindows()