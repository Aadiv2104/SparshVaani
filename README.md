g
# SparshVaani - Sign Language to Speech Converter

This project uses computer vision and deep learning to convert sign language gestures into spoken words.

## Structure
- `dataset/train/`: Place your sign language images here.
- `models/`: Stores the trained model.
- `src/`: Contains source code for training and real-time prediction.

## Usage
1. Train the model using `train_model.py`.
2. Run `sparshvaani_realtime.py` to start gesture recognition via webcam.

Press 's' to speak the recognized letter and 'q' to quit.

## Requirements
- TensorFlow
- OpenCV
- pyttsx3
