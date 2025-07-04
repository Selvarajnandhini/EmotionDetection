# EmotionDetection
## 😊 Emotion Detection with Python and Deep Learning

This project uses a webcam to detect a person’s **emotion** using a deep learning model trained on facial expressions.

## 📦 What’s in the Project
- `face.py` – main Python script that captures webcam input and displays emotions  
- `face.hdf5` – pre-trained model used for emotion classification  
- `requirements.txt` – Python libraries needed  
- `README.md` – this file  

## 🧠 Emotions It Can Detect

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral
## 🔧 Setup & Run

1. Clone the repo:
   ```bash
   git clone https://github.com/Selvarajnandhini/EmotionDetection.git
   cd EmotionDetection
### Install dependencies:
 pip install -r requirements.txt
## Run the application:
python face.py

->A webcam window will open with emotion labels overlaid.
->Press q to quit the app.
## 🧠 How It Works
Captures video from your webcam

Detects faces using OpenCV Haar Cascades

Preprocesses each detected face (resize, grayscale, normalize)

Feeds the face image into the face.hdf5 CNN model

Displays the predicted emotion label over the face
## ✅ requirements.txt
1.tensorflow
2.numpy
3.opencv-python

## 👤 Author
Selvaraj Nandhini
GitHub
