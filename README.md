# Face Recognition Using MediaPipe & LBPH

This project shows how face detection and face recognition can be done without machine learning models.  
It uses **MediaPipe** for detecting faces and **OpenCV LBPH** for recognizing which person the face belongs to.

The system has three main parts: capturing images, training the recognizer, and running real-time prediction.

---

## How It Works

### 1. Capture Faces
You first run a script that opens your webcam and asks you to enter a person's name.  
The system detects the face using MediaPipe and saves about 50 cropped face images.  
You do this for at least **two different people**.  
These images become the dataset used for training.

### 2. Train the Model
After capturing images, you run the training script.  
It reads all the saved face images, labels them by name, and trains the LBPH recognizer.  
The trained model and label map are saved in the `models` folder.

### 3. Real-Time Recognition
Finally, you run the recognition script.  
It uses MediaPipe to detect faces in the webcam and LBPH to identify each detected face.  
The system shows the person’s name on the screen in real time.

---

## How To Run

### Install Required Packages
```bash
pip install opencv-contrib-python mediapipe
