import cv2
import mediapipe as mp
import json
import os

mp_face_detection = mp.solutions.face_detection

# Load model and labels
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("models/lbph_model.yml")

with open("models/label_map.json", "r") as f:
    label_map = json.load(f)
    id_to_name = {int(k): v for k, v in label_map.items()}

cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x1 = int(bboxC.xmin * w)
                y1 = int(bboxC.ymin * h)
                x2 = int((bboxC.xmin + bboxC.width) * w)
                y2 = int((bboxC.ymin + bboxC.height) * h)

                face_img = frame[y1:y2, x1:x2]
                if face_img.size == 0:
                    continue
                face_img = cv2.resize(face_img, (200, 200))
                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

                label_id, confidence = recognizer.predict(gray)

                name = id_to_name.get(label_id, "Unknown")
                text = f"{name} ({confidence:.1f})"

                if confidence < 80:
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                    name = "Unknown"
                    text = "Unknown"

                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.imshow('Face Recognition - LBPH + MediaPipe', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()