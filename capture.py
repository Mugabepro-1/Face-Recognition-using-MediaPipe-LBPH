import cv2
import mediapipe as mp
import os
import sys

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


def capture_faces(person_name, num_samples=50):
    dataset_dir = "dataset"
    person_dir = os.path.join(dataset_dir, person_name)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        count = 0
        print(f"Collecting {num_samples} samples for {person_name}. Press 'q' when done.")

        while cap.isOpened() and count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(frame, detection)

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

                    cv2.imshow('Collecting face', frame)
                    cv2.imwrite(os.path.join(person_dir, f"{count}.jpg"), gray)
                    count += 1
                    print(f"Saved {count}/{num_samples}")

            if cv2.waitKey(150) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python capture.py <person_name>")
        sys.exit(1)
    capture_faces(sys.argv[1])