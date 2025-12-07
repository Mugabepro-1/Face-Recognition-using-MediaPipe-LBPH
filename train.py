import cv2
import os
import numpy as np
import pickle
import json
from pathlib import Path


def train_lbph():
    dataset_dir = "dataset"
    people = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

    if len(people) < 2:
        print("Need at least 2 people in dataset/")
        return

    faces = []
    labels = []
    label_map = {}
    label_id = 0

    for person in people:
        person_path = os.path.join(dataset_dir, person)
        label_map[label_id] = person

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            faces.append(img)
            labels.append(label_id)

        label_id += 1

    # Save label map
    os.makedirs("models", exist_ok=True)
    with open("models/label_map.json", "w") as f:
        json.dump(label_map, f, indent=4)

    # Train LBPH
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.save("models/lbph_model.yml")

    print("Training complete!")
    print("Label map:", label_map)


if __name__ == "__main__":
    train_lbph()