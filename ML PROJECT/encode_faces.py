import cv2
import os
import numpy as np
import pickle

DATASET_DIR = "dataset"
faces = []
labels = []
label_ids = {}
current_id = 0

for person_name in os.listdir(DATASET_DIR):
    person_folder = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_folder):
        continue

    label_ids[person_name] = current_id
    id_ = current_id
    current_id += 1

    for img_file in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        faces.append(img)
        labels.append(id_)

faces = np.array(faces)
labels = np.array(labels)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, labels)

os.makedirs("models", exist_ok=True)
recognizer.save("models/face_model.yml")

with open("models/labels.pkl", "wb") as f:
    pickle.dump(label_ids, f)

print("Training completed. Model saved in models/face_model.yml")
