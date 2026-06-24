import cv2
import pandas as pd
from datetime import datetime
import pickle
import os

# Load model and labels
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("models/face_model.yml")

with open("models/labels.pkl", "rb") as f:
    label_ids = pickle.load(f)
labels = {v:k for k,v in label_ids.items()}

attendance_file = "attendance.csv"
if not os.path.exists(attendance_file):
    pd.DataFrame(columns=["Name", "Date", "Time"]).to_csv(attendance_file, index=False)

attendance_df = pd.read_csv(attendance_file)

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roi_gray)
        if conf < 80:
            name = labels[id_]
            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")
            if not ((attendance_df['Name']==name) & (attendance_df['Date']==date_str)).any():
                attendance_df = pd.concat([attendance_df, pd.DataFrame([[name, date_str, time_str]], columns=["Name","Date","Time"])], ignore_index=True)
                attendance_df.to_csv(attendance_file, index=False)
                print(f"Attendance marked: {name}")
        else:
            name = "Unknown"

        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)

    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
