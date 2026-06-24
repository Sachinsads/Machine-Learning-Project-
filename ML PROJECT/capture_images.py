import cv2
import os

# Input student details
roll = input("Enter Roll Number: ")
name = input("Enter Name: ")
label = f"{roll}_{name.replace(' ', '_')}"

# Create folder for student images
DATASET_DIR = "dataset"
person_dir = os.path.join(DATASET_DIR, label)
os.makedirs(person_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
print("Press 'c' to capture image, 'q' to quit")
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Capture Images", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        img_path = os.path.join(person_dir, f"{label}_{count}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"Saved {img_path}")
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Captured {count} images for {label}")
