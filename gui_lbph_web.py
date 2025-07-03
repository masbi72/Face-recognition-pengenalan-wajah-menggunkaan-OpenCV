# gui_lbph_web.py

import cv2
import os
import pickle

def run_gui():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    model_path = 'FacialRecognition/trainer/trainer.yml'
    label_path = 'FacialRecognition/trainer/label_map.pkl'

    if not os.path.exists(model_path) or not os.path.exists(label_path):
        print("[ERROR] Model atau mapping belum ditemukan.")
        return

    recognizer.read(model_path)
    with open(label_path, 'rb') as f:
        label_map = pickle.load(f)

    id_to_name = {v: k.capitalize() for k, v in label_map.items()}

    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    print("[INFO] Tekan ESC untuk keluar GUI LBPH")

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            id_pred, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            name = id_to_name.get(id_pred, f"User {id_pred}")
            conf_text = f"{round(100 - confidence)}%"

            color = (0, 255, 0) if confidence < 70 else (0, 0, 255)
            label = f"{name} ({conf_text})"

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), font, 0.8, color, 2)

        cv2.imshow("Web GUI - Face Recognition", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cam.release()
    cv2.destroyAllWindows()
