# gui_lbph_debug.py
import cv2
import os
import pickle

def run_gui():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("FacialRecognition/trainer/trainer.yml")

    label_path = "FacialRecognition/trainer/label_names.pkl"
    if not os.path.exists(label_path):
        print("[ERROR] label_names.pkl tidak ditemukan.")
        return

    with open(label_path, "rb") as f:
        label_obj = pickle.load(f)
        # Pastikan label_dict adalah dict: str(id) -> name
        if isinstance(label_obj, dict):
            label_dict = label_obj
        elif isinstance(label_obj, list):
            # Convert list to dict: id mulai dari 1
            label_dict = {str(i+1): name for i, name in enumerate(label_obj)}
        else:
            print("[ERROR] Format label_names.pkl tidak dikenali.")
            return

    cam = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier("FacialRecognition/haarcascade_frontalface_default.xml")

    if not cam.isOpened():
        print("[ERROR] Tidak dapat membuka kamera.")
        return

    print("[INFO] GUI Debug LBPH dimulai...")

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            id_pred, conf = recognizer.predict(gray[y:y + h, x:x + w])
            name = label_dict.get(str(id_pred), f"User {id_pred}")
            color = (0, 255, 0) if conf < 70 else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Face Recognition (Debug)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_gui()
