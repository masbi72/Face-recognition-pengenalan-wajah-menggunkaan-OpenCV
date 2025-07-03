import os
import pickle
import numpy as np

def load_encodings(folder='encodings'):
    """
    Membaca seluruh encoding dari folder 'encodings' yang disusun per user.
    - Format bisa dalam subfolder (dengan file encoding.pkl), atau file .pkl langsung.
    """
    encodings = []

    if not os.path.exists(folder):
        os.makedirs(folder)
        return encodings

    for item in os.listdir(folder):
        item_path = os.path.join(folder, item)

        if os.path.isdir(item_path):
            # Jika folder user, cari file encoding.pkl
            encoding_file = os.path.join(item_path, "encoding.pkl")
            if os.path.exists(encoding_file):
                try:
                    with open(encoding_file, "rb") as f:
                        data = pickle.load(f)
                        encodings.append(data)
                except Exception as e:
                    print(f"[WARN] Gagal membaca {encoding_file}: {e}")

        elif item.endswith(".pkl"):
            # Jika file pkl langsung
            try:
                with open(item_path, "rb") as f:
                    data = pickle.load(f)
                    encodings.append(data)
            except Exception as e:
                print(f"[WARN] Gagal membaca {item_path}: {e}")

    return encodings

def cosine_sim(a, b):
    """Menghitung cosine similarity antara dua vektor"""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
