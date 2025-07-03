import os

def show_menu():
    print("\n=== Sistem Login Wajah ===")
    print("1. Ambil Dataset Wajah")
    print("2. Latih Model LBPH")
    print("3. Jalankan GUI Debug")
    print("4. Jalankan Web Login Flask")
    print("5. Buat Ulang label_names.pkl dari Dataset")
    print("6. Rename nama file dataset menjadi username_index.jpg")
    print("0. Keluar")

while True:
    show_menu()
    choice = input("Pilih menu (0-6): ")

    if choice == "1":
        os.system("python FacialRecognition/01_face_dataset.py")
    elif choice == "2":
        print("[INFO] Melatih model LBPH dari dataset...")
        os.system("python FacialRecognition/02_face_training.py")
        print("[INFO] Pelatihan selesai.")
    elif choice == "3":
        os.system("python gui_lbph_debug.py")
    elif choice == "4":
        # Cek label_names.pkl sebelum menjalankan Flask
        label_pkl = "FacialRecognition/trainer/label_names.pkl"
        if not os.path.exists(label_pkl):
            print(f"[ERROR] label_names.pkl tidak ditemukan di {label_pkl}.")
            print("Silakan jalankan menu 5 untuk membuat ulang label_names.pkl.")
        else:
            os.system("python app.py")
    elif choice == "5":
        os.system("python FacialRecognition/generate_label_names.py")
    elif choice == "6":
        os.system("python FacialRecognition/rename_dataset.py")
    elif choice == "0":
        print("Keluar...")
        break
    else:
        print("[WARNING] Pilihan tidak valid. Silakan pilih angka 0-6.")
