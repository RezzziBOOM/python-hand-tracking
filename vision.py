import cv2
import mediapipe as mp
import tkinter as tk
from threading import Thread

# Inisialisasi Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Fungsi menghitung jumlah jari
def count_fingers(landmarks):
    tips = [4, 8, 12, 16, 20]
    count = 0
    if landmarks[tips[0]].x < landmarks[tips[0] - 1].x:
        count += 1
    for tip in tips[1:]:
        if landmarks[tip].y < landmarks[tip - 2].y:
            count += 1
    return count

# Fungsi deteksi gestur Love Korea
def detect_korean_love(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    distance = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
    return distance < 0.05

# Fungsi untuk memulai deteksi kamera
def start_detection():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                finger_count = count_fingers(hand_landmarks.landmark)
                gesture_text = f"Jari: {finger_count}"
                if detect_korean_love(hand_landmarks.landmark):
                    gesture_text = "I Love You"
                cv2.putText(frame, gesture_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Deteksi Jari dan Gestur", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC untuk keluar
            break

    cap.release()
    cv2.destroyAllWindows()

# Fungsi ketika tombol ditekan (pakai thread supaya GUI tidak freeze)
def on_start_button_click():
    Thread(target=start_detection).start()

# GUI Tkinter
root = tk.Tk()
root.title("Deteksi Gestur Tangan")
root.geometry("300x150")
root.resizable(False, False)

label = tk.Label(root, text="Deteksi Gestur Jari & Love Korea", font=("Arial", 12))
label.pack(pady=10)

start_button = tk.Button(root, text="Mulai Deteksi", command=on_start_button_click, bg="green", fg="white", font=("Arial", 10))
start_button.pack(pady=5)

exit_button = tk.Button(root, text="Keluar", command=root.destroy, bg="red", fg="white", font=("Arial", 10))
exit_button.pack(pady=5)

root.mainloop()
