import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)


def count_fingers(landmarks):
    tips = [4, 8, 12, 16, 20]  
    count = 0
    
   
    if landmarks[tips[0]].x < landmarks[tips[0] - 1].x:
        count += 1
    
   
    for tip in tips[1:]:
        if landmarks[tip].y < landmarks[tip - 2].y: 
            count += 1
            
    return count

# Fungsi untuk mendeteksi gestur "Love Korea"
def detect_korean_love(landmarks):
    # Landmark jempol dan telunjuk
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    
    # Hitung jarak antara ujung jempol dan ujung telunjuk
    distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
    
    # Jika jarak sangat kecil, dianggap membentuk hati
    return distance < 0.05

# Menyalakan kamera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Membalikkan gambar (mirror) agar seperti kamera
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Deteksi tangan
    result = hands.process(rgb_frame)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Gambar landmark di tangan
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Hitung jumlah jari yang diangkat
            finger_count = count_fingers(hand_landmarks.landmark)
            gesture_text = f"Jari: {finger_count}"
            
            # Deteksi gestur "Love Korea"
            if detect_korean_love(hand_landmarks.landmark):
                gesture_text = "I Love You"
            
            # Tampilkan hasil di layar
            cv2.putText(frame, gesture_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Tampilkan hasil
    cv2.imshow("Deteksi Jari dan Gestur", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  # Tekan ESC untuk keluar
        break

cap.release()
cv2.destroyAllWindows()
