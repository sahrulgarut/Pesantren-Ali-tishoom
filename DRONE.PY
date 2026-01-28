import cv2
import os
import time
from ultralytics import YOLO

# 1. SETUP MODEL DAN FOLDER
model = YOLO('yolov8n.pt') 
output_folder = "hasil_patroli"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

last_save_time = 0
save_interval = 5  # Jeda simpan foto (dalam detik) agar memori tidak penuh

def drone_ai_pro(frame, zoom_level=2.0):
    global last_save_time
    h, w = frame.shape[:2]
    ch, cw = int(h / zoom_level), int(w / zoom_level)
    sy, sx = (h - ch) // 2, (w - cw) // 2
    
    cropped = frame[sy:sy+ch, sx:sx+cw]
    zoomed_frame = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)

    results = model(zoomed_frame, conf=0.4, verbose=False)
    target_detected = False

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            
            # Kelas: Orang (0), Mobil (2), Motor (3), Truk (7)
            if cls in [0, 2, 3, 7]:
                target_detected = True
                label = f"{model.names[cls]}"
                
                # Gambar Box & Label
                cv2.rectangle(zoomed_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(zoomed_frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # 2. LOGIKA AUTO-SAVE (Jika target ditemukan & sudah melewati interval waktu)
    current_time = time.time()
    if target_detected and (current_time - last_save_time > save_interval):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{output_folder}/deteksi_{timestamp}.jpg"
        cv2.imwrite(filename, zoomed_frame)
        print(f" [!] Target Terdeteksi! Foto disimpan: {filename}")
        last_save_time = current_time

    return zoomed_frame

# --- RUN PROGRAM ---
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret: break

    output = drone_ai_pro(frame, zoom_level=2.0)
    
    cv2.putText(output, "STATUS: PATROL ACTIVE", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow('Drone AI Monitor', output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()