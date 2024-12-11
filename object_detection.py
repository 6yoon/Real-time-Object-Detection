import cv2
from ultralytics import YOLO
import threading
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import argparse

# 커맨드라인 인자 파서 설정
parser = argparse.ArgumentParser(description="Real-time Object Detection using YOLO")
parser.add_argument("video_source", type=str, help="URL of the video stream (e.g., http://127.0.0.1:51988/)")
args = parser.parse_args()

# YOLO 모델 로드 (사전 훈련된 가중치 사용)
try:
    model = YOLO("yolo11n.pt")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

# 비디오 캡처 객체 생성 (Streamlink가 사용하는 로컬 주소)
cap = cv2.VideoCapture(args.video_source)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# 프레임 저장용 변수
frame = None
running = True

def capture_frames():
    global frame, running
    while running:
        success, new_frame = cap.read()
        if success:
            # 프레임을 복사하여 저장 (스레드 안전성 확보)
            frame = new_frame.copy()
        else:
            print("Warning: Could not read frame.")

# 프레임 캡처를 위한 스레드 시작
threading.Thread(target=capture_frames, daemon=True).start()

last_detected_cars = []  # 마지막 감지된 차량 정보 저장

# 나눔고딕 폰트 파일 경로
font_path = "NanumGothicBold.ttf"  # 나눔고딕 폰트 파일 경로
font_size = 24
font = ImageFont.truetype(font_path, font_size)

while True:
    if frame is None:
        continue  # 프레임이 준비될 때까지 대기

    # 원본 프레임 크기 가져오기
    original_height, original_width = frame.shape[:2]

    # 새로운 너비와 높이 계산 (50% 축소)
    new_width = int(original_width * 0.5)
    new_height = int(original_height * 0.5)

    # 중앙 기준으로 프레임 자르기
    x_start = (original_width - new_width) // 2
    y_start = (original_height - new_height) // 2

    # 프레임 자르기
    cropped_frame = frame[y_start:y_start + new_height, x_start:x_start + new_width]

    # BGR에서 RGB로 변환
    rgb_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)

    # 모델 예측 (신뢰도 임계값 조정)
    results = model(rgb_frame, conf=0.4)  # 신뢰도 임계값을 0.4로 설정

    # 감지된 결과 처리
    detected_count = 0  
    car_count = 0       
    person_count = 0    
    last_detected_cars.clear()  
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1) + x_start, int(y1) + y_start, int(x2) + x_start, int(y2) + y_start
            
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if conf > 0.4:  
                detected_count += 1

                name = model.names[cls]

                if name in ["car", "truck", "bus"]:  
                    car_count += 1
                
                if name == "person":  
                    person_count += 1

                last_detected_cars.append((x1, y1, x2, y2, name))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                img_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(img_pil)

                label = f"{name} {conf:.2f}"
                draw.text((x1, y1 - 10), label, font=font, fill=(0, 255, 0))

                frame = np.array(img_pil)

    detected_cars_count = len(last_detected_cars)
    
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)

    draw.rectangle([10, 20, original_width - 10, 100], fill=(0, 0, 0, 150)) 

    draw.text((20, 30), f"현재 이동 중인 차량: {car_count}", font=font, fill=(255, 255, 255))
    draw.text((20, 60), f"현재 유동 인구: {person_count}", font=font, fill=(255, 255, 255))

    frame = np.array(img_pil)

    cv2.imshow("Real-time Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
running = False
cap.release()
cv2.destroyAllWindows()