import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12), # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),# Ring finger
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky & Palm base
]

FINGER_TIPS = [4,8,12,16,20]

eraserThickness=50
brushThickness=30

topBarPath="C:/Users/SANGA/Downloads/Screenshot 2026-06-30 at 08-18-24 Canva AI - Canva.png"

MODEL_PATH = "C:/Users/SANGA/Downloads/hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.VIDEO
)

imgCanvas=np.zeros((720,1280,3),np.uint8)
detector = vision.HandLandmarker.create_from_options(options)

drawColor=(255,0,255)

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

print("Press 'q' to quit")

while cap.isOpened():
    success, frame = cap.read()
    frame=cv2.flip(frame,1)
    topBar=cv2.resize(cv2.imread(topBarPath),(1280,100))
    frame[0:100,0:1280]=topBar
    if not success:
        continue
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

    lmlist = []

    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            coords = []
            for id, lm in enumerate(hand_landmarks):
                cx, cy = int(lm.x * w), int(lm.y * h)
                coords.append((cx, cy))
                lmlist.append([id, cx, cy])
            
        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            cv2.line(frame, coords[start_idx], coords[end_idx], (255,0,0),2)
            
        for (cx, cy) in coords:
            cv2.circle(frame, (cx, cy), 5, (0,0,255), cv2.FILLED)

        fingers = []
        
        # Check X-axis value for Thumb (index 1)
        if lmlist[4][1] < lmlist[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # Check Y-axis value for Fingers (index 2)
        for id in FINGER_TIPS:
            if lmlist[id][2] < lmlist[id - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

    if len(lmlist)!=0:
        xp,yp=0,0
        x1,y1=lmlist[8][1:]
        x2,y2=lmlist[12][1:]
        if fingers[2] and fingers[3]:
            if y1 <125:
                if 300<x1<450:
                    drawColor=(255,0,255)
                elif 450<x1<600:
                    drawColor=(0,0,225)
                elif 600<x1<750:
                    drawColor=(225,0,0)
                elif 750<x1<900:
                    drawColor=(0,255,0)
                elif 900<x1<1050:
                    drawColor=(0,0,0)
            cv2.rectangle(frame,(x1,y1-25),(x2,y2+25),drawColor,cv2.FILLED)

        if fingers[2] and fingers[3]==False:
            cv2.circle(frame,(x1,y1),15,drawColor,cv2.FILLED)
            if xp==0 and yp==0:
                xp,yp=x1,y1
            if drawColor==(0,0,0):
                cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,eraserThickness)
                cv2.line(frame,(xp,yp),(x1,y1),drawColor,eraserThickness)
            else:
                cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,brushThickness)
                cv2.line(frame,(xp,yp),(x1,y1),drawColor,brushThickness)
            xp,yp=x1,y1
        
    imgGray=cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
    _,imgInv=cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
    imgInv=cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    frame=cv2.bitwise_and(frame,imgInv)
    frame=cv2.bitwise_or(frame,imgCanvas)

    # frame=cv2.addWeighted(imgCanvas,0.5,frame,0.5,0)
    cv2.imshow("AI Painter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()