import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pyautogui

wCam,hCam=640,430
wScr,hScr=pyautogui.size()
frameR=100
smoothening=7

xp,yp=0,0
cx,cy=0,0

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12), # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),# Ring finger
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky & Palm base
]

FINGER_TIPS = [4,8,12,16,20]

MODEL_PATH = "C:/Users/SANGA/Downloads/hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
print("Press 'q' to quit")
cap.set(3,wCam)
cap.set(4,hCam)

while cap.isOpened():
    success, frame = cap.read()
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
            cv2.line(frame, coords[start_idx], coords[end_idx], (255, 0, 255), 3)
            
        for (cx, cy) in coords:
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
            
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
        x1,y1=lmlist[8][1:]
        x2,y2=lmlist[12][1:]
        if fingers[2]==1 and fingers [3]==0:
            x3=np.interp(x1,(frameR,wCam-frameR),(0,wScr))
            y3=np.interp(y1,(frameR,hCam-frameR),(0,hScr))
            cx=xp+(x3-xp)/smoothening
            cy=yp+(y3-yp)/smoothening
            pyautogui.moveTo(wScr-cx,cy)
            xp,yp=cx,cy
        if fingers[2]==1 and fingers [3]==1:
            length=np.hypot(x2-x1,y2-y1)
            if length<40:
                cv2.circle(frame,(x1,y1),15,(0,255,0),cv2.FILLED)
                pyautogui.click()
    cv2.rectangle(frame,(frameR,frameR),(wCam-frameR,hCam-frameR),(255,0,255),2)
    frame=cv2.flip(frame,1)
    cv2.imshow("AI Virtual Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()