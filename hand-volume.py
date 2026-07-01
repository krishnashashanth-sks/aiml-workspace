import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pycaw.pycaw import AudioUtilities,IAudioEndpointVolume
from comtypes import CLSCTX_ALL

devices=AudioUtilities.GetSpeakers()
interface=devices.Activate(IAudioEndpointVolume._iid_,CLSCTX_ALL,None)
volume=interface.QueryInterface(IAudioEndpointVolume)
volume_range=volume.GetVolumeRange()
minVol,maxVol=volume_range[0],volume_range[1]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12), # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),# Ring finger
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky & Palm base
]

model_path = "C:/Users/SANGA/Downloads/hand_landmarker.task"

base_options=python.BaseOptions(model_asset_path=model_path)
options=vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.VIDEO
)

detector=vision.HandLandmarker.create_from_options(options)

cap=cv2.VideoCapture(0)
print("Press 'q' to quit")

while cap.isOpened():
    success,frame=cap.read()
    if not success:
        continue
    frame=cv2.flip(frame,1)
    h,w,_=frame.shape
    rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    mp_image=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb_frame)

    frame_timestamp_ms=int(cap.get(cv2.CAP_PROP_POS_MSEC))

    detection_result=detector.detect_for_video(mp_image,frame_timestamp_ms)
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            coords = []
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                coords.append((cx, cy))
            for connection in HAND_CONNECTIONS:
                start_idx,end_idx=connection
                cv2.line(frame,coords[start_idx],coords[end_idx],(255,0,255),3)
            for (cx,cy) in coords:
                cv2.circle(frame,(cx,cy),5,(0,255,0),cv2.FILLED)

            x1,y1=coords[4]
            x2,y2=coords[8]
            length=np.hypot(x2-x1,y2-y1)
            vol=np.interp(length,[20,220],[minVol,maxVol])
            volp=np.interp(length,[20,220],[0,100])
            volume.SetMasterVolumeLevel(vol,None)
            cv2.putText(frame,f"Volume:{int(volp)}",(20,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,355),5)
            
    cv2.imshow("Hand Volume",frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()