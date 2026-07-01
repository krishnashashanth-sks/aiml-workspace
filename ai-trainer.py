import math
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Constants & Configuration ---
POSE_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24), (11, 13), (13, 15), 
    (12, 14), (14, 16), (23, 25), (25, 27), (24, 26), (26, 28)
]

MODEL_PATH = "models/pose_landmarker_heavy.task"
VIDEO_PATH = "models/13990524_2160_3840_30fps.mp4"

# --- Helper Function ---
def findAngle(frame, lmlist, p1, p2, p3, draw=True):
    """Calculates and draws the angle between three landmarks."""
    # Each entry in lmlist is [id, cx, cy], so we index for X and for Y
    x1, y1 = lmlist[p1][1:]
    x2, y2 = lmlist[p2][1:]
    x3, y3 = lmlist[p3][1:]
    
    # Calculate the angle
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
        
    if draw:
        # Draw connections
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)
        cv2.line(frame, (x3, y3), (x2, y2), (255, 255, 255), 3)
        
        # Draw joints
        for (x, y) in [(x1, y1), (x2, y2), (x3, y3)]:
            cv2.circle(frame, (x, y), 8, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, (x, y), 13, (255, 0, 0), 2)
            
        cv2.putText(frame, str(int(angle)), (x2 - 50, y2 + 50), 
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
    return angle

# --- MediaPipe Setup ---
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.PoseLandmarker.create_from_options(options)

# --- Video Setup ---
cap = cv2.VideoCapture(VIDEO_PATH)
print("Press 'q' to quit")

count = 0
direction = 0  # 0 for going up, 1 for going down

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
        
    frame = cv2.resize(frame, (671, 1200))
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    
    detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)
    
    lmlist = []
    
    if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
        for pose_landmarks in  detection_result.pose_landmarks:
            for idx, lm in enumerate(pose_landmarks):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([idx, cx, cy])
            
    # Draw and calculate only if the list is populated
    if lmlist:
        # Example tracking: Shoulder=11, Elbow=13, Wrist=15
        angle = findAngle(frame, lmlist, 11, 13, 15, draw=True)

        # Interpolation mapping for a bicep curl
        per = np.interp(angle, (220, 320), (0, 100))
        bar = np.interp(angle, (220, 320), (650, 100))
        
        if per == 100:
            if direction == 0:
                count += 0.5
                direction = 1
        if per == 0:
            if direction == 1:
                count += 0.5
                direction = 0
                
        cv2.rectangle(frame, (580, 100), (610, 650), (0, 255, 0), 3)
        cv2.rectangle(frame, (580, int(bar)), (610, 650), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, f"{int(per)}%", (560, 80), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cv2.putText(frame, f"REPS: {int(count)}", (50, 100), 
                cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 4)
    
    cv2.imshow("AI Trainer", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()