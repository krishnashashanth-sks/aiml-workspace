# AIML Workspace — Project Portfolio

Welcome — this repository contains my small interactive AIML demos and utilities implemented in Python using MediaPipe and OpenCV. I am based in Hyderabad.

About
- Name: Sanga Krishna Shashanth (krishnashashanth-sks)
- Location: Hyderabad, India
- Contact: https://github.com/krishnashashanth-sks

Repository summary
This repo focuses on real-time hand / pose-based interaction demos using MediaPipe models and OpenCV. The `models/` folder contains the required .task model files (not all large model files are checked in). See each script for brief usage notes.

Quick links
- models/ — MediaPipe task files used by the scripts
- ai-trainer.py — pose-based rep counter (uses pose_landmarker)
- ai-virtual-mouse.py — control mouse pointer with hand gestures
- fingers-count.py — count raised fingers in webcam
- hand-volume.py — control system volume by hand distance
- virtual-painter.py — draw on-screen using hand gestures

Requirements
- Python 3.8+
- Packages (example):
  - opencv-python
  - numpy
  - mediapipe (or the MediaPipe Tasks Python package used in the scripts)
  - pyautogui (for ai-virtual-mouse)
  - pycaw and comtypes (for hand-volume on Windows)

Install example (recommended in a virtualenv):

python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.\.venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt

(If you don't have a requirements.txt, install packages manually: pip install opencv-python numpy mediapipe pyautogui comtypes pycaw)

Notes about models
- The scripts expect MediaPipe .task files in the models/ directory:
  - models/hand_landmarker.task — used by ai-virtual-mouse.py, fingers-count.py, hand-volume.py, virtual-painter.py
  - models/pose_landmarker_heavy.task — used by ai-trainer.py
- If you don't have these files, download them from the official MediaPipe tasks/model release pages and place them in `models/`.
- Some scripts reference a local absolute path for the model (virtual-painter.py). Edit the MODEL_PATH constant to point to `models/hand_landmarker.task` if you move the file into the repository.

How to run each demo
1) Realtime Rep Counter — ai-trainer.py
   - Purpose: Detect pose landmarks and count repetitions (example bicep curl counter).
   - Run: python ai-trainer.py
   - Notes: Expects `models/pose_landmarker_heavy.task` and an example video in models/ (the script uses a video file by default). Change VIDEO_PATH in the script to use webcam (0) or your own video.

2) AI Virtual Mouse — ai-virtual-mouse.py
   - Purpose: Move the system mouse pointer and click using hand gestures.
   - Run: python ai-virtual-mouse.py
   - Notes: Requires `pyautogui`. The script maps index-finger coordinates to screen coordinates. Tweak frame size, smoothing and frameR for different setups.

3) Fingers Counter — fingers-count.py
   - Purpose: Count the number of raised fingers from webcam feed.
   - Run: python fingers-count.py
   - Notes: Uses hand landmark logic; output appears on the OpenCV window.

4) Hand Volume Control — hand-volume.py
   - Purpose: Control system volume by distance between thumb and index finger.
   - Run: python hand-volume.py
   - Notes: Uses pycaw and comtypes — these are Windows specific. On Linux/mac you will need an alternative audio control library or adapt the code.

5) Virtual Painter — virtual-painter.py
   - Purpose: Paint/draw on screen with hand gestures and use top-bar color selection.
   - Run: python virtual-painter.py
   - Notes: The script references a top bar image under `models/` (a screenshot). Update TOP_BAR path if needed. Also set MODEL_PATH to `models/hand_landmarker.task` for portability.
