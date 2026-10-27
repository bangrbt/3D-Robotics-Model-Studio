"""MediaPipe recognition. The gate consumes actual new frames, never copies."""
from dataclasses import dataclass, field
from pathlib import Path
import sys
import urllib.request
import zipfile

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "gesture_recognizer.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"


@dataclass
class Observation:
    gesture: str | None = None
    confidence: float = 0.0
    landmarks: list = field(default_factory=list)
    handedness: str | None = None
    brightness: float = 0.0
    area: float = 0.0


def ensure_model(path=MODEL):
    path = Path(path)
    if path.is_file() and zipfile.is_zipfile(path):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part")
    print("Dang tai Gesture Recognizer chinh thuc cua Google...", flush=True)
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
            temporary.write_bytes(response.read())
        if not zipfile.is_zipfile(temporary):
            raise RuntimeError("Mo hinh tai ve khong hop le.")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def create_recognizer(path=MODEL):
    options = mp.tasks.vision.GestureRecognizerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp.tasks.vision.GestureRecognizer.create_from_options(options)


def recognize(recognizer, frame, timestamp_ms):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = recognizer.recognize_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
    observation = Observation(brightness=float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()))
    if result.hand_landmarks:
        observation.landmarks = result.hand_landmarks[0]
        if result.handedness and result.handedness[0]:
            observation.handedness = result.handedness[0][0].category_name
        xs = np.clip([p.x for p in observation.landmarks], 0, 1)
        ys = np.clip([p.y for p in observation.landmarks], 0, 1)
        observation.area = float((max(xs) - min(xs)) * (max(ys) - min(ys)))
    if result.gestures and result.gestures[0]:
        category = max(result.gestures[0], key=lambda item: item.score)
        observation.gesture = category.category_name
        observation.confidence = float(category.score)
    return observation


def open_camera(index):
    backends = (cv2.CAP_DSHOW, cv2.CAP_ANY) if sys.platform == "win32" else (cv2.CAP_ANY,)
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        cap.release()
    raise RuntimeError(f"Khong mo duoc webcam {index}. Dong Zoom/Teams, kiem tra quyen Camera, hoac thu --camera 1.")
