"""One Hand Landmarker pass per frame; no pose/gesture classifier overhead."""
from pathlib import Path
import urllib.request
import zipfile

import cv2
import mediapipe as mp

from .core import HandObservation

ROOT=Path(__file__).resolve().parent.parent
MODEL=ROOT/"models"/"hand_landmarker.task"
MODEL_URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def ensure_model(path=MODEL):
    path=Path(path)
    if path.is_file() and zipfile.is_zipfile(path):
        return path
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(".part")
    try:
        with urllib.request.urlopen(MODEL_URL,timeout=60) as response:
            temporary.write_bytes(response.read())
        if not zipfile.is_zipfile(temporary):
            raise RuntimeError("Mô hình bàn tay tải về không hợp lệ.")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def create_detector(path=MODEL):
    return mp.tasks.vision.HandLandmarker.create_from_options(mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,min_hand_detection_confidence=.6,min_hand_presence_confidence=.6,min_tracking_confidence=.6))


def detect(detector,frame,timestamp):
    result=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),timestamp)
    return HandObservation(result.hand_landmarks[0] if result.hand_landmarks else [],
                           result.hand_world_landmarks[0] if result.hand_world_landmarks else [],
                           result.handedness[0][0].category_name if result.handedness and result.handedness[0] else None,
                           frame.shape[1],frame.shape[0])
