"""Pose Landmarker runtime; inference uses the original, unmirrored frame."""
from pathlib import Path
import urllib.request
import zipfile

import cv2
import mediapipe as mp

from .core import Observation

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT/"models"/"pose_landmarker_full.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"


def ensure_model(path=MODEL):
    path = Path(path)
    if path.is_file() and zipfile.is_zipfile(path):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part")
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
            temporary.write_bytes(response.read())
        if not zipfile.is_zipfile(temporary):
            raise RuntimeError("Mo hinh tai ve khong hop le; hay kiem tra mang va thu lai.")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def create_detector(path=MODEL):
    return mp.tasks.vision.PoseLandmarker.create_from_options(mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=.6,
        min_pose_presence_confidence=.6,
        min_tracking_confidence=.6,
    ))


def detect(detector, frame, timestamp_ms):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
    return Observation(result.pose_landmarks[0] if result.pose_landmarks else [],
                       result.pose_world_landmarks[0] if result.pose_world_landmarks else [],
                       frame.shape[1], frame.shape[0])
