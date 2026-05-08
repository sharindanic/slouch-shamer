import os
import time
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker.task")

# MediaPipe pose landmark indices
_EAR_L,  _EAR_R  = 7,  8
_SHO_L,  _SHO_R  = 11, 12
_HIP_L,  _HIP_R  = 23, 24


def _ensure_model() -> None:
    if not os.path.exists(_MODEL_PATH):
        print("Downloading pose model (~5 MB) …")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("Model ready.")


class PostureDetector:
    def __init__(self):
        _ensure_model()
        opts = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(opts)

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(time.time() * 1000)
        return self._landmarker.detect_for_video(mp_img, ts_ms)

    def get_landmarks(self, results, frame_shape):
        if not results.pose_landmarks:
            return None

        h, w = frame_shape[:2]
        lm = results.pose_landmarks[0]  # first detected pose

        def px(idx):
            p = lm[idx]
            return (int(p.x * w), int(p.y * h))

        def vis(idx):
            return lm[idx].visibility or 0.0

        # Use whichever side the camera sees better
        if vis(_EAR_L) >= vis(_EAR_R):
            return dict(ear=px(_EAR_L), shoulder=px(_SHO_L), hip=px(_HIP_L))
        return dict(ear=px(_EAR_R), shoulder=px(_SHO_R), hip=px(_HIP_R))

    @staticmethod
    def calculate_angle(ear, shoulder, hip) -> float:
        a, b, c = np.array(ear), np.array(shoulder), np.array(hip)
        ba, bc = a - b, c - b
        denom = np.linalg.norm(ba) * np.linalg.norm(bc)
        if denom < 1e-6:
            return 180.0
        cos_val = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_val)))

    def close(self):
        self._landmarker.close()
