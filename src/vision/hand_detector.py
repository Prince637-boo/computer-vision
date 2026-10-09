import os
import time
import cv2
import mediapipe as mp
from typing import List, Tuple, Any

class HandDetector:
    """Wrapper around MediaPipe HandLandmarker Tasks API."""
    
    def __init__(self, model_path: str, num_hands: int = 2):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MediaPipe task model not found at: {model_path}")

        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def process_frame(self, frame_bgr: cv2.typing.MatLike) -> Any:
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)
        return self.landmarker.detect_for_video(mp_image, timestamp_ms)

    def extract_landmarks(self, results: Any, width: int, height: int) -> List[List[Tuple[int, int]]]:
        all_hands = []
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                hand_points = []
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * width), int(lm.y * height)
                    hand_points.append((cx, cy))
                all_hands.append(hand_points)
        return all_hands

    def close(self) -> None:
        if self.landmarker:
            self.landmarker.close()