import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2

from src.vision.camera import CameraStream
from src.vision.gestures import GestureAnalyzer
from src.vision.hand_detector import HandDetector


class VisionWorker:
    """Keep camera capture and CPU hand tracking off Panda3D's render thread."""

    def __init__(
        self,
        camera: CameraStream,
        detector: HandDetector,
        gesture_analyzer: GestureAnalyzer,
    ):
        self.camera = camera
        self.detector = detector
        self.gesture_analyzer = gesture_analyzer
        self._running = threading.Event()
        self._lock = threading.Lock()
        self._frame_condition = threading.Condition(self._lock)
        self._threads: List[threading.Thread] = []
        self._latest_frame = None
        self._latest_background: Optional[Tuple[int, int, bytes]] = None
        self._latest_hands: List[Dict[str, Any]] = []
        self._error: Optional[Exception] = None
        self._frame_id = 0
        self._hands_updated_at = 0.0

    def start(self) -> None:
        if self._threads:
            raise RuntimeError("The vision worker can only be started once.")
        self._running.set()
        self._threads = [
            threading.Thread(
                target=self._capture_frames,
                name="camera-capture",
                daemon=True,
            ),
            threading.Thread(
                target=self._track_hands,
                name="hand-tracking",
                daemon=True,
            ),
        ]
        for thread in self._threads:
            thread.start()

    def snapshot(
        self,
    ) -> Tuple[Optional[Tuple[int, int, bytes]], List[Dict[str, Any]]]:
        with self._lock:
            error = self._error
            background = self._latest_background
            hands = (
                list(self._latest_hands)
                if time.monotonic() - self._hands_updated_at <= 0.25
                else []
            )
        if error is not None:
            raise RuntimeError("The camera or hand-tracking worker failed.") from error
        return background, hands

    def stop(self) -> None:
        self._running.clear()
        with self._frame_condition:
            self._frame_condition.notify_all()
        self.camera.release()
        for thread in self._threads:
            thread.join(timeout=2.0)
            if thread.is_alive():
                raise RuntimeError(
                    f"The {thread.name} worker did not stop within two seconds."
                )

    def _capture_frames(self) -> None:
        consecutive_capture_failures = 0
        try:
            while self._running.is_set():
                success, frame = self.camera.read()
                if not success or frame is None:
                    consecutive_capture_failures += 1
                    if consecutive_capture_failures >= 10:
                        raise RuntimeError("The camera stopped producing frames.")
                    time.sleep(0.01)
                    continue

                consecutive_capture_failures = 0
                height, width, _ = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                background = (width, height, cv2.flip(rgb_frame, 0).tobytes())
                with self._frame_condition:
                    self._latest_frame = frame
                    self._latest_background = background
                    self._frame_id += 1
                    self._frame_condition.notify()
        except Exception as error:
            self._report_error(error)

    def _track_hands(self) -> None:
        processed_frame_id = 0
        try:
            while self._running.is_set():
                with self._frame_condition:
                    self._frame_condition.wait_for(
                        lambda: self._frame_id > processed_frame_id
                        or not self._running.is_set(),
                        timeout=0.1,
                    )
                    if not self._running.is_set():
                        return
                    frame = self._latest_frame
                    frame_id = self._frame_id

                if frame is None or frame_id == processed_frame_id:
                    continue

                height, width, _ = frame.shape
                results = self.detector.process_frame(frame)
                hand_points = self.detector.extract_landmarks(results, width, height)
                hands = self.gesture_analyzer.analyze_hands(
                    hand_points, width, height
                )
                detected_at = time.monotonic()
                hands = [dict(hand, detected_at=detected_at) for hand in hands]
                with self._lock:
                    self._latest_hands = hands
                    self._hands_updated_at = detected_at
                processed_frame_id = frame_id
        except Exception as error:
            self._report_error(error)

    def _report_error(self, error: Exception) -> None:
        with self._frame_condition:
            self._error = error
            self._running.clear()
            self._frame_condition.notify_all()
