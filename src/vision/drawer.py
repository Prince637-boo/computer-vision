import cv2
from typing import List, Tuple
from src.config import HAND_CONNECTIONS, COLOR_JOINT, COLOR_CONNECTION, COLOR_TEXT

class Visualizer:
    """Utility class for drawing landmarks, connections, and metadata."""

    @staticmethod
    def draw_hands(frame: cv2.typing.MatLike, hands_points: List[List[Tuple[int, int]]]) -> None:
        for hand in hands_points:
            # Draw landmarks
            for pt in hand:
                cv2.circle(frame, pt, 5, COLOR_JOINT, -1)
            
            # Draw bone structure
            for p1, p2 in HAND_CONNECTIONS:
                if p1 < len(hand) and p2 < len(hand):
                    cv2.line(frame, hand[p1], hand[p2], COLOR_CONNECTION, 2)

    @staticmethod
    def draw_fps(frame: cv2.typing.MatLike, fps: int) -> None:
        cv2.putText(
            frame,
            f"FPS: {fps}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            COLOR_TEXT,
            2
        )