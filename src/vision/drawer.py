import cv2
from typing import List, Tuple, Dict, Any
from src.config import HAND_CONNECTIONS, COLOR_JOINT, COLOR_CONNECTION, COLOR_TEXT
from src.vision.gestures import GestureType

class Visualizer:
    """Utility class for drawing landmarks, connections, and metadata."""

    @staticmethod
    def draw_hands(frame: cv2.typing.MatLike, hands_points: List[List[Tuple[int, int]]]) -> None:
        for hand in hands_points:
            for pt in hand:
                cv2.circle(frame, pt, 5, COLOR_JOINT, -1)
            
            for p1, p2 in HAND_CONNECTIONS:
                if p1 < len(hand) and p2 < len(hand):
                    cv2.line(frame, hand[p1], hand[p2], COLOR_CONNECTION, 2)

    @staticmethod
    def draw_gesture_info(frame: cv2.typing.MatLike, gesture_data: Dict[str, Any]) -> None:
        gesture = gesture_data.get("gesture", GestureType.NONE)
        if gesture == GestureType.NONE:
            return

        midpoint = gesture_data["midpoint_px"]
        thumb = gesture_data["thumb_pt"]
        index = gesture_data["index_pt"]
        norm_pos = gesture_data["normalized_pos"]

        # Color feedback: Red = PINCH, Cyan = IDLE
        color = (0, 0, 255) if gesture == GestureType.PINCH else (255, 255, 0)

        # Draw line between Thumb and Index
        cv2.line(frame, thumb, index, color, 2)

        # Draw midpoint circle
        cv2.circle(frame, midpoint, 8, color, -1)

        # Text overlay
        status_text = f"Gesture: {gesture.value} | Pos: {norm_pos}"
        cv2.putText(frame, status_text, (midpoint[0] + 15, midpoint[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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