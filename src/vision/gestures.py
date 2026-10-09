from enum import Enum
from typing import List, Tuple, Dict, Any
from src.core.utils import calculate_distance_2d, normalize_coordinates

class GestureType(Enum):
    NONE = "NONE"
    IDLE = "IDLE"
    PINCH = "PINCH"

class GestureAnalyzer:
    """Analyzes hand landmark points to detect interactive gestures."""

    # Landmark Indices (MediaPipe Hand Topology)
    THUMB_TIP = 4
    INDEX_TIP = 8

    def __init__(self, pinch_threshold_pixels: float = 35.0):
        self.pinch_threshold = pinch_threshold_pixels

    def analyze_hand(self, hand_points: List[Tuple[int, int]], width: int, height: int) -> Dict[str, Any]:
        """
        Analyzes a single hand landmark list and returns current gesture state and interaction point.
        """
        if len(hand_points) < 21:
            return {"gesture": GestureType.NONE, "distance": 0.0, "position_3d": (0.0, 0.0)}

        thumb_pt = hand_points[self.THUMB_TIP]
        index_pt = hand_points[self.INDEX_TIP]

        # Calculate Euclidean distance between Thumb and Index tip
        distance = calculate_distance_2d(thumb_pt, index_pt)
        is_pinching = distance <= self.pinch_threshold

        # Interaction midpoint between index and thumb
        midpoint_px = (
            int((thumb_pt[0] + index_pt[0]) / 2),
            int((thumb_pt[1] + index_pt[1]) / 2)
        )

        # Convert midpoint to normalized 3D viewport space [-1.0, 1.0]
        norm_x, norm_y = normalize_coordinates(midpoint_px, width, height)

        return {
            "gesture": GestureType.PINCH if is_pinching else GestureType.IDLE,
            "distance": round(distance, 1),
            "midpoint_px": midpoint_px,
            "normalized_pos": (norm_x, norm_y),
            "thumb_pt": thumb_pt,
            "index_pt": index_pt
        }