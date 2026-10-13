import math
import time
from enum import Enum
from typing import Any, Dict, List, Sequence, Tuple

from src.core.smoothing import OneEuro
from src.core.utils import normalize_coordinates


class GestureType(Enum):
    NONE = "NONE"
    IDLE = "IDLE"
    PINCH = "PINCH"
    OPEN_PALM = "OPEN PALM"
    FIST = "FIST"


class GestureAnalyzer:
    """Detect and smooth hand gestures from MediaPipe's 21 hand landmarks."""

    THUMB_TIP = 4
    INDEX_TIP = 8
    WRIST = 0
    FINGER_TIPS = (8, 12, 16, 20)
    FINGER_PIPS = (6, 10, 14, 18)

    PALM_IDS = (0, 5, 9, 13, 17)

    def __init__(self, pinch_threshold_pixels: float = 35.0):
        self.pinch_threshold = pinch_threshold_pixels
        self._palm_filters: Dict[int, Tuple[OneEuro, OneEuro]] = {}
        self._pinching: Dict[int, bool] = {}
        self._fists: Dict[int, bool] = {}

    def analyze_hand(
        self, hand_points: List[Tuple[int, int]], width: int, height: int
    ) -> Dict[str, Any]:
        """Analyze one hand, preserving the original single-hand API."""
        return self._analyze_hand(hand_points, width, height, 0)

    def analyze_hands(
        self, hands: Sequence[List[Tuple[int, int]]], width: int, height: int
    ) -> List[Dict[str, Any]]:
        """Analyze all visible hands and retain independent smoothing state."""
        ordered_hands = sorted(hands, key=lambda points: points[0][0] if points else 0)
        if not ordered_hands:
            self._palm_filters.clear()
            self._pinching.clear()
            self._fists.clear()
            return []

        self._palm_filters = {
            hand_id: filters
            for hand_id, filters in self._palm_filters.items()
            if hand_id < len(ordered_hands)
        }
        self._pinching = {
            hand_id: state
            for hand_id, state in self._pinching.items()
            if hand_id < len(ordered_hands)
        }
        self._fists = {
            hand_id: state
            for hand_id, state in self._fists.items()
            if hand_id < len(ordered_hands)
        }
        return [
            self._analyze_hand(points, width, height, hand_id)
            for hand_id, points in enumerate(ordered_hands)
        ]

    def _analyze_hand(
        self,
        hand_points: Sequence[Tuple[int, int]],
        width: int,
        height: int,
        hand_id: int,
    ) -> Dict[str, Any]:
        if len(hand_points) < 21:
            self._palm_filters.pop(hand_id, None)
            self._pinching.pop(hand_id, None)
            self._fists.pop(hand_id, None)
            return {"gesture": GestureType.NONE, "distance": 0.0}

        points = [(float(x), float(y)) for x, y in hand_points]
        thumb_pt = points[self.THUMB_TIP]
        index_pt = points[self.INDEX_TIP]
        wrist = points[self.WRIST]
        distance = math.hypot(thumb_pt[0] - index_pt[0], thumb_pt[1] - index_pt[1])
        palm_size = math.hypot(
            points[9][0] - wrist[0], points[9][1] - wrist[1]
        )
        pinch_limit = min(self.pinch_threshold, max(14.0, palm_size * 0.5))
        pinch_threshold = pinch_limit * (1.2 if self._pinching.get(hand_id, False) else 0.85)
        is_pinching = distance <= pinch_threshold
        index_extended = math.hypot(
            points[self.INDEX_TIP][0] - wrist[0],
            points[self.INDEX_TIP][1] - wrist[1],
        ) > math.hypot(points[6][0] - wrist[0], points[6][1] - wrist[1]) * 1.18
        self._pinching[hand_id] = is_pinching and index_extended

        extended_fingers = sum(
            math.hypot(
                points[tip][0] - wrist[0], points[tip][1] - wrist[1]
            )
            > math.hypot(
                points[pip][0] - wrist[0], points[pip][1] - wrist[1]
            )
            * 1.18
            for tip, pip in zip(self.FINGER_TIPS, self.FINGER_PIPS)
        )
        was_fist = self._fists.get(hand_id, False)
        is_fist = extended_fingers <= (1 if was_fist else 0)
        self._fists[hand_id] = is_fist

        if is_pinching and index_extended:
            gesture = GestureType.PINCH
        elif extended_fingers >= 3:
            self._fists[hand_id] = False
            gesture = GestureType.OPEN_PALM
        elif is_fist:
            gesture = GestureType.FIST
        else:
            gesture = GestureType.IDLE

        midpoint_px = (
            int((thumb_pt[0] + index_pt[0]) / 2),
            int((thumb_pt[1] + index_pt[1]) / 2),
        )
        norm_x, norm_y = normalize_coordinates(midpoint_px, width, height)
        wrist_pos = normalize_coordinates((int(wrist[0]), int(wrist[1])), width, height)
        palm_px = (
            sum(points[index][0] for index in self.PALM_IDS) / len(self.PALM_IDS),
            sum(points[index][1] for index in self.PALM_IDS) / len(self.PALM_IDS),
        )
        palm_filters = self._palm_filters.get(hand_id)
        if palm_filters is None:
            palm_filters = (OneEuro(), OneEuro())
            self._palm_filters[hand_id] = palm_filters
        timestamp = time.monotonic()
        filtered_palm = (
            palm_filters[0](palm_px[0], timestamp),
            palm_filters[1](palm_px[1], timestamp),
        )
        palm_pos = normalize_coordinates(
            (int(filtered_palm[0]), int(filtered_palm[1])), width, height
        )

        return {
            "gesture": gesture,
            "distance": round(distance, 1),
            "midpoint_px": midpoint_px,
            "normalized_pos": (norm_x, norm_y),
            "wrist_pos": wrist_pos,
            "palm_pos": palm_pos,
            "thumb_pt": (int(thumb_pt[0]), int(thumb_pt[1])),
            "index_pt": (int(index_pt[0]), int(index_pt[1])),
        }
