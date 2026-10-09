import time
import math
from typing import Tuple

class FPSCounter:
    """Utility class to measure and display real-time frames per second."""
    
    def __init__(self):
        self.prev_time = time.time()

    def update(self) -> int:
        curr_time = time.time()
        delta = curr_time - self.prev_time
        fps = 1 / delta if delta > 0 else 0
        self.prev_time = curr_time
        return int(fps)

def calculate_distance_2d(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """Calculates Euclidean distance between two 2D pixel coordinates."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def normalize_coordinates(point: Tuple[int, int], width: int, height: int) -> Tuple[float, float]:
    """
    Normalizes 2D pixel coordinates to [-1.0, 1.0] workspace coordinate space.
    Top-Left: (-1, 1), Center: (0, 0), Bottom-Right: (1, -1)
    """
    norm_x = (point[0] / width) * 2.0 - 1.0
    norm_y = -((point[1] / height) * 2.0 - 1.0)  # Invert Y for standard 3D coordinate space
    return round(norm_x, 3), round(norm_y, 3)