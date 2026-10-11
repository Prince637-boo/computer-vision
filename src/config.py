from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "assets" / "models" / "hand_landmarker.task"

# Camera Settings
CAMERA_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS_TARGET = 30

# Detection Thresholds
NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.5
MIN_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Visual Styles (BGR Format)
COLOR_JOINT = (0, 0, 255)       # Red
COLOR_CONNECTION = (0, 255, 0)  # Green
COLOR_TEXT = (255, 0, 255)      # Magenta

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
]

# 3D Interaction Settings
INTERACTION_POINT_RADIUS = 0.1 # Radius of the red sphere for the interaction
OBJECT_DEFAULT_SIZE = 1.0       # Default size of the object to grab
# Workspace scale: maps [-1.0, 1.0] from hand to [-3.0, 3.0] in 3D scene
WORKSPACE_SCALE = 3.0

# PyVista Scene Styles
COLOR_OBJECT_DEFAULT = 'cyan'
COLOR_OBJECT_GRABBED = 'red'
COLOR_WORKSPACE_GRID = 'gray'