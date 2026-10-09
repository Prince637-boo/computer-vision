computer-vision/
│
├── assets/
│   └── models/
│       └── hand_landmarker.task      # MediaPipe model file
│
├── src/
│   ├── __init__.py
│   ├── config.py                     # Centralized settings and constants
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── camera.py                 # Encapsulated camera capture stream
│   │   ├── hand_detector.py          # MediaPipe detection logic wrapper
│   │   └── drawer.py                 # Visualization and rendering utilities
│   │
│   └── core/
│       ├── __init__.py
│       └── utils.py                  # FPS calculator, math utilities (e.g., pinch distance)
│
├── main.py                           # Application entry point
├── requirements.txt                  # Dependency list
└── README.md