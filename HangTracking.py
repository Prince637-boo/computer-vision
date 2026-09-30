import cv2
import mediapipe as mp
import time
import os

# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_ID = 1
MODEL_PATH = "hand_landmarker.task"

WIDTH = 640
HEIGHT = 480

# ============================================================
# MEDIA PIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================================================
# CONNEXIONS DE LA MAIN
# ============================================================

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),

    (0, 5), (5, 6), (6, 7), (7, 8),

    (5, 9), (9, 10), (10, 11), (11, 12),

    (9, 13), (13, 14), (14, 15), (15, 16),

    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

# ============================================================
# VERIFICATION DU MODELE
# ============================================================

if not os.path.exists(MODEL_PATH):
    print(f"❌ Modèle introuvable : {MODEL_PATH}")
    print("Place hand_landmarker.task dans le même dossier que ce script.")
    exit()

# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)

if not cap.isOpened():
    print("❌ Impossible d'ouvrir /dev/video0")
    exit()

# MJPG : format supporté par ta caméra
cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*"MJPG")
)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

print("✅ Caméra ouverte")
print("✅ MediaPipe initialisé")
print("👉 Appuie sur Q pour quitter")

# ============================================================
# BOUCLE PRINCIPALE
# ============================================================

previous_time = time.time()
timestamp_ms = 0

try:

    with HandLandmarker.create_from_options(options) as landmarker:

        while True:

            # ------------------------------------------------
            # CAPTURE
            # ------------------------------------------------

            success, frame = cap.read()

            if not success:
                print("⚠️ Frame impossible à lire")
                continue

            # Miroir
            frame = cv2.flip(frame, 1)

            height, width, _ = frame.shape

            # ------------------------------------------------
            # CONVERSION BGR -> RGB
            # ------------------------------------------------

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            # ------------------------------------------------
            # TIMESTAMP
            # ------------------------------------------------

            timestamp_ms = int(time.time() * 1000)

            # ------------------------------------------------
            # DETECTION DES MAINS
            # ------------------------------------------------

            results = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            # ------------------------------------------------
            # DESSIN
            # ------------------------------------------------

            if results.hand_landmarks:

                for hand_landmarks in results.hand_landmarks:

                    points = []

                    # Points
                    for landmark in hand_landmarks:

                        x = int(landmark.x * width)
                        y = int(landmark.y * height)

                        points.append((x, y))

                        cv2.circle(
                            frame,
                            (x, y),
                            5,
                            (0, 0, 255),
                            -1
                        )

                    # Connexions
                    for p1, p2 in HAND_CONNECTIONS:

                        cv2.line(
                            frame,
                            points[p1],
                            points[p2],
                            (0, 255, 0),
                            2
                        )

            # ------------------------------------------------
            # FPS
            # ------------------------------------------------

            current_time = time.time()

            fps = 1 / (current_time - previous_time)

            previous_time = current_time

            cv2.putText(
                frame,
                f"FPS: {int(fps)}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 255),
                2
            )

            # ------------------------------------------------
            # AFFICHAGE
            # ------------------------------------------------

            cv2.imshow(
                "Prince Vision - Hand Tracking",
                frame
            )

            # Q pour quitter
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

finally:

    print("Fermeture de la camera...")

    cap.release()
    cv2.destroyAllWindows()

    print("✅ Programme termine.")