import cv2
from src.config import CAMERA_ID, FRAME_WIDTH, FRAME_HEIGHT, MODEL_PATH
from src.vision.camera import CameraStream
from src.vision.hand_detector import HandDetector
from src.vision.gestures import GestureAnalyzer, GestureType
from src.interactor.panda_engine import Panda3DEngine

def main():
    print("🚀 Démarrage du système Jarvis Hologram AR (Mercedes-Benz)...")

    # 1. Moteur 3D Panda3D
    app = Panda3DEngine()
    app.load_car_model("assets/models/mercedes_benz.glb")

    # 2. Pipeline de vision
    camera = CameraStream(CAMERA_ID, FRAME_WIDTH, FRAME_HEIGHT)
    detector = HandDetector(str(MODEL_PATH))
    gesture_analyzer = GestureAnalyzer()

    # 3. Tâche de traitement en temps réel
    def vision_task(task):
        success, frame = camera.read()
        if not success or frame is None:
            return task.cont

        h, w, _ = frame.shape

        # Tracking des mains
        results = detector.process_frame(frame)
        hands_landmarks = detector.extract_landmarks(results, w, h)

        is_pinching = False
        hand_pos = (0.0, 0.0, 0.0)

        if len(hands_landmarks) > 0:
            gesture_info = gesture_analyzer.analyze_hand(hands_landmarks[0], w, h)
            is_pinching = (gesture_info["gesture"] == GestureType.PINCH)
            hand_pos = gesture_info["normalized_pos"]

        # Mise à jour AR et interaction 3D
        app.update_background(frame)
        app.update_hand_interaction(hand_pos, is_pinching)

        return task.cont

    app.taskMgr.add(vision_task, "VisionTask")
    app.run()

if __name__ == "__main__":
    main()