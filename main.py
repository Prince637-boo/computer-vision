from src.config import (
    CAMERA_ID,
    CAR_MODEL_PATH,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MODEL_PATH,
    NUM_HANDS,
)
from src.vision.camera import CameraStream
from src.vision.hand_detector import HandDetector
from src.vision.gestures import GestureAnalyzer, GestureType
from src.vision.worker import VisionWorker
from src.interactor.panda_engine import Panda3DEngine


def main():
    print("Starting the Mercedes-Benz holographic vehicle lab...")

    app = Panda3DEngine()
    app.load_car_model(str(CAR_MODEL_PATH))

    camera = CameraStream(CAMERA_ID, FRAME_WIDTH, FRAME_HEIGHT)
    detector = HandDetector(str(MODEL_PATH), num_hands=NUM_HANDS)
    gesture_analyzer = GestureAnalyzer()
    vision_worker = VisionWorker(camera, detector, gesture_analyzer)
    last_background_frame = None

    def vision_task(task):
        nonlocal last_background_frame
        frame, hands_info = vision_worker.snapshot()
        if frame is not None and frame is not last_background_frame:
            app.update_background(frame)
            last_background_frame = frame
        if hands_info:
            primary_hand = hands_info[0]
            both_palms_open = (
                len(hands_info) > 1
                and primary_hand["gesture"] == GestureType.OPEN_PALM
                and hands_info[1]["gesture"] == GestureType.OPEN_PALM
            )
            second_hand_pos = hands_info[1]["wrist_pos"] if both_palms_open else None
            app.update_hand_interaction(
                primary_hand["normalized_pos"],
                primary_hand["gesture"] == GestureType.PINCH,
                second_hand_norm_pos=second_hand_pos,
                gesture=primary_hand["gesture"],
                palm_norm_pos=primary_hand["palm_pos"],
                primary_wrist_norm_pos=primary_hand["wrist_pos"],
                hand_sample_time=primary_hand["detected_at"],
            )
        else:
            app.update_hand_interaction(
                (0.0, 0.0),
                False,
                gesture=GestureType.NONE,
            )

        return task.cont

    app.taskMgr.add(vision_task, "VisionTask")
    vision_worker.start()
    try:
        app.run()
    finally:
        try:
            vision_worker.stop()
        finally:
            detector.close()
            app.destroy()

if __name__ == "__main__":
    main()