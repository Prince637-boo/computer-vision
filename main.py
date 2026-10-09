import cv2
from src.config import CAMERA_ID, FRAME_WIDTH, FRAME_HEIGHT, MODEL_PATH
from src.vision.camera import CameraStream
from src.vision.hand_detector import HandDetector
from src.vision.drawer import Visualizer
from src.core.utils import FPSCounter

def main():
    print(" Initializing Hand Tracking Pipeline...")
    
    camera = CameraStream(CAMERA_ID, FRAME_WIDTH, FRAME_HEIGHT)
    detector = HandDetector(str(MODEL_PATH))
    fps_counter = FPSCounter()

    print(" Pipeline ready. Press 'Q' to quit.")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print(" Frame capture failed.")
                continue

            h, w, _ = frame.shape

            # Detection & Processing
            results = detector.process_frame(frame)
            hands_landmarks = detector.extract_landmarks(results, w, h)

            # Rendering
            Visualizer.draw_hands(frame, hands_landmarks)
            Visualizer.draw_fps(frame, fps_counter.update())

            cv2.imshow("Prince Vision - Modular Pipeline", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print(" Cleaning up resources...")
        detector.close()
        camera.release()
        cv2.destroyAllWindows()
        print("✅ Shutdown complete.")

if __name__ == "__main__":
    main()