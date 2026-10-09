import cv2
import time
from typing import Dict, Any, Optional
from src.config import CAMERA_ID, FRAME_WIDTH, FRAME_HEIGHT, MODEL_PATH
from src.vision.camera import CameraStream
from src.vision.hand_detector import HandDetector
from src.vision.gestures import GestureAnalyzer, GestureType
from src.vision.drawer import Visualizer
from src.core.utils import FPSCounter
from src.interactor.engine_3d import Scene3DEngine

def main():
    print("Initializing Integrated Vision-3D Pipeline...")
    
    # Initialize all modules
    camera = CameraStream(CAMERA_ID, FRAME_WIDTH, FRAME_HEIGHT)
    detector = HandDetector(str(MODEL_PATH))
    gesture_analyzer = GestureAnalyzer(pinch_threshold_pixels=35.0)
    
    # Initialize PyVista Scene
    engine_3d = Scene3DEngine()
    fps_counter = FPSCounter()

    print("Pipeline ready. Use PINCH gesture near the object to grab.")
    print("Press 'Q' on the OpenCV window or Ctrl+C in terminal to quit.")

    try:
        while engine_3d.is_interaction_visible():
            success, frame = camera.read()
            if not success or frame is None:
                continue

            h, w, _ = frame.shape

            # 1. Hand Detection
            results = detector.process_frame(frame)
            hands_landmarks = detector.extract_landmarks(results, w, h)

            # 2. Render Hand Skeleton
            Visualizer.draw_hands(frame, hands_landmarks)

            gesture_info: Optional[Dict[str, Any]] = None
            if len(hands_landmarks) > 0:
                # Analyze primary hand
                gesture_info = gesture_analyzer.analyze_hand(hands_landmarks[0], w, h)
                Visualizer.draw_gesture_info(frame, gesture_info)

            # 3. Process 3D Interaction
            if gesture_info:
                engine_3d.update_interaction_point(
                    gesture_info["normalized_pos"], 
                    is_active=True
                )
                engine_3d.process_grab_logic(
                    gesture_info["gesture"] == GestureType.PINCH,
                    gesture_info["normalized_pos"]
                )
            else:
                engine_3d.update_interaction_point((0, 0, 0), is_active=False)

            # 4. Update Windows
            Visualizer.draw_fps(frame, fps_counter.update())
            cv2.imshow("Prince Vision - Computer Vision Feed", frame)
            
            # Update PyVista rendering loop
            engine_3d.render_update()

            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user.")

    finally:
        print("🧹 Cleaning up Vision-3D resources...")
        engine_3d.close()
        detector.close()
        camera.release()
        cv2.destroyAllWindows()
        print("✅ Shutdown complete.")

if __name__ == "__main__":
    main()