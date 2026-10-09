import cv2
from typing import Optional, Tuple, Union

class CameraStream:
    """Encapsulates OpenCV camera capture initialization, prioritizing explicit device selection."""
    
    def __init__(self, device_id: Union[int, str] = 2, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.cap = None

        print(f"🔍 Connexion au périphérique vidéo configuré (ID: {device_id})...")
        
        # Test direct du périphérique demandé
        if isinstance(device_id, int):
            self.cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
        else:
            self.cap = cv2.VideoCapture(device_id)

        # Vérification si le périphérique est ouvert et valide
        if not self.cap.isOpened() or self.cap.read()[1] is None:
            print(f"⚠️ Le périphérique {device_id} n'a pas répondu. Scan automatique des autres caméras...")
            if self.cap:
                self.cap.release()
            
            # Fallback : scanne les autres indices (0 à 4)
            for idx in [0, 1, 2, 3, 4]:
                if idx == device_id:
                    continue
                cap_test = cv2.VideoCapture(idx, cv2.CAP_V4L2)
                if cap_test.isOpened():
                    ret, frame = cap_test.read()
                    if ret and frame is not None:
                        self.cap = cap_test
                        print(f"✅ Caméra alternative trouvée sur /dev/video{idx}")
                        break
                    cap_test.release()

        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("❌ Impossible d'ouvrir une vidéo. Assure-toi que DroidCam est connecté.")

        # Configuration de la résolution et du format MJPG
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self) -> Tuple[bool, Optional[cv2.typing.MatLike]]:
        if not self.cap or not self.cap.isOpened():
            return False, None
        success, frame = self.cap.read()
        if not success or frame is None:
            return False, None
        return True, cv2.flip(frame, 1)

    def release(self) -> None:
        if self.cap and self.cap.isOpened():
            self.cap.release()