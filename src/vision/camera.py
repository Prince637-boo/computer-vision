import cv2
from typing import Optional, Tuple, Union

class CameraStream:
    """Encapsulates OpenCV camera capture initialization and frame retrieval."""
    
    def __init__(self, device_id: Union[int, str], width: int = 640, height: int = 480):
        self.device_id = device_id
        self.width = width
        self.height = height
        
        if isinstance(device_id, str):
            self.cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
        else:
            self.cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video capture device: {device_id}")

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self) -> Tuple[bool, Optional[cv2.typing.MatLike]]:
        success, frame = self.cap.read()
        if not success or frame is None:
            return False, None
        return True, cv2.flip(frame, 1)

    def release(self) -> None:
        if self.cap and self.cap.isOpened():
            self.cap.release()