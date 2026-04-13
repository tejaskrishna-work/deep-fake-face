import cv2
from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT


class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Failed to read frame from webcam")
        frame = cv2.flip(frame, 1)
        return frame

    def release(self):
        self.cap.release()