import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from config import MODEL_PATH, MAX_NUM_FACES


class FaceTracker:
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=MAX_NUM_FACES,
            running_mode=vision.RunningMode.IMAGE,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.detector.detect(mp_image)
        return result.face_landmarks

    def detect_first(self, frame_bgr):
        faces = self.detect(frame_bgr)
        if faces:
            return faces[0]
        return None