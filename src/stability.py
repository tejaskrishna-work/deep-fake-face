import cv2
import numpy as np
from config import (
    SMOOTHING_MIN_ALPHA,
    SMOOTHING_MAX_ALPHA,
    SMOOTHING_MOTION_SCALE,
    POSE_BLEND,
    LEFT_EYE_OUTER,
    RIGHT_EYE_OUTER,
    NOSE_TIP,
    MOUTH_LEFT,
    MOUTH_RIGHT,
    CHIN,
)


class LandmarkStabilizer:
    def __init__(self):
        self.prev_raw = None
        self.prev_smoothed = None
        self.prev_anchor_smoothed = None

        self.anchor_indices = [
            LEFT_EYE_OUTER,
            RIGHT_EYE_OUTER,
            NOSE_TIP,
            MOUTH_LEFT,
            MOUTH_RIGHT,
            CHIN,
        ]

        self.last_alpha = SMOOTHING_MIN_ALPHA
        self.last_motion = 0.0

    def _compute_alpha(self, points):
        if self.prev_raw is None:
            return SMOOTHING_MAX_ALPHA, 0.0

        motion = np.mean(np.linalg.norm(points - self.prev_raw, axis=1))
        alpha = SMOOTHING_MIN_ALPHA + (motion / SMOOTHING_MOTION_SCALE) * (
            SMOOTHING_MAX_ALPHA - SMOOTHING_MIN_ALPHA
        )
        alpha = float(np.clip(alpha, SMOOTHING_MIN_ALPHA, SMOOTHING_MAX_ALPHA))
        return alpha, float(motion)

    def update(self, points):
        points = points.astype(np.float32)

        alpha, motion = self._compute_alpha(points)

        if self.prev_smoothed is None:
            point_smoothed = points.copy()
        else:
            point_smoothed = alpha * points + (1.0 - alpha) * self.prev_smoothed

        current_anchors = points[self.anchor_indices]

        if self.prev_anchor_smoothed is None:
            anchor_smoothed = current_anchors.copy()
        else:
            anchor_alpha = min(0.55, alpha + 0.08)
            anchor_smoothed = (
                anchor_alpha * current_anchors
                + (1.0 - anchor_alpha) * self.prev_anchor_smoothed
            )

        stabilized_by_pose = point_smoothed.copy()

        # Estimate a global similarity-like transform from current anchors to smoothed anchors
        matrix, _ = cv2.estimateAffinePartial2D(
            current_anchors,
            anchor_smoothed,
            method=cv2.LMEDS,
        )

        if matrix is not None:
            transformed = cv2.transform(points[None, :, :], matrix)[0]
            stabilized_by_pose = (
                (1.0 - POSE_BLEND) * point_smoothed + POSE_BLEND * transformed
            ).astype(np.float32)

        self.prev_raw = points.copy()
        self.prev_smoothed = stabilized_by_pose.copy()
        self.prev_anchor_smoothed = anchor_smoothed.copy()

        self.last_alpha = alpha
        self.last_motion = motion

        return stabilized_by_pose