import os
import time
import numpy as np


def to_int_tuple(pt):
    return int(pt[0]), int(pt[1])


def landmarks_to_pixels(landmarks, image_width, image_height):
    pts = []
    for lm in landmarks:
        pts.append([lm.x * image_width, lm.y * image_height])
    return np.array(pts, dtype=np.float32)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def timestamp_str():
    return time.strftime("%Y%m%d_%H%M%S")


def scale_points_about_center(points, center, scale):
    return (points - center) * scale + center


def clamp(value, lo, hi):
    return max(lo, min(hi, value))