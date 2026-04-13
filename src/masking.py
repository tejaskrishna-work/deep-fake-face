import cv2
import numpy as np
from config import (
    FACE_MASK_SHRINK,
    FACE_MASK_BLUR,
    OCCLUSION_EDGE_THRESHOLD1,
    OCCLUSION_EDGE_THRESHOLD2,
    OCCLUSION_DILATE,
    OCCLUSION_STRENGTH,
)

# Ordered face oval landmarks
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
    361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
    176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109
]


def create_face_region_mask(frame_shape, points, shrink=FACE_MASK_SHRINK, blur=FACE_MASK_BLUR):
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    poly = points[FACE_OVAL].astype(np.float32)
    center = np.mean(poly, axis=0)
    poly = (poly - center) * shrink + center
    poly = poly.astype(np.int32)

    cv2.fillPoly(mask, [poly], 255)

    k = blur if blur % 2 == 1 else blur + 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask


def estimate_boundary_occlusion_mask(frame_bgr, face_mask):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, OCCLUSION_EDGE_THRESHOLD1, OCCLUSION_EDGE_THRESHOLD2)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=OCCLUSION_DILATE)

    # Focus occlusion suppression near face boundary band
    eroded = cv2.erode(face_mask, kernel, iterations=8)
    boundary_band = cv2.subtract(face_mask, eroded)

    occ = cv2.bitwise_and(edges, boundary_band)
    occ = cv2.GaussianBlur(occ, (9, 9), 0)
    return occ


def refine_overlay_mask(overlay_mask, frame_bgr, points):
    face_mask = create_face_region_mask(frame_bgr.shape, points)
    final_mask = cv2.bitwise_and(overlay_mask, face_mask)

    occ = estimate_boundary_occlusion_mask(frame_bgr, face_mask)

    suppress = (occ.astype(np.float32) * OCCLUSION_STRENGTH).astype(np.uint8)
    final_mask = cv2.subtract(final_mask, suppress)

    final_mask = cv2.GaussianBlur(final_mask, (9, 9), 0)
    return final_mask, face_mask, occ