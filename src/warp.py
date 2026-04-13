import cv2
import numpy as np
from utils import landmarks_to_pixels, scale_points_about_center
from config import LEFT_EYE_OUTER, RIGHT_EYE_OUTER, MOUTH_LEFT, MOUTH_RIGHT


def load_overlay(path):
    overlay = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if overlay is None:
        raise FileNotFoundError(f"Overlay image not found: {path}")

    if len(overlay.shape) == 2:
        alpha = np.full_like(overlay, 255)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGRA)
        overlay[:, :, 3] = alpha
        return overlay

    if overlay.shape[2] == 4:
        return overlay

    if overlay.shape[2] == 3:
        gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
        alpha = np.where(gray > 10, 255, 0).astype(np.uint8)
        bgra = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = alpha
        return bgra

    raise ValueError("Unsupported overlay image format")


def detect_overlay_landmarks(overlay_bgra, tracker):
    overlay_bgr = overlay_bgra[:, :, :3]
    face = tracker.detect_first(overlay_bgr)
    if face is None:
        return None

    h, w = overlay_bgr.shape[:2]
    return landmarks_to_pixels(face, w, h)


def build_warp_indices(total_landmarks=478):
    important = [
        1, 4, 6, 8, 10, 13, 14, 17, 33, 37, 40, 46, 52, 55, 61, 67, 70, 78,
        82, 84, 87, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133, 136, 144,
        145, 146, 148, 149, 150, 152, 154, 155, 157, 158, 159, 160, 161, 162,
        163, 172, 173, 176, 178, 181, 185, 191, 195, 197, 199, 205, 209, 213,
        221, 234, 246, 249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 291,
        295, 296, 297, 300, 308, 311, 312, 314, 317, 321, 323, 324, 332, 334,
        336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381,
        382, 384, 385, 386, 387, 388, 389, 390, 397, 400, 402, 405, 409, 415,
        454, 466,
    ]
    stepped = list(range(0, min(total_landmarks, 468), 6))
    merged = sorted(set(important + stepped))
    return merged


def build_delaunay_triangles(points, width, height):
    rect = (0, 0, width, height)
    subdiv = cv2.Subdiv2D(rect)

    for p in points:
        x = min(max(float(p[0]), 0.0), width - 1.0)
        y = min(max(float(p[1]), 0.0), height - 1.0)
        subdiv.insert((x, y))

    triangle_list = subdiv.getTriangleList()
    triangle_indices = []
    seen = set()

    for t in triangle_list:
        tri = np.array(
            [[t[0], t[1]], [t[2], t[3]], [t[4], t[5]]],
            dtype=np.float32
        )

        if np.any(tri[:, 0] < 0) or np.any(tri[:, 0] >= width):
            continue
        if np.any(tri[:, 1] < 0) or np.any(tri[:, 1] >= height):
            continue

        idxs = []
        for vertex in tri:
            dists = np.linalg.norm(points - vertex, axis=1)
            idx = int(np.argmin(dists))
            idxs.append(idx)

        if len(set(idxs)) != 3:
            continue

        key = tuple(sorted(idxs))
        if key in seen:
            continue
        seen.add(key)
        triangle_indices.append(tuple(idxs))

    return triangle_indices


def warp_triangle(src_bgra, dst_bgr, dst_mask, src_tri, dst_tri):
    src_rect = cv2.boundingRect(np.float32([src_tri]))
    dst_rect = cv2.boundingRect(np.float32([dst_tri]))

    x1, y1, w1, h1 = src_rect
    x2, y2, w2, h2 = dst_rect

    if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
        return

    src_tri_rect = np.array(
        [[src_tri[i][0] - x1, src_tri[i][1] - y1] for i in range(3)],
        dtype=np.float32
    )
    dst_tri_rect = np.array(
        [[dst_tri[i][0] - x2, dst_tri[i][1] - y2] for i in range(3)],
        dtype=np.float32
    )

    src_patch = src_bgra[y1:y1 + h1, x1:x1 + w1]
    if src_patch.size == 0:
        return

    warp_mat = cv2.getAffineTransform(src_tri_rect, dst_tri_rect)
    warped_patch = cv2.warpAffine(
        src_patch,
        warp_mat,
        (w2, h2),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    tri_mask = np.zeros((h2, w2), dtype=np.uint8)
    cv2.fillConvexPoly(tri_mask, np.int32(dst_tri_rect), 255)

    warped_rgb = warped_patch[:, :, :3]
    warped_alpha = warped_patch[:, :, 3]
    tri_alpha = cv2.bitwise_and(warped_alpha, tri_mask)

    roi = dst_bgr[y2:y2 + h2, x2:x2 + w2]
    roi_mask = dst_mask[y2:y2 + h2, x2:x2 + w2]

    alpha = tri_alpha.astype(np.float32) / 255.0
    alpha_3 = alpha[:, :, None]

    roi[:] = (
        warped_rgb.astype(np.float32) * alpha_3 +
        roi.astype(np.float32) * (1.0 - alpha_3)
    ).astype(np.uint8)

    roi_mask[:] = np.maximum(roi_mask, tri_alpha)


class AffineOverlayWarper:
    def __init__(self, overlay_path, tracker):
        self.overlay_bgra = load_overlay(overlay_path)
        self.overlay_points_full = detect_overlay_landmarks(self.overlay_bgra, tracker)

        self.use_landmark_overlay = self.overlay_points_full is not None

        h, w = self.overlay_bgra.shape[:2]
        if self.use_landmark_overlay:
            self.src_left_eye = self.overlay_points_full[LEFT_EYE_OUTER]
            self.src_right_eye = self.overlay_points_full[RIGHT_EYE_OUTER]
            self.src_mouth_mid = (
                self.overlay_points_full[MOUTH_LEFT] + self.overlay_points_full[MOUTH_RIGHT]
            ) / 2.0
        else:
            self.src_left_eye = np.array([0.30 * w, 0.38 * h], dtype=np.float32)
            self.src_right_eye = np.array([0.70 * w, 0.38 * h], dtype=np.float32)
            self.src_mouth_mid = np.array([0.50 * w, 0.72 * h], dtype=np.float32)

    def render(self, frame_shape, dst_points_full, scale=1.0):
        frame_h, frame_w = frame_shape[:2]

        src_pts = np.array(
            [self.src_left_eye, self.src_right_eye, self.src_mouth_mid],
            dtype=np.float32
        )

        src_center = np.mean(src_pts, axis=0)
        src_pts = scale_points_about_center(src_pts, src_center, scale)

        dst_left_eye = dst_points_full[LEFT_EYE_OUTER]
        dst_right_eye = dst_points_full[RIGHT_EYE_OUTER]
        dst_mouth_mid = (dst_points_full[MOUTH_LEFT] + dst_points_full[MOUTH_RIGHT]) / 2.0

        dst_pts = np.array(
            [dst_left_eye, dst_right_eye, dst_mouth_mid],
            dtype=np.float32
        )

        matrix = cv2.getAffineTransform(src_pts, dst_pts)
        warped_bgra = cv2.warpAffine(
            self.overlay_bgra,
            matrix,
            (frame_w, frame_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        warped_bgr = warped_bgra[:, :, :3]
        warped_mask = warped_bgra[:, :, 3]
        warped_mask = cv2.GaussianBlur(warped_mask, (9, 9), 0)

        return warped_bgr, warped_mask


class PiecewiseOverlayWarper:
    def __init__(self, overlay_path, tracker):
        self.overlay_bgra = load_overlay(overlay_path)
        self.src_points_full = detect_overlay_landmarks(self.overlay_bgra, tracker)
        self.available = self.src_points_full is not None

        if not self.available:
            self.indices = []
            self.src_points = None
            self.triangles = []
            return

        self.indices = build_warp_indices(len(self.src_points_full))
        self.src_points = self.src_points_full[self.indices]

        oh, ow = self.overlay_bgra.shape[:2]
        self.triangles = build_delaunay_triangles(self.src_points, ow, oh)

        if not self.triangles:
            self.available = False

    def render(self, frame_shape, dst_points_full, scale=1.0):
        if not self.available:
            raise RuntimeError("Triangle warper not available for this overlay.")

        frame_h, frame_w = frame_shape[:2]

        dst_points = dst_points_full[self.indices].copy()
        center = np.mean(dst_points, axis=0)
        dst_points = scale_points_about_center(dst_points, center, scale)

        warped_bgr = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        warped_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)

        for tri in self.triangles:
            src_tri = np.float32([
                self.src_points[tri[0]],
                self.src_points[tri[1]],
                self.src_points[tri[2]],
            ])
            dst_tri = np.float32([
                dst_points[tri[0]],
                dst_points[tri[1]],
                dst_points[tri[2]],
            ])

            warp_triangle(
                self.overlay_bgra,
                warped_bgr,
                warped_mask,
                src_tri,
                dst_tri,
            )

        warped_mask = cv2.GaussianBlur(warped_mask, (9, 9), 0)
        return warped_bgr, warped_mask


class HybridOverlayWarper:
    def __init__(self, overlay_path, tracker):
        self.affine = AffineOverlayWarper(overlay_path, tracker)
        self.triangle = PiecewiseOverlayWarper(overlay_path, tracker)

    def triangle_available(self):
        return self.triangle.available

    def overlay_landmarks_available(self):
        return self.affine.use_landmark_overlay

    def render(self, frame_shape, dst_points_full, scale=1.0, mode="auto"):
        if mode == "triangle":
            if not self.triangle.available:
                raise RuntimeError("Triangle mode requested but overlay is not suitable.")
            return self.triangle.render(frame_shape, dst_points_full, scale=scale)

        if mode == "affine":
            return self.affine.render(frame_shape, dst_points_full, scale=scale)

        if self.triangle.available:
            return self.triangle.render(frame_shape, dst_points_full, scale=scale)

        return self.affine.render(frame_shape, dst_points_full, scale=scale)