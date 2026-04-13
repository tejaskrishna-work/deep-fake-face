import cv2
import numpy as np


def alpha_blend(base_bgr, overlay_bgr, mask):
    alpha = mask.astype(np.float32) / 255.0
    alpha = np.expand_dims(alpha, axis=2)

    base = base_bgr.astype(np.float32)
    overlay = overlay_bgr.astype(np.float32)

    out = overlay * alpha + base * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def _masked_channel_stats(image, mask):
    vals = image[mask > 0]
    if vals.size == 0:
        return None, None
    mean = vals.mean(axis=0)
    std = vals.std(axis=0) + 1e-6
    return mean, std


def color_match_to_target(overlay_bgr, target_bgr, mask):
    src_mean, src_std = _masked_channel_stats(overlay_bgr, mask)
    tgt_mean, tgt_std = _masked_channel_stats(target_bgr, mask)

    if src_mean is None or tgt_mean is None:
        return overlay_bgr

    out = overlay_bgr.astype(np.float32)
    out = (out - src_mean) * (tgt_std / src_std) + tgt_mean
    return np.clip(out, 0, 255).astype(np.uint8)


def seamless_blend(base_bgr, overlay_bgr, mask):
    if mask is None or cv2.countNonZero(mask) < 50:
        return base_bgr

    x, y, w, h = cv2.boundingRect(mask)
    if w <= 0 or h <= 0:
        return base_bgr

    center = (x + w // 2, y + h // 2)

    try:
        return cv2.seamlessClone(
            overlay_bgr,
            base_bgr,
            mask,
            center,
            cv2.NORMAL_CLONE,
        )
    except cv2.error:
        return alpha_blend(base_bgr, overlay_bgr, mask)