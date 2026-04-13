from config import (
    LEFT_EYE_OUTER,
    RIGHT_EYE_OUTER,
    MOUTH_LEFT,
    MOUTH_RIGHT,
    NOSE_TIP,
    CHIN,
)


def get_anchor_points(points):
    return {
        "left_eye": points[LEFT_EYE_OUTER],
        "right_eye": points[RIGHT_EYE_OUTER],
        "mouth_left": points[MOUTH_LEFT],
        "mouth_right": points[MOUTH_RIGHT],
        "nose": points[NOSE_TIP],
        "chin": points[CHIN],
    }