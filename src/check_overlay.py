import sys
import cv2
from tracker import FaceTracker
from warp import load_overlay, detect_overlay_landmarks, build_warp_indices, build_delaunay_triangles


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/check_overlay.py assets/overlays/mask.png")
        return

    image_path = sys.argv[1]

    tracker = FaceTracker()
    overlay_bgra = load_overlay(image_path)

    h, w = overlay_bgra.shape[:2]
    channels = overlay_bgra.shape[2] if len(overlay_bgra.shape) == 3 else 1

    print(f"Image: {image_path}")
    print(f"Shape: {overlay_bgra.shape}")
    print(f"Width x Height: {w} x {h}")
    print(f"Channels: {channels}")

    points_full = detect_overlay_landmarks(overlay_bgra, tracker)

    if points_full is None:
        print("Result: NOT acceptable for triangle warping.")
        print("Reason: No face detected in overlay image.")
        return

    print(f"Face detected: yes")
    print(f"Landmark count: {len(points_full)}")

    indices = build_warp_indices(len(points_full))
    reduced = points_full[indices]
    triangles = build_delaunay_triangles(reduced, w, h)

    print(f"Reduced landmark count used for warp: {len(reduced)}")
    print(f"Delaunay triangle count: {len(triangles)}")

    if len(triangles) == 0:
        print("Result: NOT acceptable for triangle warping.")
        print("Reason: Could not build valid triangle mesh.")
        return

    print("Result: ACCEPTABLE for triangle warping.")

    preview = overlay_bgra[:, :, :3].copy()
    for pt in reduced:
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(preview, (x, y), 1, (0, 255, 0), -1)

    cv2.imshow("Overlay Check Preview", preview)
    print("Press any key in the preview window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()