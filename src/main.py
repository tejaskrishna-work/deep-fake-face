import cv2
import time
from camera import Camera
from tracker import FaceTracker
from warp import HybridOverlayWarper
from blend import seamless_blend, color_match_to_target
from pose import get_anchor_points
from utils import (
    landmarks_to_pixels,
    to_int_tuple,
    ensure_dir,
    timestamp_str,
)
from stability import LandmarkStabilizer
from masking import refine_overlay_mask
from overlay_manager import OverlayManager
from benchmark import BenchmarkLogger
from config import (
    WINDOW_NAME,
    CONTROL_WINDOW_NAME,
    DEFAULT_SHOW_LANDMARKS,
    DEFAULT_SHOW_ANCHORS,
    DEFAULT_SHOW_OVERLAY,
    DEFAULT_OVERLAY_SCALE,
    DEFAULT_WARP_MODE,
    DEFAULT_COMPARE_MODE,
    MIN_SCALE_PERCENT,
    MAX_SCALE_PERCENT,
    OVERLAYS_DIR,
    DEFAULT_OVERLAY_FILENAME,
)


recording = False
video_writer = None


def nothing(_):
    pass


def setup_controls():
    cv2.namedWindow(CONTROL_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROL_WINDOW_NAME, 620, 420)

    cv2.createTrackbar("Show Green Landmarks", CONTROL_WINDOW_NAME, DEFAULT_SHOW_LANDMARKS, 1, nothing)
    cv2.createTrackbar("Show Red Anchors", CONTROL_WINDOW_NAME, DEFAULT_SHOW_ANCHORS, 1, nothing)
    cv2.createTrackbar("Show Overlay", CONTROL_WINDOW_NAME, DEFAULT_SHOW_OVERLAY, 1, nothing)
    cv2.createTrackbar("Overlay Scale %", CONTROL_WINDOW_NAME, DEFAULT_OVERLAY_SCALE, MAX_SCALE_PERCENT, nothing)
    cv2.createTrackbar("Warp Mode", CONTROL_WINDOW_NAME, DEFAULT_WARP_MODE, 2, nothing)
    cv2.createTrackbar("Compare Mode", CONTROL_WINDOW_NAME, DEFAULT_COMPARE_MODE, 1, nothing)

    cv2.createTrackbar("Show HUD", CONTROL_WINDOW_NAME, 1, 1, nothing)
    cv2.createTrackbar("HUD in Screenshot", CONTROL_WINDOW_NAME, 0, 1, nothing)
    cv2.createTrackbar("HUD in Recording", CONTROL_WINDOW_NAME, 0, 1, nothing)

    cv2.setTrackbarMin("Overlay Scale %", CONTROL_WINDOW_NAME, MIN_SCALE_PERCENT)


def get_controls():
    show_landmarks = cv2.getTrackbarPos("Show Green Landmarks", CONTROL_WINDOW_NAME)
    show_anchors = cv2.getTrackbarPos("Show Red Anchors", CONTROL_WINDOW_NAME)
    show_overlay = cv2.getTrackbarPos("Show Overlay", CONTROL_WINDOW_NAME)
    overlay_scale_percent = cv2.getTrackbarPos("Overlay Scale %", CONTROL_WINDOW_NAME)
    warp_mode_value = cv2.getTrackbarPos("Warp Mode", CONTROL_WINDOW_NAME)
    compare_mode_value = cv2.getTrackbarPos("Compare Mode", CONTROL_WINDOW_NAME)

    show_hud = cv2.getTrackbarPos("Show HUD", CONTROL_WINDOW_NAME)
    hud_in_screenshot = cv2.getTrackbarPos("HUD in Screenshot", CONTROL_WINDOW_NAME)
    hud_in_recording = cv2.getTrackbarPos("HUD in Recording", CONTROL_WINDOW_NAME)

    if overlay_scale_percent < MIN_SCALE_PERCENT:
        overlay_scale_percent = MIN_SCALE_PERCENT

    warp_mode = "auto"
    if warp_mode_value == 1:
        warp_mode = "affine"
    elif warp_mode_value == 2:
        warp_mode = "triangle"

    compare_mode = "off"
    if compare_mode_value == 1:
        compare_mode = "split"

    return {
        "show_landmarks": show_landmarks,
        "show_anchors": show_anchors,
        "show_overlay": show_overlay,
        "overlay_scale": overlay_scale_percent / 100.0,
        "warp_mode": warp_mode,
        "compare_mode": compare_mode,
        "show_hud": show_hud,
        "hud_in_screenshot": hud_in_screenshot,
        "hud_in_recording": hud_in_recording,
    }


def toggle_trackbar(name):
    value = cv2.getTrackbarPos(name, CONTROL_WINDOW_NAME)
    cv2.setTrackbarPos(name, CONTROL_WINDOW_NAME, 0 if value else 1)


def adjust_scale(delta):
    value = cv2.getTrackbarPos("Overlay Scale %", CONTROL_WINDOW_NAME)
    value = max(MIN_SCALE_PERCENT, min(MAX_SCALE_PERCENT, value + delta))
    cv2.setTrackbarPos("Overlay Scale %", CONTROL_WINDOW_NAME, value)


def adjust_warp_mode():
    value = cv2.getTrackbarPos("Warp Mode", CONTROL_WINDOW_NAME)
    value = (value + 1) % 3
    cv2.setTrackbarPos("Warp Mode", CONTROL_WINDOW_NAME, value)


def adjust_compare_mode():
    value = cv2.getTrackbarPos("Compare Mode", CONTROL_WINDOW_NAME)
    value = (value + 1) % 2
    cv2.setTrackbarPos("Compare Mode", CONTROL_WINDOW_NAME, value)


def warp_mode_label(mode):
    if mode == "triangle":
        return "triangle"
    if mode == "affine":
        return "affine"
    return "auto"


def init_writer(frame_shape):
    global video_writer
    ensure_dir("recordings")

    h, w = frame_shape[:2]
    filename = f"recordings/recording_{timestamp_str()}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (w, h))
    return filename


def release_writer():
    global video_writer
    if video_writer is not None:
        video_writer.release()
        video_writer = None


def draw_status(output, fps, recording_on, overlay_scale, warp_mode, compare_mode, overlay_info):
    cv2.putText(output, f"FPS: {int(fps)}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(output, f"Scale: {int(overlay_scale * 100)}%", (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(output, f"Warp: {warp_mode_label(warp_mode)}", (12, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
    cv2.putText(output, f"Compare: {compare_mode}", (12, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 255, 180), 2)
    cv2.putText(
        output,
        f"Overlay: {overlay_info['name']}",
        (12, 158),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 220, 180),
        2,
    )
    cv2.putText(
        output,
        f"Overlay-face: {'yes' if overlay_info['overlay_face_ok'] else 'no'}  Triangle-ready: {'yes' if overlay_info['triangle_ok'] else 'no'}",
        (12, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 255, 255),
        2,
    )

    if recording_on:
        cv2.putText(output, "REC", (12, 222), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(
        output,
        "Keys: n/p overlay  c compare  m warp  g/r/o overlays  +/- scale  s shot  v rec  q quit",
        (12, output.shape[0] - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )


def render_pipeline(frame, points, warper, controls, show_landmarks, show_anchors):
    output = frame.copy()

    if controls["show_overlay"]:
        try:
            overlay_bgr, overlay_mask = warper.render(
                frame.shape,
                points,
                scale=controls["overlay_scale"],
                mode=controls["warp_mode"],
            )
        except RuntimeError:
            overlay_bgr, overlay_mask = warper.render(
                frame.shape,
                points,
                scale=controls["overlay_scale"],
                mode="affine",
            )

        overlay_mask, _, _ = refine_overlay_mask(overlay_mask, frame, points)
        overlay_bgr = color_match_to_target(overlay_bgr, frame, overlay_mask)
        output = seamless_blend(output, overlay_bgr, overlay_mask)

    anchors = get_anchor_points(points)

    if show_landmarks:
        for pt in points:
            cv2.circle(output, to_int_tuple(pt), 1, (0, 255, 0), -1)

    if show_anchors:
        for pt in anchors.values():
            cv2.circle(output, to_int_tuple(pt), 4, (0, 0, 255), -1)

    return output


def make_compare_frame(left_frame, right_frame):
    h = min(left_frame.shape[0], right_frame.shape[0])
    w = min(left_frame.shape[1], right_frame.shape[1])

    left = cv2.resize(left_frame, (w, h))
    right = cv2.resize(right_frame, (w, h))

    combined = cv2.hconcat([left, right])

    cv2.putText(combined, "LEFT: affine baseline", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(combined, "RIGHT: stabilized hybrid", (w + 18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.line(combined, (w, 0), (w, h), (255, 255, 255), 2)
    return combined


def make_display_and_export_frames(base_frame, fps, recording_on, controls, overlay_info):
    display_frame = base_frame.copy()
    screenshot_frame = base_frame.copy()
    recording_frame = base_frame.copy()

    if controls["show_hud"]:
        draw_status(
            display_frame,
            fps,
            recording_on,
            controls["overlay_scale"],
            controls["warp_mode"],
            controls["compare_mode"],
            overlay_info,
        )

    if controls["hud_in_screenshot"]:
        draw_status(
            screenshot_frame,
            fps,
            recording_on,
            controls["overlay_scale"],
            controls["warp_mode"],
            controls["compare_mode"],
            overlay_info,
        )

    if controls["hud_in_recording"]:
        draw_status(
            recording_frame,
            fps,
            recording_on,
            controls["overlay_scale"],
            controls["warp_mode"],
            controls["compare_mode"],
            overlay_info,
        )

    return display_frame, screenshot_frame, recording_frame


def main():
    global recording, video_writer

    ensure_dir("captures")
    ensure_dir("recordings")
    ensure_dir("reports")

    cam = Camera()
    tracker = FaceTracker()
    overlay_manager = OverlayManager(OVERLAYS_DIR, tracker, DEFAULT_OVERLAY_FILENAME)
    stabilizer = LandmarkStabilizer()
    logger = BenchmarkLogger("reports")

    print("Overlay scan results:")
    for item in overlay_manager.scan_all():
        print(
            f"- {item['name']}: "
            f"overlay_face_ok={item['overlay_face_ok']} "
            f"triangle_ok={item['triangle_ok']}"
        )

    prev_time = time.perf_counter()
    frame_index = 0

    cv2.namedWindow(WINDOW_NAME)
    setup_controls()

    last_screenshot_frame = None

    try:
        while True:
            frame = cam.read()
            controls = get_controls()
            warper = overlay_manager.get_warper()
            overlay_info = overlay_manager.describe_current()

            faces = tracker.detect(frame)
            tracked = len(faces) > 0

            if tracked:
                raw_points = landmarks_to_pixels(faces[0], frame.shape[1], frame.shape[0])
                stabilized_points = stabilizer.update(raw_points)

                hybrid_frame = render_pipeline(
                    frame,
                    stabilized_points,
                    warper,
                    controls,
                    controls["show_landmarks"],
                    controls["show_anchors"],
                )

                baseline_controls = dict(controls)
                baseline_controls["warp_mode"] = "affine"

                baseline_frame = render_pipeline(
                    frame,
                    raw_points,
                    warper,
                    baseline_controls,
                    0,
                    0,
                )

                if controls["compare_mode"] == "split":
                    base_output = make_compare_frame(baseline_frame, hybrid_frame)
                else:
                    base_output = hybrid_frame
            else:
                base_output = frame.copy()

            curr_time = time.perf_counter()
            fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            display_output, screenshot_output, recording_output = make_display_and_export_frames(
                base_output,
                fps,
                recording,
                controls,
                overlay_info,
            )

            last_screenshot_frame = screenshot_output

            logger.log(
                frame_index=frame_index,
                fps=fps,
                tracked=tracked,
                motion=stabilizer.last_motion,
                smoothing_alpha=stabilizer.last_alpha,
                overlay_name=overlay_info["name"],
                warp_mode=controls["warp_mode"],
                compare_mode=controls["compare_mode"],
                overlay_face_ok=overlay_info["overlay_face_ok"],
                triangle_ok=overlay_info["triangle_ok"],
            )
            frame_index += 1

            if recording and video_writer is not None:
                video_writer.write(recording_output)

            cv2.imshow(WINDOW_NAME, display_output)

            key = cv2.waitKey(1) & 0xFF

            if key == 27 or key == ord("q"):
                break
            elif key == ord("g"):
                toggle_trackbar("Show Green Landmarks")
            elif key == ord("r"):
                toggle_trackbar("Show Red Anchors")
            elif key == ord("o"):
                toggle_trackbar("Show Overlay")
            elif key == ord("+") or key == ord("="):
                adjust_scale(5)
            elif key == ord("-") or key == ord("_"):
                adjust_scale(-5)
            elif key == ord("m"):
                adjust_warp_mode()
            elif key == ord("c"):
                adjust_compare_mode()
            elif key == ord("h"):
                toggle_trackbar("Show HUD")
            elif key == ord("j"):
                toggle_trackbar("HUD in Screenshot")
            elif key == ord("k"):
                toggle_trackbar("HUD in Recording")
            elif key == ord("n"):
                name = overlay_manager.next_overlay()
                print(f"Switched overlay -> {name}")
            elif key == ord("p"):
                name = overlay_manager.prev_overlay()
                print(f"Switched overlay -> {name}")
            elif key == ord("s"):
                if last_screenshot_frame is not None:
                    filename = f"captures/capture_{timestamp_str()}.png"
                    cv2.imwrite(filename, last_screenshot_frame)
                    print(f"Saved screenshot: {filename}")
            elif key == ord("v"):
                recording = not recording
                if recording:
                    filename = init_writer(base_output.shape)
                    print(f"Recording started: {filename}")
                else:
                    release_writer()
                    print("Recording stopped")

    finally:
        release_writer()
        cam.release()
        cv2.destroyAllWindows()

        summary = logger.summary()
        logger.close()

        print("\nSession summary")
        print(f"Benchmark CSV: {summary['file']}")
        print(f"Rows written: {summary['rows_written']}")
        print(f"Average FPS: {summary['avg_fps']:.2f}")
        print(f"Tracking success rate: {summary['tracking_rate'] * 100:.2f}%")
        print(f"Triangle-capable frame rate: {summary['triangle_rate'] * 100:.2f}%")
        print("Done.")


if __name__ == "__main__":
    main()