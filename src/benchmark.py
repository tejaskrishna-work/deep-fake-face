import csv
import os
from utils import ensure_dir, timestamp_str


class BenchmarkLogger:
    def __init__(self, reports_dir="reports"):
        ensure_dir(reports_dir)
        self.reports_dir = reports_dir
        self.filename = os.path.join(
            reports_dir,
            f"benchmark_{timestamp_str()}.csv"
        )
        self.rows_written = 0
        self.total_frames = 0
        self.tracked_frames = 0
        self.triangle_frames = 0
        self.sum_fps = 0.0

        self.file = open(self.filename, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            "frame_index",
            "fps",
            "tracked",
            "motion",
            "smoothing_alpha",
            "overlay_name",
            "warp_mode",
            "compare_mode",
            "overlay_face_ok",
            "triangle_ok",
        ])

    def log(
        self,
        frame_index,
        fps,
        tracked,
        motion,
        smoothing_alpha,
        overlay_name,
        warp_mode,
        compare_mode,
        overlay_face_ok,
        triangle_ok,
    ):
        self.writer.writerow([
            frame_index,
            round(float(fps), 4),
            int(bool(tracked)),
            round(float(motion), 6),
            round(float(smoothing_alpha), 6),
            overlay_name,
            warp_mode,
            compare_mode,
            int(bool(overlay_face_ok)),
            int(bool(triangle_ok)),
        ])

        self.rows_written += 1
        self.total_frames += 1
        self.sum_fps += float(fps)

        if tracked:
            self.tracked_frames += 1
        if triangle_ok:
            self.triangle_frames += 1

    def summary(self):
        avg_fps = self.sum_fps / self.total_frames if self.total_frames else 0.0
        tracking_rate = self.tracked_frames / self.total_frames if self.total_frames else 0.0
        triangle_rate = self.triangle_frames / self.total_frames if self.total_frames else 0.0

        return {
            "file": self.filename,
            "rows_written": self.rows_written,
            "avg_fps": avg_fps,
            "tracking_rate": tracking_rate,
            "triangle_rate": triangle_rate,
        }

    def close(self):
        if not self.file.closed:
            self.file.close()