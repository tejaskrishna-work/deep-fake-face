import os
from warp import HybridOverlayWarper


class OverlayManager:
    def __init__(self, overlays_dir, tracker, default_filename=None):
        self.overlays_dir = overlays_dir
        self.tracker = tracker

        self.overlay_files = self._scan_overlay_files()
        if not self.overlay_files:
            raise RuntimeError(f"No overlay images found in: {overlays_dir}")

        self.index = 0
        if default_filename:
            for i, name in enumerate(self.overlay_files):
                if name == default_filename:
                    self.index = i
                    break

        self.cache = {}
        self.current_warper = self._get_or_create_warper(self.current_path)

    def _scan_overlay_files(self):
        valid_ext = {".png", ".jpg", ".jpeg", ".webp"}
        files = []
        for name in sorted(os.listdir(self.overlays_dir)):
            path = os.path.join(self.overlays_dir, name)
            if os.path.isfile(path):
                ext = os.path.splitext(name)[1].lower()
                if ext in valid_ext:
                    files.append(name)
        return files

    @property
    def current_name(self):
        return self.overlay_files[self.index]

    @property
    def current_path(self):
        return os.path.join(self.overlays_dir, self.current_name)

    def _get_or_create_warper(self, path):
        if path not in self.cache:
            self.cache[path] = HybridOverlayWarper(path, self.tracker)
        return self.cache[path]

    def get_warper(self):
        return self.current_warper

    def next_overlay(self):
        self.index = (self.index + 1) % len(self.overlay_files)
        self.current_warper = self._get_or_create_warper(self.current_path)
        return self.current_name

    def prev_overlay(self):
        self.index = (self.index - 1) % len(self.overlay_files)
        self.current_warper = self._get_or_create_warper(self.current_path)
        return self.current_name

    def describe_current(self):
        warper = self.current_warper
        return {
            "name": self.current_name,
            "path": self.current_path,
            "overlay_face_ok": warper.overlay_landmarks_available(),
            "triangle_ok": warper.triangle_available(),
        }

    def scan_all(self):
        results = []
        for name in self.overlay_files:
            path = os.path.join(self.overlays_dir, name)
            warper = self._get_or_create_warper(path)
            results.append(
                {
                    "name": name,
                    "path": path,
                    "overlay_face_ok": warper.overlay_landmarks_available(),
                    "triangle_ok": warper.triangle_available(),
                }
            )
        return results