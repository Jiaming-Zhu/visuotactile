from __future__ import annotations

import importlib.util
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "preview_camera.py"
SPEC = importlib.util.spec_from_file_location("preview_camera", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec from {MODULE_PATH}")
preview_camera = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preview_camera)


class PreviewCameraHelpersTest(unittest.TestCase):
    def test_opencv_gui_available_returns_false_for_headless_build(self) -> None:
        build_info = """
General configuration for OpenCV 4.12.0 =====================================
  GUI:                           NONE
"""

        available = preview_camera._opencv_gui_available(build_info=build_info)

        self.assertFalse(available)

    def test_select_preview_backend_prefers_tk_when_opencv_gui_missing(self) -> None:
        backend = preview_camera._select_preview_backend(
            opencv_gui_available=False,
            tkinter_available=True,
        )

        self.assertEqual(backend, "tk")

    def test_select_preview_backend_falls_back_to_terminal_without_gui(self) -> None:
        backend = preview_camera._select_preview_backend(
            opencv_gui_available=False,
            tkinter_available=False,
        )

        self.assertEqual(backend, "terminal")

    def test_ensure_frame_size_keeps_matching_shape(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        resized = preview_camera._ensure_frame_size(frame, width=640, height=480)

        self.assertEqual(resized.shape, (480, 640, 3))
        self.assertTrue(np.array_equal(resized, frame))

    def test_ensure_frame_size_resizes_mismatched_shape(self) -> None:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        resized = preview_camera._ensure_frame_size(frame, width=640, height=480)

        self.assertEqual(resized.shape, (480, 640, 3))

    def test_make_snapshot_path_uses_prefix_and_timestamp(self) -> None:
        output_dir = Path("/tmp/preview-test")
        timestamp = datetime(2026, 3, 28, 14, 15, 16, 123000)

        save_path = preview_camera._make_snapshot_path(
            output_dir=output_dir,
            prefix="webcam_test",
            now=timestamp,
        )

        self.assertEqual(
            save_path,
            output_dir / "webcam_test_20260328_141516_123.jpg",
        )


if __name__ == "__main__":
    unittest.main()
