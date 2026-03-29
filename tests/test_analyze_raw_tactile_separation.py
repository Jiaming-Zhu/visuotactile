from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "visualization"
    / "analyze_raw_tactile_separation.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_raw_tactile_separation", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec from {MODULE_PATH}")
analysis_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis_module
SPEC.loader.exec_module(analysis_module)


class RawTactileSeparationHelpersTest(unittest.TestCase):
    def test_resample_tactile_keeps_expected_shape(self) -> None:
        tactile = np.arange(24 * 10, dtype=np.float32).reshape(24, 10)

        resampled = analysis_module.resample_tactile(tactile, target_len=16)

        self.assertEqual(resampled.shape, (24, 16))

    def test_summary_features_returns_six_stats_per_channel(self) -> None:
        tactile = np.arange(24 * 12, dtype=np.float32).reshape(24, 12)

        features = analysis_module.build_summary_features(tactile)

        self.assertEqual(features.shape, (24 * 6,))

    def test_centroid_distance_matrix_is_symmetric(self) -> None:
        features = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [1.0, 1.0],
                [1.1, 1.0],
            ],
            dtype=np.float32,
        )
        labels = ["a", "a", "b", "b"]

        class_names, matrix = analysis_module.compute_centroid_distance_matrix(features, labels)

        self.assertEqual(class_names, ["a", "b"])
        self.assertEqual(matrix.shape, (2, 2))
        self.assertTrue(np.allclose(matrix, matrix.T))
        self.assertTrue(np.allclose(np.diag(matrix), 0.0))

    def test_linear_probe_handles_easy_binary_problem(self) -> None:
        features = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.1],
                [0.2, 0.0],
                [2.0, 2.0],
                [2.1, 2.2],
                [2.2, 2.1],
            ],
            dtype=np.float32,
        )
        labels = np.array(["left", "left", "left", "right", "right", "right"])

        result = analysis_module.evaluate_linear_probe(
            features,
            labels,
            random_state=7,
        )

        self.assertGreaterEqual(result["mean_accuracy"], 0.95)
        self.assertEqual(result["num_samples"], 6)
        self.assertEqual(result["num_classes"], 2)


if __name__ == "__main__":
    unittest.main()
