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
    / "diagnose_gating_hollow_vs_yoga.py"
)
SPEC = importlib.util.spec_from_file_location("diagnose_gating_hollow_vs_yoga", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec from {MODULE_PATH}")
diagnosis_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = diagnosis_module
SPEC.loader.exec_module(diagnosis_module)


class DiagnoseGatingHelpersTest(unittest.TestCase):
    def test_compute_true_class_margin_prefers_true_logit_gap(self) -> None:
        logits = np.array([0.2, 1.1, -0.3], dtype=np.float32)

        margin = diagnosis_module.compute_true_class_margin(logits, true_idx=1)

        self.assertAlmostEqual(margin, 0.9, places=6)

    def test_leave_one_out_probe_predictions_handles_easy_problem(self) -> None:
        features = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.0, 0.1],
                [2.0, 2.0],
                [2.1, 2.0],
                [2.0, 2.1],
            ],
            dtype=np.float32,
        )
        labels = np.array(["a", "a", "a", "b", "b", "b"])
        sample_rows = [{"episode_name": f"ep_{idx}"} for idx in range(len(labels))]

        rows, accuracy = diagnosis_module.leave_one_out_probe_predictions(
            features,
            labels,
            sample_rows=sample_rows,
        )

        self.assertEqual(len(rows), 6)
        self.assertGreaterEqual(accuracy, 0.99)
        self.assertTrue(all("pred_label" in row for row in rows))
        self.assertTrue(all("correct" in row for row in rows))

    def test_stage_summary_row_captures_probe_and_pair_metrics(self) -> None:
        row = diagnosis_module.build_stage_summary_row(
            seed=42,
            stage_name="cls_out",
            probe_result={
                "mean_accuracy": 0.9,
                "std_accuracy": 0.1,
                "n_splits": 5,
            },
            pair_accuracy=0.6,
            stiffness_accuracy=0.7,
            material_accuracy=0.8,
        )

        self.assertEqual(row["seed"], 42)
        self.assertEqual(row["stage"], "cls_out")
        self.assertAlmostEqual(row["probe_mean_accuracy"], 0.9)
        self.assertAlmostEqual(row["pair_accuracy"], 0.6)
        self.assertAlmostEqual(row["stiffness_accuracy"], 0.7)
        self.assertAlmostEqual(row["material_accuracy"], 0.8)

    def test_train_small_mlp_head_predictions_handles_easy_problem(self) -> None:
        features = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.0, 0.1],
                [2.0, 2.0],
                [2.1, 2.0],
                [2.0, 2.1],
            ],
            dtype=np.float32,
        )
        labels = np.array(["a", "a", "a", "b", "b", "b"])
        sample_rows = [{"episode_name": f"ep_{idx}"} for idx in range(len(labels))]

        rows, metrics = diagnosis_module.train_small_mlp_head_predictions(
            features,
            labels,
            sample_rows=sample_rows,
            random_state=7,
            num_epochs=120,
            hidden_dim=8,
        )

        self.assertEqual(len(rows), 6)
        self.assertGreaterEqual(metrics["mean_accuracy"], 0.99)
        self.assertEqual(metrics["n_splits"], 3)
        self.assertTrue(all("pred_label" in row for row in rows))

    def test_decompose_final_linear_logits_splits_projection_and_bias(self) -> None:
        hidden = np.array([1.0, 2.0], dtype=np.float32)
        weight = np.array([[2.0, -1.0], [0.5, 1.5]], dtype=np.float32)
        bias = np.array([0.3, -0.2], dtype=np.float32)

        projection, logits = diagnosis_module.decompose_final_linear_logits(hidden, weight, bias)

        np.testing.assert_allclose(projection, np.array([0.0, 3.5], dtype=np.float32))
        np.testing.assert_allclose(logits, np.array([0.3, 3.3], dtype=np.float32))

    def test_train_frozen_task_head_predictions_handles_easy_labels(self) -> None:
        features = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.0, 0.1],
                [0.1, 0.1],
                [0.2, 0.0],
                [2.0, 2.0],
                [2.1, 2.0],
                [2.0, 2.1],
                [2.1, 2.1],
                [2.2, 2.0],
            ],
            dtype=np.float32,
        )
        labels = np.array(
            [
                "soft",
                "soft",
                "soft",
                "soft",
                "soft",
                "medium",
                "medium",
                "medium",
                "medium",
                "medium",
            ]
        )
        sample_rows = [{"episode_name": f"ep_{idx}"} for idx in range(len(labels))]

        rows, metrics = diagnosis_module.train_frozen_task_head_predictions(
            features,
            labels,
            sample_rows=sample_rows,
            task_name="stiffness",
            random_state=11,
            num_epochs=160,
            hidden_dim=8,
        )

        self.assertEqual(len(rows), 10)
        self.assertGreaterEqual(metrics["mean_accuracy"], 0.99)
        self.assertEqual(metrics["task"], "stiffness")
        self.assertTrue(all("pred_label" in row for row in rows))

    def test_train_frozen_task_head_train_eval_split_handles_easy_labels(self) -> None:
        train_features = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.0, 0.1],
                [2.0, 2.0],
                [2.1, 2.0],
                [2.0, 2.1],
            ],
            dtype=np.float32,
        )
        train_labels = np.array(["soft", "soft", "soft", "medium", "medium", "medium"])
        eval_features = np.array(
            [
                [0.05, 0.05],
                [2.05, 2.05],
            ],
            dtype=np.float32,
        )
        eval_labels = np.array(["soft", "medium"])
        eval_rows = [{"episode_name": "eval_a"}, {"episode_name": "eval_b"}]

        rows, metrics = diagnosis_module.train_frozen_task_head_train_eval_split(
            train_features,
            train_labels,
            eval_features,
            eval_labels,
            eval_rows,
            task_name="stiffness",
            random_state=13,
            num_epochs=120,
            hidden_dim=8,
        )

        self.assertEqual(len(rows), 2)
        self.assertGreaterEqual(metrics["eval_accuracy"], 0.99)
        self.assertEqual(metrics["task"], "stiffness")
        self.assertEqual(metrics["num_eval_samples"], 2)


if __name__ == "__main__":
    unittest.main()
