from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "aggregate_multi_seed_results.py"
SPEC = importlib.util.spec_from_file_location("aggregate_multi_seed_results", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec from {MODULE_PATH}")
aggregate_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = aggregate_module
SPEC.loader.exec_module(aggregate_module)


class AggregateMultiSeedResultsTest(unittest.TestCase):
    def test_extract_metrics_supports_legacy_flat_results(self) -> None:
        payload = {
            "loss": 1.23,
            "mass": 0.8,
            "stiffness": 0.7,
            "material": 0.6,
            "avg_gate_score": 0.2,
        }

        metrics = aggregate_module.extract_metrics(
            result=payload,
            tasks=["mass", "stiffness", "material"],
            extra_metric_keys=["avg_gate_score"],
        )

        self.assertEqual(metrics["loss"], 1.23)
        self.assertAlmostEqual(metrics["avg"], (0.8 + 0.7 + 0.6) / 3.0)
        self.assertEqual(metrics["avg_gate_score"], 0.2)

    def test_extract_metrics_supports_nested_results(self) -> None:
        payload = {
            "loss": 2.34,
            "summary": {
                "average_accuracy": 0.75,
                "average_tactile_weight": 0.61,
            },
            "tasks": {
                "mass": {"accuracy": 0.9},
                "stiffness": {"accuracy": 0.7},
                "material": {"accuracy": 0.65},
            },
        }

        metrics = aggregate_module.extract_metrics(
            result=payload,
            tasks=["mass", "stiffness", "material"],
            extra_metric_keys=["average_tactile_weight"],
        )

        self.assertEqual(metrics["loss"], 2.34)
        self.assertEqual(metrics["avg"], 0.75)
        self.assertEqual(metrics["mass"], 0.9)
        self.assertEqual(metrics["stiffness"], 0.7)
        self.assertEqual(metrics["material"], 0.65)
        self.assertEqual(metrics["average_tactile_weight"], 0.61)


if __name__ == "__main__":
    unittest.main()
