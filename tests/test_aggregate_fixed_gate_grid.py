from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "aggregate_fixed_gate_grid.py"
)


def load_module():
    if not MODULE_PATH.exists():
        raise AssertionError(f"Expected aggregation script at {MODULE_PATH}")
    spec = importlib.util.spec_from_file_location("aggregate_fixed_gate_grid", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_eval_json(path: Path, avg: float, mass: float, stiffness: float, material: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": {"average_accuracy": avg},
        "loss": 1.0 - avg,
        "avg_gate_score": 0.1,
        "tasks": {
            "mass": {"accuracy": mass},
            "stiffness": {"accuracy": stiffness},
            "material": {"accuracy": material},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_object_level_json(path: Path, avg: float, ci_low: float, ci_high: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "object_macro": {
            "mass": avg,
            "stiffness": avg,
            "material": avg,
            "avg": avg,
        },
        "grouped_bootstrap_avg": {
            "mean": avg,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "num_objects": 5,
            "num_resamples": 2000,
            "seed": 7,
        },
        "per_object": {
            "ObjA": {"mass": avg, "stiffness": avg, "material": avg, "avg": avg},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class AggregateFixedGateGridTest(unittest.TestCase):
    def test_aggregate_gate_object_level_reports_macro_and_bootstrap(self) -> None:
        module = load_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dirs = {}
            for seed, avg, ci in [(42, 0.80, (0.70, 0.90)), (123, 0.90, (0.82, 0.96))]:
                run_dir = root / f"fusion_fixed_gate_g002_seed{seed}"
                run_dirs[seed] = run_dir
                write_object_level_json(
                    run_dir / "object_level_ood_test" / "object_level_results.json",
                    avg=avg,
                    ci_low=ci[0],
                    ci_high=ci[1],
                )

            summary = module.aggregate_gate_object_level(run_dirs, "ood_test")

            self.assertEqual(summary["used_seeds"], [42, 123])
            self.assertAlmostEqual(summary["object_macro_avg"]["mean"], 0.85)
            self.assertAlmostEqual(summary["object_macro_avg"]["std"], 0.05)
            self.assertEqual(summary["grouped_bootstrap_avg"]["num_objects"], [5, 5])

    def test_select_best_gate_prefers_highest_ood_mean(self) -> None:
        module = load_module()

        payload = {
            "variants": {
                "0.01": {
                    "eval": {"ood_test": {"average_accuracy": {"mean": 0.91}}},
                },
                "0.02": {
                    "eval": {"ood_test": {"average_accuracy": {"mean": 0.94}}},
                },
                "0.05": {
                    "eval": {"ood_test": {"average_accuracy": {"mean": 0.90}}},
                },
            }
        }

        best = module.select_best_gate(payload, split_name="ood_test", metric="average_accuracy")

        self.assertEqual(best["gate_key"], "0.02")
        self.assertAlmostEqual(best["mean"], 0.94)

    def test_build_paired_delta_summary_aligns_seedwise_values(self) -> None:
        module = load_module()

        reference_summary = {
            "eval": {
                "ood_test": {
                    "used_seeds": [42, 123],
                    "average_accuracy": {"values": [0.95, 0.93]},
                }
            }
        }
        variant_block = {
            "eval": {
                "ood_test": {
                    "used_seeds": [42, 123],
                    "average_accuracy": {"values": [0.90, 0.89]},
                }
            }
        }

        paired = module.build_paired_delta_summary(
            reference_summary=reference_summary,
            variant_block=variant_block,
            split_name="ood_test",
            metric="average_accuracy",
        )

        self.assertEqual(paired["used_seeds"], [42, 123])
        self.assertEqual(len(paired["deltas"]), 2)
        self.assertAlmostEqual(paired["deltas"][0], 0.05)
        self.assertAlmostEqual(paired["deltas"][1], 0.04)
        self.assertAlmostEqual(paired["mean_delta"], 0.045)

    def test_gate_tag_supports_dense_low_budget_values(self) -> None:
        module = load_module()

        self.assertEqual(module.gate_tag(0.01), "001")
        self.assertEqual(module.gate_tag(0.02), "002")
        self.assertEqual(module.gate_tag(0.30), "030")


if __name__ == "__main__":
    unittest.main()
