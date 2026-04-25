from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluate_object_level_ood.py"
)


def load_module():
    if not MODULE_PATH.exists():
        raise AssertionError(f"Expected analysis script at {MODULE_PATH}")
    spec = importlib.util.spec_from_file_location("evaluate_object_level_ood", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ObjectLevelEvidenceTest(unittest.TestCase):
    def test_extract_object_id_from_episode_path(self) -> None:
        module = load_module()

        episode_dir = Path(
            "/tmp/Plaintextdataset/ood_test/Sponge_Red/episode_20260201_225717_453905"
        )

        object_id = module.object_id_from_episode_dir(episode_dir)

        self.assertEqual(object_id, "Sponge_Red")

    def test_episode_dir_from_sample_supports_image_and_tactile_samples(self) -> None:
        module = load_module()

        image_sample = SimpleNamespace(
            img_path=Path("/tmp/Plaintextdataset/ood_test/Sponge_Red/episode_a/visual_anchor.jpg")
        )
        tactile_sample = SimpleNamespace(
            tactile_path=Path("/tmp/Plaintextdataset/ood_test/Box_Rock_Blue/episode_b/tactile_data.pkl")
        )

        self.assertEqual(
            module.episode_dir_from_sample(image_sample),
            Path("/tmp/Plaintextdataset/ood_test/Sponge_Red/episode_a"),
        )
        self.assertEqual(
            module.episode_dir_from_sample(tactile_sample),
            Path("/tmp/Plaintextdataset/ood_test/Box_Rock_Blue/episode_b"),
        )

    def test_summarize_object_metrics_reports_per_object_and_macro_average(self) -> None:
        module = load_module()

        rows = [
            {
                "object_id": "Sponge_Red",
                "mass_label": 0,
                "mass_pred": 0,
                "stiffness_label": 1,
                "stiffness_pred": 1,
                "material_label": 2,
                "material_pred": 0,
            },
            {
                "object_id": "Sponge_Red",
                "mass_label": 0,
                "mass_pred": 0,
                "stiffness_label": 1,
                "stiffness_pred": 0,
                "material_label": 2,
                "material_pred": 2,
            },
            {
                "object_id": "Box_Rock_Blue",
                "mass_label": 2,
                "mass_pred": 2,
                "stiffness_label": 2,
                "stiffness_pred": 2,
                "material_label": 3,
                "material_pred": 3,
            },
            {
                "object_id": "Box_Rock_Blue",
                "mass_label": 2,
                "mass_pred": 1,
                "stiffness_label": 2,
                "stiffness_pred": 2,
                "material_label": 3,
                "material_pred": 3,
            },
        ]

        summary = module.summarize_object_metrics(rows)

        self.assertEqual(summary["num_objects"], 2)
        self.assertAlmostEqual(summary["per_object"]["Sponge_Red"]["mass"], 1.0)
        self.assertAlmostEqual(summary["per_object"]["Sponge_Red"]["stiffness"], 0.5)
        self.assertAlmostEqual(summary["per_object"]["Sponge_Red"]["material"], 0.5)
        self.assertAlmostEqual(summary["per_object"]["Box_Rock_Blue"]["mass"], 0.5)
        self.assertAlmostEqual(summary["per_object"]["Box_Rock_Blue"]["stiffness"], 1.0)
        self.assertAlmostEqual(summary["per_object"]["Box_Rock_Blue"]["material"], 1.0)
        self.assertAlmostEqual(summary["object_macro"]["mass"], 0.75)
        self.assertAlmostEqual(summary["object_macro"]["stiffness"], 0.75)
        self.assertAlmostEqual(summary["object_macro"]["material"], 0.75)
        self.assertAlmostEqual(summary["object_macro"]["avg"], 0.75)

    def test_grouped_bootstrap_uses_object_as_resampling_unit(self) -> None:
        module = load_module()

        per_object_avg = {
            "Sponge_Red": 0.25,
            "Box_Rock_Blue": 0.75,
            "YogaBrick_Foil_ANCHOR": 1.0,
        }

        stats = module.grouped_bootstrap_mean(
            per_object_avg,
            num_resamples=200,
            seed=7,
        )

        self.assertAlmostEqual(stats["mean"], (0.25 + 0.75 + 1.0) / 3.0)
        self.assertLessEqual(stats["ci_low"], stats["mean"])
        self.assertGreaterEqual(stats["ci_high"], stats["mean"])
        self.assertEqual(stats["num_objects"], 3)

    def test_optional_float_returns_none_for_missing_gate_score(self) -> None:
        module = load_module()

        self.assertIsNone(module.optional_float(None))
        self.assertEqual(module.optional_float(0.125), 0.125)


if __name__ == "__main__":
    unittest.main()
