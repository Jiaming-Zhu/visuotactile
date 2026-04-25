from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_fusion_qmf.py"


def load_qmf_module():
    spec = importlib.util.spec_from_file_location("train_fusion_qmf", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class QMFBaselineTest(unittest.TestCase):
    def test_qmf_training_script_exists(self) -> None:
        self.assertTrue(
            MODULE_PATH.exists(),
            f"Expected QMF baseline script at {MODULE_PATH}",
        )

    def test_normalize_quality_weights_prefers_higher_quality_modality(self) -> None:
        if not MODULE_PATH.exists():
            self.skipTest("QMF module does not exist yet")

        qmf_module = load_qmf_module()
        vision_quality = torch.tensor([[2.0], [0.1]], dtype=torch.float32)
        tactile_quality = torch.tensor([[0.1], [2.0]], dtype=torch.float32)

        weights = qmf_module.normalize_quality_weights(
            vision_quality=vision_quality,
            tactile_quality=tactile_quality,
            temperature=1.0,
        )

        self.assertEqual(tuple(weights.shape), (2, 2))
        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-6))
        self.assertGreater(weights[0, 0].item(), weights[0, 1].item())
        self.assertGreater(weights[1, 1].item(), weights[1, 0].item())

    def test_qmf_model_forward_returns_task_logits_and_weights(self) -> None:
        if not MODULE_PATH.exists():
            self.skipTest("QMF module does not exist yet")

        qmf_module = load_qmf_module()
        model = qmf_module.QMFModel(
            fusion_dim=32,
            dropout=0.0,
            freeze_visual=False,
            mass_classes=3,
            stiffness_classes=4,
            material_classes=5,
            use_imagenet_weights=False,
        )
        model.eval()

        images = torch.randn(2, 3, 224, 224)
        tactile = torch.randn(2, 24, 3000)
        padding_mask = torch.zeros(2, 3000, dtype=torch.bool)

        with torch.no_grad():
            outputs = model(images, tactile, padding_mask=padding_mask)

        for task, class_count in (("mass", 3), ("stiffness", 4), ("material", 5)):
            self.assertIn(task, outputs)
            self.assertIn(f"vision_{task}", outputs)
            self.assertIn(f"tactile_{task}", outputs)
            self.assertIn(f"{task}_weights", outputs)
            self.assertEqual(tuple(outputs[task].shape), (2, class_count))
            self.assertEqual(tuple(outputs[f"vision_{task}"].shape), (2, class_count))
            self.assertEqual(tuple(outputs[f"tactile_{task}"].shape), (2, class_count))
            self.assertEqual(tuple(outputs[f"{task}_weights"].shape), (2, 2))
            self.assertTrue(
                torch.allclose(outputs[f"{task}_weights"].sum(dim=1), torch.ones(2), atol=1e-6)
            )


if __name__ == "__main__":
    unittest.main()
