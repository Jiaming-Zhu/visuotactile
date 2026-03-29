from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "train_fusion_gating_online.py"
)
SPEC = importlib.util.spec_from_file_location("train_fusion_gating_online", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec from {MODULE_PATH}")
online_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = online_module
SPEC.loader.exec_module(online_module)


class SupervisedContrastiveLossTest(unittest.TestCase):
    def test_returns_zero_when_batch_has_no_positive_pairs(self) -> None:
        criterion = online_module.SupervisedContrastiveLoss(temperature=0.07)
        features = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 1, 2], dtype=torch.long)

        loss = criterion(features, labels)

        self.assertTrue(torch.isclose(loss, torch.tensor(0.0)))

    def test_returns_finite_scalar_for_supervised_pairs(self) -> None:
        criterion = online_module.SupervisedContrastiveLoss(temperature=0.1)
        features = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        loss = criterion(features, labels)

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
