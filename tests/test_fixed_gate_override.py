from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "train_fusion_gating2.py"
)
SPEC = importlib.util.spec_from_file_location("train_fusion_gating2", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec from {MODULE_PATH}")
gating_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gating_module
SPEC.loader.exec_module(gating_module)


class FixedGateOverrideTest(unittest.TestCase):
    def test_resolve_gate_score_returns_constant_budget_when_fixed_value_is_set(self) -> None:
        gate_mlp = nn.Sequential(
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid(),
        )
        v_global = torch.randn(3, 4)
        t_global = torch.randn(3, 4)

        gate_score = gating_module.resolve_gate_score(
            gate_mlp=gate_mlp,
            v_global=v_global,
            t_global=t_global,
            fixed_gate_value=0.1,
        )

        self.assertEqual(tuple(gate_score.shape), (3, 1))
        self.assertTrue(torch.allclose(gate_score, torch.full((3, 1), 0.1)))

    def test_resolve_gate_score_uses_gate_mlp_when_no_fixed_value_is_set(self) -> None:
        gate_mlp = nn.Sequential(
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid(),
        )
        v_global = torch.randn(2, 4)
        t_global = torch.randn(2, 4)

        gate_score = gating_module.resolve_gate_score(
            gate_mlp=gate_mlp,
            v_global=v_global,
            t_global=t_global,
            fixed_gate_value=None,
        )

        self.assertEqual(tuple(gate_score.shape), (2, 1))
        self.assertTrue(torch.all(gate_score >= 0.0).item())
        self.assertTrue(torch.all(gate_score <= 1.0).item())


if __name__ == "__main__":
    unittest.main()
