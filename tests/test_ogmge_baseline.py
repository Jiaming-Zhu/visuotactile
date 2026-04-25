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
    / "train_fusion_standard_ogmge.py"
)


def load_module():
    if not MODULE_PATH.exists():
        raise AssertionError(f"Expected OGM-GE script at {MODULE_PATH}")
    spec = importlib.util.spec_from_file_location("train_fusion_standard_ogmge", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ToyBranchModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vis_backbone = nn.Linear(4, 4)
        self.vis_proj = nn.Linear(4, 4)
        self.tac_encoder = nn.Linear(4, 4)
        self.transformer_encoder = nn.Linear(4, 4)
        self.head_mass = nn.Linear(4, 3)
        self.head_stiffness = nn.Linear(4, 3)
        self.head_material = nn.Linear(4, 3)
        self.visual_proxy_head_mass = nn.Linear(4, 3)
        self.tactile_proxy_head_mass = nn.Linear(4, 3)


class OGMGEBaselineTest(unittest.TestCase):
    def test_balanced_scores_keep_identity_scales(self) -> None:
        module = load_module()

        visual_scale, tactile_scale = module.compute_ogmge_scales(
            visual_strength=0.7,
            tactile_strength=0.7,
            alpha=1.0,
            min_scale=0.05,
        )

        self.assertAlmostEqual(visual_scale, 1.0)
        self.assertAlmostEqual(tactile_scale, 1.0)

    def test_visual_dominance_downweights_only_visual_branch(self) -> None:
        module = load_module()

        visual_scale, tactile_scale = module.compute_ogmge_scales(
            visual_strength=0.8,
            tactile_strength=0.2,
            alpha=1.0,
            min_scale=0.05,
        )

        self.assertLess(visual_scale, 1.0)
        self.assertGreaterEqual(visual_scale, 0.05)
        self.assertAlmostEqual(tactile_scale, 1.0)

    def test_collect_branch_parameter_names_separates_visual_tactile_and_shared(self) -> None:
        module = load_module()
        model = ToyBranchModel()

        groups = module.collect_branch_parameter_names(model)

        self.assertIn("vis_backbone.weight", groups["visual"])
        self.assertIn("vis_proj.weight", groups["visual"])
        self.assertIn("visual_proxy_head_mass.weight", groups["visual"])
        self.assertIn("tac_encoder.weight", groups["tactile"])
        self.assertIn("tactile_proxy_head_mass.weight", groups["tactile"])
        self.assertIn("transformer_encoder.weight", groups["shared"])
        self.assertIn("head_mass.weight", groups["shared"])

    def test_apply_branch_gradient_modulation_scales_only_targeted_groups(self) -> None:
        module = load_module()
        model = ToyBranchModel()

        name_groups = module.collect_branch_parameter_names(model)
        named_params = dict(model.named_parameters())

        for param in named_params.values():
            param.grad = torch.ones_like(param)

        module.apply_branch_gradient_modulation(
            named_params=named_params,
            name_groups=name_groups,
            visual_scale=0.5,
            tactile_scale=0.25,
            noise_std=0.0,
        )

        self.assertTrue(torch.allclose(named_params["vis_backbone.weight"].grad, 0.5 * torch.ones_like(named_params["vis_backbone.weight"])))
        self.assertTrue(torch.allclose(named_params["tac_encoder.weight"].grad, 0.25 * torch.ones_like(named_params["tac_encoder.weight"])))
        self.assertTrue(torch.allclose(named_params["head_mass.weight"].grad, torch.ones_like(named_params["head_mass.weight"])))


if __name__ == "__main__":
    unittest.main()
