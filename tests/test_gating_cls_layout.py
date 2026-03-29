from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


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


class GatingClsLayoutTest(unittest.TestCase):
    def test_num_cls_tokens_uses_single_token_by_default(self) -> None:
        self.assertEqual(gating_module.num_cls_tokens(False), 1)

    def test_num_cls_tokens_uses_three_tokens_when_enabled(self) -> None:
        self.assertEqual(gating_module.num_cls_tokens(True), 3)

    def test_task_cls_indices_share_single_token_when_disabled(self) -> None:
        indices = gating_module.task_cls_indices(False)

        self.assertEqual(indices, {"mass": 0, "stiffness": 0, "material": 0})

    def test_task_cls_indices_assign_distinct_tokens_when_enabled(self) -> None:
        indices = gating_module.task_cls_indices(True)

        self.assertEqual(indices, {"mass": 0, "stiffness": 1, "material": 2})

    def test_max_sequence_length_accounts_for_three_cls_tokens(self) -> None:
        self.assertEqual(gating_module.max_sequence_length(False), 425)
        self.assertEqual(gating_module.max_sequence_length(True), 427)


if __name__ == "__main__":
    unittest.main()
