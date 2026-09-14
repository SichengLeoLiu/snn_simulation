#!/usr/bin/env python3
"""Unit tests for the threshold-scaling baseline (no CIFAR, no GPU)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = Path(__file__).resolve().parent
for path in (ROOT, EXP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_cifar_lambda_scale_baseline_seed42 import self_check  # noqa: E402


class LambdaScaleBaselineTest(unittest.TestCase):
    def test_self_check(self) -> None:
        self_check()


if __name__ == "__main__":
    unittest.main()
