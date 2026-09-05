"""CPU checks for DeepLabV3-ResNet50 IF conversion and MNE detach."""
import math
import sys
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
EXP = Path(__file__).resolve().parent
for path in (ROOT, EXP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Models.DeepLab import convert_bottleneck_relu_to_if  # noqa: E402
from Models.layer import IF  # noqa: E402
from Models.ResNet import resnet18  # noqa: E402
import run_voc_deeplabv3_resnet50_mne_seed42 as runner  # noqa: E402
from utils import collect_weight_layer_matches, compute_mne_l2_regularization  # noqa: E402


class TinyBottleneck(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 4, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 4, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(4)
        self.conv3 = nn.Conv2d(4, 8, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(nn.Conv2d(4, 8, 1, bias=False), nn.BatchNorm2d(8))


class DeepLabMNEChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_torchvision_bottleneck_gets_three_ifs_and_resnet_matches(self):
        class Stage(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Sequential(TinyBottleneck())

        model = Stage()
        convert_bottleneck_relu_to_if(model.layer1[0])
        block = model.layer1[0]
        self.assertIsInstance(block.if1, IF)
        self.assertIsInstance(block.if3, IF)
        self.assertIsInstance(block.relu, nn.Identity)
        x = torch.zeros(2, 4, 8, 8)
        y = block(x)
        self.assertEqual(tuple(y.shape), (2, 8, 8, 8))
        rows = {row["name"]: row for row in collect_weight_layer_matches(model, "resnet")}
        self.assertTrue(rows["layer1.0.conv1"]["matched"])
        self.assertEqual(rows["layer1.0.conv1"]["role"], "residual_preact")
        self.assertEqual(rows["layer1.0.conv3"]["role"], "residual_terminal")
        self.assertEqual(rows["layer1.0.downsample.0"]["role"], "shortcut")
        self.assertEqual(rows["layer1.0.conv3"]["if_name"], rows["layer1.0.downsample.0"]["if_name"])

    def test_cifar_resnet18_matching_unchanged(self):
        model = resnet18(num_classes=10)
        rows = collect_weight_layer_matches(model, "resnet")
        summary = runner.summarize_weight_layer_matches(rows)
        self.assertGreater(summary["n_matched"], 0)
        self.assertEqual(summary["unmatched_body"], [])

    def test_runner_self_check_and_detach(self):
        card = runner.self_check(torch.device("cpu"))
        self.assertEqual(card["n_if"], 56)
        self.assertEqual(card["n_matched"], 60)
        self.assertEqual(card["n_unmatched"], 1)
        self.assertGreater(card["beta_match"], 0)

    def test_detach_does_not_change_initial_weight_grad(self):
        model = runner.make_model(0, torch.device("cpu"), load_coco=False)
        weights = [row["weight"] for row in collect_weight_layer_matches(model, "resnet") if row["matched"]]
        grads = []
        for detach in (True, False):
            model.zero_grad(set_to_none=True)
            kw = dict(runner.DETACH, detach_lambda=detach)
            compute_mne_l2_regularization(model, quant_level=16, **kw).backward()
            grads.append([weight.grad.detach().clone() for weight in weights])
            thresh = next(module.thresh for module in model.modules() if isinstance(module, IF))
            if detach:
                self.assertIsNone(thresh.grad)
            else:
                self.assertIsNotNone(thresh.grad)
        for left, right in zip(grads[0], grads[1]):
            torch.testing.assert_close(left, right)

    def test_beta_matches_autograd_norm(self):
        model = runner.make_model(0, torch.device("cpu"), load_coco=False)
        report = runner.matched_weight_beta(model)
        self.assertTrue(math.isfinite(report["beta_match"]))
        self.assertAlmostEqual(
            report["beta_match"] * report["unit_mne_grad_norm"],
            report["reference_l2_grad_norm"],
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
