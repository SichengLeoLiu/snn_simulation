"""CPU checks; synthetic images and tiny models are not research results."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import nn

import run_voc_fcn32s_mne_tuning as tuning


class TinyFCN(tuning.base.FCN32sVGG16):
    def __init__(self):
        nn.Module.__init__(self)
        self.T = 0
        self.merge = nn.Flatten(0, 1)
        self.spike_schedule = "normal"
        self.first_layer_input_noise_sigma = 0.0
        self.first_layer_input_noise_type = "gaussian"
        self.first_layer_input_noise_position = "post_input_if"
        self._mne_layer_map = "legacy"
        self.stage1 = nn.Sequential(nn.Conv2d(3, 4, 1), tuning.base.IF(L=16, thresh=2.0),
                                    nn.Conv2d(4, 4, 1), tuning.base.IF(L=16, thresh=3.0))
        self.stage2 = self.stage3 = self.stage4 = self.stage5 = nn.Identity()
        self.fc6 = self.fc7 = nn.Identity()
        self.classifier = nn.Conv2d(4, 21, 1)


def make_tiny(seed, device, load_imagenet=True):
    tuning.base.seed_all(seed)
    return TinyFCN().to(device), {"synthetic_fixture": True}


def synthetic_voc(root):
    voc = root / "VOC2012"
    sets = voc / "ImageSets/Segmentation"
    sets.mkdir(parents=True)
    (voc / "JPEGImages").mkdir()
    (voc / "SegmentationClass").mkdir()
    for split, ids in (("train", range(6)), ("val", range(6, 8))):
        (sets / f"{split}.txt").write_text("\n".join(map(str, ids)) + "\n")
        for n in ids:
            rng = np.random.default_rng(n)
            Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(voc / f"JPEGImages/{n}.jpg")
            Image.fromarray(rng.integers(0, 3, (32, 32), dtype=np.uint8)).save(voc / f"SegmentationClass/{n}.png")


class TuningChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_strength_matches_actual_mne_weight_gradients(self):
        model = TinyFCN().double()
        report = tuning.strength_report(model)
        penalty = tuning.base.compute_mne_l2_regularization(model, quant_level=16, detach_lambda=True)
        weights = [r["weight"] for r in tuning.base.collect_weight_layer_matches(model) if r["matched"]]
        grads = torch.autograd.grad(penalty, weights)
        actual = torch.sqrt(sum(g.square().sum() for g in grads)).item()
        self.assertAlmostEqual(actual, report["unit_mne_grad_norm"], places=10)
        self.assertAlmostEqual(report["beta_match"] * actual, report["reference_l2_grad_norm"], places=10)
        for g, w, row in zip(grads, weights, report["layers"]):
            torch.testing.assert_close(g, w * row["unit_mne_weight_gradient_coeff"])

    def test_detach_affects_lambda_not_the_initial_weight_gradient(self):
        model = TinyFCN()
        for detach in (True, False):
            model.zero_grad(set_to_none=True)
            loss = tuning.base.compute_mne_l2_regularization(model, quant_level=16, detach_lambda=detach)
            loss.backward()
            if detach:
                reference = model.stage1[0].weight.grad.clone()
                self.assertIsNone(model.stage1[1].thresh.grad)
            else:
                torch.testing.assert_close(reference, model.stage1[0].weight.grad)
                self.assertLess(float(model.stage1[1].thresh.grad), 0)

    def test_equal_head_scope_and_bias_exclusion(self):
        for method in tuning.METHODS:
            model = TinyFCN()
            opt = tuning.optimizer_for(model, method, 10, 1e-3)
            lookup = {id(p): group["weight_decay"] for group in opt.param_groups for p in group["params"]}
            self.assertEqual(lookup[id(model.classifier.weight)], 5e-4)
            self.assertEqual(lookup[id(model.classifier.bias)], 0)
            self.assertEqual(lookup[id(model.stage1[0].weight)], 5e-3 if method == "l2wo" else 0)
            self.assertEqual(lookup[id(model.stage1[1].thresh)], 0)

    def test_actual_architecture_mapping_without_allocating_weights(self):
        with torch.device("meta"):
            model = tuning.base.FCN32sVGG16()
        rows = tuning.base.collect_weight_layer_matches(model)
        self.assertEqual(sum(r["matched"] for r in rows), 15)
        self.assertEqual([r["name"] for r in rows if not r["matched"]], ["classifier"])

    def test_split_is_fixed_disjoint_and_order_independent(self):
        ids = list(map(str, range(20)))
        first = tuning.split_ids(ids, ["val"], 5, 2026)
        self.assertEqual(first, tuning.split_ids(ids[::-1], ["val"], 5, 2026))
        self.assertFalse(set(first["fit"]) & set(first["tune"]))
        self.assertEqual(set(first["fit"] + first["tune"]), set(ids))
        with self.assertRaises(ValueError):
            tuning.split_ids(ids, ["1"], 5, 2026)

    def test_accumulation_tail_has_full_strength(self):
        self.assertEqual([tuning.accumulation_divisor(i, 10, 8) for i in range(10)], [8] * 8 + [2] * 2)

    def test_skip_all_ignore_or_nonfinite_batch(self):
        ignore = torch.full((1, 4, 4), tuning.base.IGNORE_INDEX)
        valid = torch.zeros((1, 4, 4), dtype=torch.long)
        self.assertEqual(tuning.skip_train_batch(ignore, torch.tensor(1.0)), "all_ignore")
        self.assertEqual(tuning.skip_train_batch(valid, torch.tensor(float("nan"))), "nonfinite")
        self.assertIsNone(tuning.skip_train_batch(valid, torch.tensor(0.2)))

    def test_selection_uses_auc_subject_to_clean_floor(self):
        rows = [{"method": "l2wo", "strength": 1, "clean_miou": 50, "auc_mean_miou": 40},
                {"method": "mne", "strength": 1, "clean_miou": 49.5, "auc_mean_miou": 42},
                {"method": "mne", "strength": 10, "clean_miou": 47, "auc_mean_miou": 44},
                {"method": "nodetach", "strength": 1, "clean_miou": 48, "auc_mean_miou": 45}]
        floor, selected = tuning.rank_candidates(rows, 1)
        self.assertEqual(floor, 49)
        self.assertEqual(selected["mne"]["strength"], 1)
        self.assertIsNone(selected["nodetach"])

    def test_config_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            tuning.immutable_json(path, {"seed": 42})
            tuning.immutable_json(path, {"seed": 42})
            with self.assertRaises(ValueError):
                tuning.immutable_json(path, {"seed": 43})

    def test_resume_keeps_locked_config_when_source_hash_changes(self):
        locked = {"protocol": {"source_hashes": {"run.py": "old"}}, "method": "l2wo",
                  "seed": 42, "strength": 0.1, "stage": "tune", "smoke": False,
                  "selection_hash": None}
        fresh = dict(locked)
        fresh["protocol"] = {"source_hashes": {"run.py": "new"}}
        for key in ("method", "seed", "strength", "stage", "smoke"):
            self.assertEqual(locked[key], fresh[key])
        self.assertNotEqual(locked["protocol"], fresh["protocol"])

    def test_selection_requires_full_grid_then_confirms_on_full_training_split(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            synthetic_voc(root)
            common = ["--voc-root", str(root), "--out-root", str(root / "out"), "--epochs", "2",
                      "--workers", "0", "--crop", "32", "--accum", "3", "--device", "cpu", "--tune-size", "2"]
            args = tuning.parse_args(["prepare"] + common)
            tuning.prepare(args)
            manifest = tuning.load_manifest(args.split_file)
            with self.assertRaises(FileNotFoundError):
                tuning.select(args)
            for method in tuning.METHODS:
                for strength in tuning.STRENGTHS:
                    out = tuning.run_dir(args.out_root, "tune", method, strength, 42)
                    # Deliberately synthetic ranking fixtures, confined to the temporary directory.
                    tuning.save_json(out / "result.json", {
                        "config": {"protocol": tuning.protocol(args, manifest), "method": method,
                                   "strength": strength, "seed": 42, "stage": "tune", "smoke": False},
                        "beta": None if method == "l2wo" else 0.1,
                        "clean_miou": 50, "auc_mean_miou": 40 + strength / 10, "sigma5_miou": 30,
                        "n_eval": 2, "snn": [{"n_images": 2, "sigma": s} for s in tuning.base.SIGMAS]})
            tuning.select(args)
            selected = json.loads(args.selection.read_text())
            self.assertEqual(len(selected["all_candidates"]), 9)
            self.assertEqual(selected["selected"]["l2wo"]["strength"], 10)
            confirm = tuning.parse_args(["run", "--stage", "confirm", "--method", "l2wo"] + common)
            with patch.object(tuning.base, "make_model", make_tiny):
                tuning.run(confirm)
            result = json.loads((tuning.run_dir(args.out_root, "confirm", "l2wo", 10, 42) / "result.json").read_text())
            self.assertEqual(result["n_train"], 6)
            self.assertEqual(result["n_eval"], 2)
            self.assertIsNotNone(result["config"]["selection_hash"])
            confirm.lr = 0.01
            with self.assertRaises(ValueError):
                tuning.run(confirm)

    def test_small_end_to_end_run_and_epoch_resume(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            synthetic_voc(root)
            common = ["--voc-root", str(root), "--epochs", "2", "--workers", "0", "--crop", "32",
                      "--accum", "3", "--device", "cpu", "--tune-size", "2"]
            original_seed = tuning.base.seed_all
            def interrupt_second_epoch(seed):
                if seed == 44:
                    raise RuntimeError("simulated interruption")
                original_seed(seed)
            states = []
            for folder in ("continuous", "resumed"):
                opts = common + ["--out-root", str(root / folder)]
                tuning.prepare(tuning.parse_args(["prepare"] + opts))
                args = tuning.parse_args(["run", "--method", "nodetach", "--strength", "1"] + opts)
                with patch.object(tuning.base, "make_model", make_tiny):
                    if folder == "resumed":
                        with patch.object(tuning.base, "seed_all", interrupt_second_epoch), self.assertRaises(RuntimeError):
                            tuning.run(args)
                    tuning.run(args)
                out = tuning.run_dir(root / folder, "tune", "nodetach", 1, 42)
                result = json.loads((out / "result.json").read_text())
                self.assertEqual(result["n_train"], 4)
                self.assertEqual(result["n_eval"], 2)
                self.assertEqual(len(result["snn"]), 6)
                self.assertEqual(result["ann"]["n_images"], 2)
                states.append(torch.load(out / "last.pth", weights_only=True)["state_dict"])
            for name in states[0]:
                torch.testing.assert_close(states[0][name], states[1][name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
