"""CPU checks for Pet VGG16-BN classification / segmentation pairing."""
import math
import sys
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
EXP = Path(__file__).resolve().parent
for path in (ROOT, EXP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Models.PetVGG import (  # noqa: E402
    DECODER_CHANNELS,
    IGNORE_INDEX,
    PetVGGClassifier,
    PetVGGSegmentor,
    count_if,
    count_maxpool2d,
    decoder_if_modules,
    encoder_if_modules,
)
from Models.layer import IF  # noqa: E402
import run_pet_vgg16bn_cls_seg_seed42 as runner  # noqa: E402
from pet import ARCHIVES, scores_from_binary_confusion, trimap_to_mask  # noqa: E402
from utils import collect_weight_layer_matches, summarize_weight_layer_matches  # noqa: E402


class PetVGGChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_encoder_has_avgpool_no_maxpool_and_thirteen_ifs(self):
        model = PetVGGClassifier(head_if=False)
        self.assertEqual(count_maxpool2d(model), 0)
        self.assertEqual(count_if(model), 13)
        self.assertEqual(sum(isinstance(m, nn.AvgPool2d) for m in model.modules()), 5)
        self.assertIsInstance(model.classifier, nn.Linear)

    def test_segmentor_stepwise_decoder_linear_readout(self):
        model = PetVGGSegmentor(head_if=False)
        self.assertEqual(count_if(model), 13 + len(DECODER_CHANNELS))
        self.assertIsInstance(model.classifier, nn.Conv2d)
        self.assertEqual(len(model.decoder), 5)
        self.assertEqual(len(encoder_if_modules(model)), 13)
        self.assertEqual(len(decoder_if_modules(model)), 5)
        self.assertEqual([index for index, _n, _m in decoder_if_modules(model)], [0, 1, 2, 3, 4])
        dummy = torch.zeros(1, 3, 224, 224)
        logits = model(dummy)
        self.assertEqual(tuple(logits.shape), (1, 2, 224, 224))

    def test_linear_readout_is_unmatched_hidden_convs_matched(self):
        cls = PetVGGClassifier(head_if=False)
        seg = PetVGGSegmentor(head_if=False)
        cls_sum = summarize_weight_layer_matches(collect_weight_layer_matches(cls, "legacy"))
        seg_sum = summarize_weight_layer_matches(collect_weight_layer_matches(seg, "legacy"))
        self.assertEqual(cls_sum["n_matched"], 13)
        self.assertEqual(cls_sum["n_unmatched"], 1)
        self.assertEqual(cls_sum["unmatched_body"], [])
        self.assertEqual(seg_sum["n_matched"], 18)
        self.assertEqual(seg_sum["n_unmatched"], 1)
        self.assertEqual(seg_sum["unmatched_body"], [])

    def test_download_keeps_complete_part_after_wget_exit_3(self):
        import tempfile
        from pet import _download_one

        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "images.tar.gz"
            part = dest.with_suffix(dest.suffix + ".part")
            part.write_bytes(b"x" * 1_500_000)
            _download_one(("https://example.invalid/skip",), dest, 1_000_000)
            self.assertTrue(dest.is_file())
            self.assertGreaterEqual(dest.stat().st_size, 1_000_000)
            self.assertFalse(part.exists())

    def test_download_urls_prefer_thor(self):
        for name, urls, min_bytes in ARCHIVES:
            self.assertTrue(urls[0].startswith("https://thor.robots.ox.ac.uk/pets/"))
            self.assertGreater(min_bytes, 1_000_000)

    def test_trimap_uncertain_is_ignore(self):
        trimap = np.array([[1, 2, 3], [1, 3, 2]], dtype=np.int16)
        mask = trimap_to_mask(trimap)
        self.assertEqual(int(mask[0, 0]), 1)
        self.assertEqual(int(mask[0, 1]), 0)
        self.assertEqual(int(mask[0, 2]), IGNORE_INDEX)
        conf = torch.tensor([[10.0, 2.0], [1.0, 7.0]])
        scores = scores_from_binary_confusion(conf)
        self.assertAlmostEqual(scores["fg_iou"], 100.0 * 7.0 / 10.0, places=4)
        self.assertAlmostEqual(scores["dice"], 100.0 * 14.0 / 17.0, places=4)

    def test_runner_self_check_and_beta(self):
        cards = runner.self_check(torch.device("cpu"))
        self.assertEqual(cards["cls"]["n_if"], 13)
        self.assertEqual(cards["seg"]["n_if"], 18)
        self.assertTrue(math.isfinite(cards["cls"]["beta_match"]))
        self.assertTrue(math.isfinite(cards["seg"]["beta_match"]))
        self.assertEqual(cards["seg_head_if_n_if"], 19)
        self.assertGreater(count_if(PetVGGClassifier(head_if=True)), 13)


class PetSeg5SeedChecks(unittest.TestCase):
    def test_import_does_not_patch_pairing_layout(self):
        import run_pet_vgg16bn_seg_5seed as seg5

        self.assertIs(runner.cfg_dir, seg5._ORIG_CFG_DIR)
        ns = Namespace(out_root=Path("/tmp/pair"), task="seg", method="mne")
        self.assertEqual(runner.cfg_dir(ns), Path("/tmp/pair/seg/mne"))

    def test_five_seed_layout_and_shared_eval_seed(self):
        import run_pet_vgg16bn_seg_5seed as seg5

        ns = Namespace(out_root=Path("/tmp/five"), method="l2wo", seed=41, head_if=False)
        self.assertEqual(seg5.cfg_dir(ns), Path("/tmp/five/l2wo/seed41"))
        self.assertEqual(seg5.EVAL_NOISE_SEED, 0)
        self.assertEqual(seg5.SEEDS, (40, 41, 42, 43, 44))
        mean, std = seg5._mean_std([1.0, 3.0])
        self.assertAlmostEqual(mean, 2.0)
        self.assertAlmostEqual(std, math.sqrt(2.0))

    def test_five_seed_self_check_and_layout_restore(self):
        import run_pet_vgg16bn_seg_5seed as seg5

        card = seg5.self_check(torch.device("cpu"))
        self.assertEqual(card["n_encoder_if"], 13)
        self.assertEqual(card["n_decoder_if"], 5)
        self.assertEqual(card["identical_crossing"], 0.0)
        self.assertGreater(card["shifted_crossing"], 0.0)
        try:
            seg5.install_output_layout()
            self.assertIs(runner.cfg_dir, seg5.cfg_dir)
        finally:
            seg5.restore_output_layout()
        self.assertIs(runner.cfg_dir, seg5._ORIG_CFG_DIR)


if __name__ == "__main__":
    unittest.main()
