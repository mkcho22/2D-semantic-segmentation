from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np
from PIL import Image
import torch

from carla_seg.data import CarlaDataset, load_label, pair_files


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for folder in ("RGB", "Label"):
            (self.root / "Training" / folder).mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def put(self, folder, name, pixels):
        Image.fromarray(np.asarray(pixels, dtype=np.uint8)).save(self.root / "Training" / folder / name)

    def test_missing_middle_label_never_shifts_pairing(self):
        for name in ("a.png", "b.png", "c.png"):
            self.put("RGB", name, np.zeros((16, 16, 3)))
        for name in ("a.png", "c.png"):
            self.put("Label", name, np.zeros((16, 16)))
        (self.root / "Training" / "RGB" / ".DS_Store").write_bytes(b"ignored")
        with self.assertRaisesRegex(ValueError, "b.png"):
            pair_files(self.root, "Training")
        with warnings.catch_warnings(record=True) as caught:
            pairs, report = pair_files(self.root, "Training", skip_missing=True)
        self.assertTrue(caught)
        self.assertEqual([(a.name, b.name) for a, b in pairs], [("a.png", "a.png"), ("c.png", "c.png")])
        self.assertEqual(report["missing_labels"], ["b.png"])
        self.assertEqual(report["paired_count"], 2)

    def test_red_channel_nearest_resize_and_255_policy(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        labels = np.zeros_like(image)
        labels[:, :, 0] = [[0, 12], [255, 4]]
        self.put("RGB", "a.png", image)
        self.put("Label", "a.png", labels)
        data = CarlaDataset(self.root, height=32, width=48)
        rgb, target = data[0]
        self.assertEqual(tuple(rgb.shape), (3, 32, 48))
        self.assertEqual(tuple(target.shape), (32, 48))
        self.assertEqual(target.dtype, torch.int64)
        self.assertEqual(set(target.unique().tolist()), {0, 4, 12})
        self.assertAlmostEqual(rgb[0, 0, 0].item(), -0.485 / 0.229, places=5)
        self.assertEqual(target[31, 0].item(), 0)

    def test_bad_class_ids_rejected_before_resize(self):
        self.put("Label", "bad.png", [[0, 13], [0, 0]])
        with self.assertRaisesRegex(ValueError, "Invalid class IDs"):
            load_label(self.root / "Training" / "Label" / "bad.png", (1, 1))

    def test_empty_dataset_rejected(self):
        with self.assertRaisesRegex(ValueError, "No matched"):
            pair_files(self.root, "Training")
