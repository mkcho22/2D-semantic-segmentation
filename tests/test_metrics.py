import math
import unittest

import torch

from carla_seg.metrics import SegmentationMetrics, hybrid_loss, multiclass_dice_loss


class MetricTests(unittest.TestCase):
    def test_global_iou_and_absent_class_policy(self):
        metric = SegmentationMetrics(4)
        metric.update(torch.tensor([0, 1]), torch.tensor([0, 0]))
        metric.update(torch.tensor([1, 2]), torch.tensor([1, 1]))
        report = metric.compute()
        self.assertAlmostEqual(report["miou"], (0.5 + 1 / 3 + 0) / 3)
        self.assertAlmostEqual(report["miou_percent"], report["miou"] * 100)
        self.assertEqual(report["per_class_iou"], [0.5, 1 / 3, 0.0, None])
        self.assertEqual(report["classes_in_macro_average"], [0, 1, 2])
        self.assertEqual(report["pixel_accuracy"], 0.5)

    def test_only_class_zero_can_be_perfect(self):
        metric = SegmentationMetrics(13)
        metric.update(torch.zeros(1, 3, 4, dtype=torch.long), torch.zeros(1, 3, 4, dtype=torch.long))
        self.assertEqual(metric.compute()["miou"], 1.0)

    def test_out_of_range_and_empty_metric_rejected(self):
        metric = SegmentationMetrics(3)
        with self.assertRaisesRegex(ValueError, "before observing"):
            metric.compute()
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            metric.update(torch.tensor([3]), torch.tensor([0]))

    def test_dice_absent_target_classes_and_gradient(self):
        logits = torch.zeros(1, 3, 1, 2, requires_grad=True)
        target = torch.zeros(1, 1, 2, dtype=torch.long)
        self.assertAlmostEqual(multiclass_dice_loss(logits, target).item(), 1 / 6, places=6)
        loss = hybrid_loss(logits, target)
        self.assertAlmostEqual(loss.item(), 0.5 * math.log(3) + 0.5 / 6, places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
