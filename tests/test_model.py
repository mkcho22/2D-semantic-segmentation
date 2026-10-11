import hashlib
from pathlib import Path
import tempfile
import unittest

import torch

from carla_seg.engine import load_model
from carla_seg.model import UNet


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_submission_state_schema_and_parameter_count(self):
        model = UNet()
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 23_381_901)
        # Fingerprint captured from the December 2025 submitted utils.UNet.
        schema = "\n".join(f"{key}:{tuple(value.shape)}" for key, value in model.state_dict().items())
        self.assertEqual(len(model.state_dict()), 136)
        self.assertEqual(hashlib.sha256(schema.encode()).hexdigest(), "e876a907cf2a17d99b3f919b80cb167e011639cb4f1cbf76f152cb4d4080d68d")

    def test_odd_input_size_and_requested_class_count(self):
        model = UNet(n_classes=7).eval()
        with torch.inference_mode():
            logits = model(torch.randn(1, 3, 33, 47))
        self.assertEqual(tuple(logits.shape), (1, 7, 33, 47))
        self.assertTrue(torch.isfinite(logits).all())

    def test_safe_checkpoint_loading_preserves_outputs(self):
        torch.manual_seed(4)
        model = UNet().eval()
        sample = torch.randn(1, 3, 32, 32)
        with torch.inference_mode():
            expected = model(sample)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "weights.pt"
            for payload in (model.state_dict(), {"model_state": model.state_dict(), "epoch": 1}):
                torch.save(payload, checkpoint)
                loaded = load_model(checkpoint, torch.device("cpu"))
                with torch.inference_mode():
                    actual = loaded(sample)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
