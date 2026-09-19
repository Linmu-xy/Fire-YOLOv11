"""ACR invariants, real detector integration, and training launcher guards."""
import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from firesmoke.common import ROOT, offline_runtime
offline_runtime()
import torch
from ultralytics.utils import DEFAULT_CFG
from copy import deepcopy
from firesmoke.aligned_context import AlignedContextResidual as ACR
from firesmoke.models import ResearchModel, architecture

sys.path.insert(0, str(ROOT / "scripts"))
import train_acr

torch.set_num_threads(2)


class ACRTests(unittest.TestCase):
    def test_identity_gradients_and_rng(self):
        for mode in ("full", "no_align", "uniform", "local"):
            state = torch.random.get_rng_state()
            module = ACR(8, 4, mode)
            self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
            x = torch.randn(2, 16, 9, 11, requires_grad=True)
            self.assertTrue(torch.equal(module(x), x))
            module(x).square().mean().backward()
            self.assertGreater(module.detail_out.weight.grad.abs().sum().item(), 0)
            if mode != "no_align":
                self.assertGreater(module.semantic_out.weight.grad.abs().sum().item(), 0)
            with torch.no_grad():
                module.detail_out.weight.normal_(0, .02)
                module.semantic_out.weight.normal_(0, .02)
            module.zero_grad()
            module(x).square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            if mode != "no_align":
                self.assertGreater(module.offset_logits.weight.grad.abs().sum().item(), 0)
            if mode in ("full", "no_align"):
                self.assertGreater(module.scale_logits.weight.grad.abs().sum().item(), 0)

    def test_alignment_geometry_and_boundaries(self):
        x = torch.arange(35.).reshape(1, 1, 5, 7)
        weights = torch.zeros(1, 9, 5, 7)
        weights[:, 4] = 1
        self.assertTrue(torch.equal(ACR.align(x, weights), x))
        weights[:, 4] = 0
        weights[:, 5] = 1
        torch.testing.assert_close(ACR.align(x, weights)[..., :-1], x[..., 1:])
        random_weights = torch.randn_like(weights).softmax(1)
        torch.testing.assert_close(ACR.align(torch.ones_like(x), random_weights), torch.ones_like(x))

    def test_graph_matches_parent_at_initialization(self):
        torch.manual_seed(5)
        parent = ResearchModel(architecture("p2"), nc=2, verbose=False).eval()
        x = torch.rand(1, 3, 64, 96)
        for method in train_acr.METHODS:
            torch.manual_seed(5)
            model = ResearchModel(architecture(method), nc=2, verbose=False).eval()
            for key, value in parent.state_dict().items():
                parts = key.split(".")
                if int(parts[1]) >= 25:
                    parts[1] = str(int(parts[1]) + 1)
                self.assertTrue(torch.equal(value, model.state_dict()[".".join(parts)]), key)
            with torch.no_grad():
                torch.testing.assert_close(parent(x)[0], model(x)[0], rtol=0, atol=0)

    def test_detection_loss_and_optimizer_step(self):
        for method in train_acr.METHODS:
            model = ResearchModel(architecture(method), nc=2, verbose=False).train()
            model.args = deepcopy(DEFAULT_CFG)
            optimizer = torch.optim.SGD(model.parameters(), lr=.001)
            for empty in (False, True):
                batch = {"img": torch.rand(2, 3, 64, 64),
                         "batch_idx": torch.empty(0) if empty else torch.tensor([0.]),
                         "cls": torch.empty(0, 1) if empty else torch.tensor([[0.]]),
                         "bboxes": torch.empty(0, 4) if empty else torch.tensor([[.5, .5, .3, .3]])}
                optimizer.zero_grad()
                loss, _ = model(batch)
                self.assertTrue(torch.isfinite(loss).all())
                loss.sum().backward()
                for p in model.parameters():
                    if p.grad is not None:
                        self.assertTrue(torch.isfinite(p.grad).all())
                optimizer.step()

    def test_checkpoint_and_fuse(self):
        from ultralytics import YOLO
        model = ResearchModel(architecture("p2_acr"), nc=2, verbose=False).eval()
        with torch.no_grad():
            model.model[25].detail_out.weight.normal_(0, .01)
        x = torch.rand(1, 3, 64, 96)
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "candidate.pt"
            torch.save({"model": model, "train_args": {"task": "detect"}}, file)
            restored = YOLO(str(file)).model.eval()
            with torch.no_grad():
                torch.testing.assert_close(model(x)[0], restored(x)[0])
                restored.fuse(verbose=False)
                torch.testing.assert_close(model(x)[0], restored(x)[0], rtol=1e-4, atol=1e-4)

    def test_launcher_guards_and_protocol(self):
        from firesmoke.data import load_protocol
        parent = load_protocol("configs/protocol-v2.yaml")
        cfg = train_acr.configuration()
        for key in ("training", "datasets", "weights", "seeds"):
            self.assertEqual(parent[key], cfg[key])
        with patch.object(train_acr, "train", side_effect=AssertionError("unexpected training")), \
             contextlib.redirect_stdout(io.StringIO()):
            with patch.object(train_acr, "preflight", return_value=([], [])):
                self.assertEqual(train_acr.main([]), 0)
                self.assertEqual(train_acr.main(["--dry-run"]), 0)
            with patch.object(train_acr, "preflight", return_value=([], ["mismatch"])):
                self.assertEqual(train_acr.main(["--execute"]), 1)
            with patch.object(train_acr, "preflight", side_effect=AssertionError("plan must be offline")):
                self.assertEqual(train_acr.main(["--plan"]), 0)


if __name__ == "__main__":
    unittest.main()
