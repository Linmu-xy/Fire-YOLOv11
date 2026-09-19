"""DCBR invariants and paired launcher safety; no optimizer or training loop."""
import contextlib
import copy
import io
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from firesmoke.common import offline_runtime
offline_runtime()
import torch
from firesmoke.band_refinement import DetailConditionedBandResidual as DCBR
from firesmoke.models import ResearchModel, architecture
from firesmoke.data import load_protocol
from firesmoke.experiment import plan
launcher_spec = importlib.util.spec_from_file_location(
    "train_dcbr", Path(__file__).resolve().parents[1] / "scripts/train_dcbr.py")
train_dcbr = importlib.util.module_from_spec(launcher_spec)
launcher_spec.loader.exec_module(train_dcbr)

torch.set_num_threads(2)


class DCBRTests(unittest.TestCase):
    def test_identity_and_output_projection_gradient(self):
        for mode in ("joint", "semantic", "whole"):
            with self.subTest(mode=mode):
                module = DCBR(4, 8, hidden=4, groups=2, mode=mode)
                x = torch.randn(2, 12, 9, 11, requires_grad=True)
                self.assertTrue(torch.equal(module(x), x))
                module(x).square().mean().backward()
                self.assertGreater(module.projection.weight.grad.abs().sum().item(), 0)
                self.assertTrue(torch.isfinite(x.grad).all())
                # After the output projection opens, context layers receive gradients.
                with torch.no_grad():
                    module.projection.weight.fill_(0.03)
                module.zero_grad()
                module(x).square().mean().backward()
                self.assertGreater(module.context[0].weight.grad.abs().sum().item(), 0)

    def test_decomposition_constant_signal_and_bounded_residual(self):
        module = DCBR(4, 8, hidden=4, groups=2)
        for height, width in ((1, 1), (3, 5), (9, 11)):
            detail = torch.randn(2, 8, height, width)
            low, mid, high = module.decompose(detail)
            torch.testing.assert_close(low + mid + high, detail)
            with torch.no_grad():
                module.projection.bias.copy_(torch.tensor([10., -10., -10., 10.]))
            x = torch.cat((torch.randn(2, 4, height, width), detail), 1)
            y = module(x)
            torch.testing.assert_close(y[:, :4], x[:, :4], rtol=0, atol=0)
            bound = module.gain * (mid.abs() + high.abs())
            self.assertTrue(((y[:, 4:] - detail).abs() <= bound + 1e-6).all())
            constant = torch.full_like(x, 2.)
            self.assertTrue(torch.equal(module(constant), constant))
        self.assertGreater(module.diagnostics(x)["gate_saturation_fraction"], 0)

    def test_ablation_isolates_band_modulation_and_local_conditioning(self):
        joint = DCBR(4, 8, hidden=4, groups=2)
        semantic = DCBR(4, 8, hidden=4, groups=2, mode="semantic")
        whole = DCBR(4, 8, hidden=4, groups=2, mode="whole")
        with torch.no_grad():
            joint.projection.weight.fill_(0.1)
        semantic.load_state_dict(joint.state_dict())
        whole.load_state_dict(joint.state_dict())
        x = torch.randn(2, 12, 9, 11)
        alternate = x.clone()
        alternate[:, 4:] += 3
        self.assertTrue(torch.equal(semantic._parts(x)[5], semantic._parts(alternate)[5]))
        self.assertFalse(torch.equal(joint._parts(x)[5], joint._parts(alternate)[5]))
        self.assertTrue(torch.equal(joint._parts(x)[5], whole._parts(x)[5]))
        self.assertFalse(torch.equal(joint(x), whole(x)))

    def test_parent_parameters_and_predictions_match_at_initialization(self):
        torch.manual_seed(17)
        parent = ResearchModel(architecture("p2"), nc=2, verbose=False).eval()
        for name in train_dcbr.METHODS:
            torch.manual_seed(17)
            model = ResearchModel(architecture(name), nc=2, verbose=False).eval()
            state = model.state_dict()
            for key, value in parent.state_dict().items():
                parts = key.split(".")
                if int(parts[1]) >= 25:
                    parts[1] = str(int(parts[1]) + 1)
                self.assertTrue(torch.equal(value, state[".".join(parts)]), (name, key))
            with torch.inference_mode():
                x = torch.rand(1, 3, 64, 96)
                torch.testing.assert_close(parent(x)[0], model(x)[0], rtol=0, atol=0)

    def test_checkpoint_roundtrip(self):
        module = DCBR(4, 8, hidden=4, groups=2)
        with torch.no_grad():
            module.projection.bias.fill_(0.2)
        x = torch.randn(1, 12, 9, 11)
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "state.pt"
            torch.save(module.state_dict(), file)
            restored = DCBR(4, 8, hidden=4, groups=2)
            restored.load_state_dict(torch.load(file, weights_only=True))
            torch.testing.assert_close(module(x), restored(x), rtol=0, atol=0)

    def test_reject_invalid_module_settings(self):
        for kwargs in ({"groups": 3}, {"gain": 0}, {"gain": 1}, {"gain": float("nan")},
                       {"mode": "typo"}, {"hidden": 0}):
            with self.assertRaises(ValueError):
                DCBR(4, 8, **kwargs)
        with self.assertRaises(ValueError):
            DCBR(4, 8)(torch.rand(1, 11, 8, 8))


class DCBRLauncherTests(unittest.TestCase):
    def test_registry_keeps_parent_protocol(self):
        cfg = load_protocol(train_dcbr.CONFIG)
        parent = load_protocol("configs/protocol-v2.yaml")
        for key in ("datasets", "training", "weights", "prepared_root", "seeds", "background_topk"):
            self.assertEqual(cfg[key], parent[key])

    def test_missing_evidence_does_not_compare_equal(self):
        cfg = load_protocol(train_dcbr.CONFIG)
        spec = plan(cfg, "dfire", "p2_dcbr", 0, "test")
        self.assertTrue(train_dcbr.paired_errors({}, spec, {}, None))
        meta = dict(portable_inventory_sha256="x", counts={"val": 2}, names=["fire", "smoke"],
                    label_policy="exclude_invalid_images")
        parent = {**spec, "method": "p2", "pretrained_sha256": "abc", "data_metadata": meta,
                  "research": {"architecture": "p2", "background_alpha": 0., "style": False, "topk": 32}}
        self.assertEqual(train_dcbr.paired_errors(parent, spec, meta, "abc"), [])
        changed = copy.deepcopy(parent)
        changed["training"]["amp"] = True
        self.assertIn("父级 training 不一致", train_dcbr.paired_errors(changed, spec, meta, "abc"))

    def test_default_and_dry_run_never_launch_training(self):
        with patch.object(train_dcbr, "preflight", return_value=([], [])), \
             patch.object(train_dcbr.subprocess, "run", side_effect=AssertionError("training forbidden")), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(train_dcbr.main([]), 0)
            self.assertEqual(train_dcbr.main(["--dry-run", "--seed", "2"]), 0)

    def test_failed_preflight_blocks_execute(self):
        with patch.object(train_dcbr, "preflight", return_value=([], ["mismatch"])), \
             patch.object(train_dcbr.subprocess, "run", side_effect=AssertionError("training forbidden")), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(train_dcbr.main(["--execute"]), 1)
