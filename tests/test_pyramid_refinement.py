"""Pinned head integration: identity, gradients, shapes, fusion and serialization."""
import copy
import tempfile
import unittest
from pathlib import Path

from firesmoke.common import offline_runtime
offline_runtime()
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils.loss import v8DetectionLoss

from firesmoke.models import ResearchModel, architecture
from firesmoke.pyramid_refinement import install_refinement, RefinementDetect

torch.set_num_threads(2)


def make_model():
    torch.manual_seed(23)
    model = ResearchModel(architecture("baseline"), nc=2, verbose=False)
    model.args = get_cfg()
    model.names = {0: "fire", 1: "smoke"}
    return model


class RefinementTests(unittest.TestCase):
    def test_parameter_counts_and_original_head(self):
        for mode, added in (("detail", 4768), ("context", 23145), ("full", 26889),
                            ("local_control", 26898), ("full_uniform", 26889), ("full_shared", 19721)):
            with self.subTest(mode=mode):
                model = make_model()
                before = {k: v.clone() for k, v in model.state_dict().items()}
                rng = torch.get_rng_state().clone()
                report = install_refinement(model, mode)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertEqual(report["new_parameters"], added)
                self.assertEqual(report["total_parameters"], 2590230 + added)
                self.assertEqual(model.model[-1].nl, 3)
                self.assertEqual(model.stride.tolist(), [8, 16, 32])
                for key, value in before.items():
                    self.assertTrue(torch.equal(value, model.state_dict()[key]), key)
                self.assertEqual([b[0][1].conv.out_channels for b in model.model[-1].cv3], [64]*3)
        model = make_model()
        self.assertEqual(install_refinement(model, "full", 32)["new_parameters"], 56841)

    def test_initial_predictions_equal_parent(self):
        parent = make_model().eval()
        for mode in ("detail", "context", "full", "local_control", "full_uniform", "full_shared"):
            candidate = copy.deepcopy(parent)
            install_refinement(candidate, mode)
            for height, width in ((64, 64), (96, 128)):
                with self.subTest(mode=mode, shape=(height, width)), torch.inference_mode():
                    x = torch.randn(2, 3, height, width)
                    expected = parent(x)
                    actual = candidate(x)
                    self.assertTrue(torch.equal(expected[0], actual[0]))
                    self.assertTrue(torch.equal(expected[1]["boxes"], actual[1]["boxes"]))
                    self.assertTrue(torch.equal(expected[1]["scores"], actual[1]["scores"]))
                    self.assertEqual(len(actual[1]["feats"]), 3)

    def test_loss_gradient_and_internal_learning_after_output_opens(self):
        batch = {"img": torch.rand(2, 3, 96, 128), "batch_idx": torch.tensor([0., 1.]),
                 "cls": torch.tensor([[0.], [1.]]), "bboxes": torch.tensor([[.5, .5, .25, .3], [.5, .5, .4, .4]])}
        for mode in ("detail", "context", "full", "local_control"):
            with self.subTest(mode=mode):
                model = make_model().train()
                install_refinement(model, mode)
                loss_fn = v8DetectionLoss(model)
                optimizer = torch.optim.SGD(model.parameters(), lr=.01)
                for step in range(3):
                    optimizer.zero_grad(set_to_none=True)
                    losses, _ = loss_fn(model(batch["img"]), batch)
                    self.assertTrue(torch.isfinite(losses).all())
                    losses.sum().backward()
                    module = model.model[-1].refinement
                    self.assertTrue(all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None))
                    if step == 0:
                        output_grads = [p.grad.abs().sum().item() for n, p in module.named_parameters()
                                        if "output" in n and p.grad is not None]
                        # TAL need not assign positives to every pyramid level.
                        self.assertTrue(output_grads and any(v > 0 for v in output_grads))
                    if step == 2:
                        internal_grads = [p.grad.abs().sum().item() for n, p in module.named_parameters()
                                          if "output" not in n and p.grad is not None]
                        self.assertTrue(internal_grads and any(v > 0 for v in internal_grads))
                    optimizer.step()
                batch_empty = {**batch, "batch_idx": torch.empty(0), "cls": torch.empty(0, 1), "bboxes": torch.empty(0, 4)}
                losses, _ = loss_fn(model(batch["img"]), batch_empty)
                self.assertTrue(torch.isfinite(losses).all())

    def test_all_refinement_paths_learn_with_dense_feature_supervision(self):
        for mode in ("detail", "context", "full", "local_control"):
            with self.subTest(mode=mode):
                model = make_model()
                install_refinement(model, mode)
                module = model.model[-1].refinement
                features = [torch.randn(2, c, 16//(2**i), 16//(2**i)) for i, c in enumerate((64, 128, 256))]
                detail = torch.randn(2, 64, 32, 32) if module.needs_detail else None
                optimizer = torch.optim.SGD(module.parameters(), lr=.1)
                for step in range(3):
                    optimizer.zero_grad(set_to_none=True)
                    regression, classification = module(features, detail)
                    objective = sum(x.square().mean() for x in regression + classification)
                    objective.backward()
                    if step == 2:
                        for name, parameter in module.named_parameters():
                            self.assertIsNotNone(parameter.grad, name)
                            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
                    optimizer.step()

    def test_save_load_and_fuse(self):
        for mode in ("detail", "context", "full", "local_control"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                model = make_model().eval()
                install_refinement(model, mode)
                for name, p in model.model[-1].refinement.named_parameters():
                    if "output" in name:
                        torch.nn.init.normal_(p, std=.01)
                sample = torch.randn(1, 3, 96, 128)
                with torch.inference_mode():
                    expected = model(sample)[0]
                checkpoint = Path(temporary) / "candidate.pt"
                torch.save({"model": model, "train_args": vars(model.args), "version": "8.4.42"}, checkpoint)
                reloaded = YOLO(str(checkpoint)).model.eval()
                self.assertIsInstance(reloaded.model[-1], RefinementDetect)
                with torch.inference_mode():
                    self.assertTrue(torch.allclose(expected, reloaded(sample)[0], atol=1e-5, rtol=1e-5))
                    reloaded.fuse(verbose=False)
                    self.assertTrue(torch.allclose(expected, reloaded(sample)[0], atol=1e-4, rtol=1e-4))

    def test_invalid_architectures_and_modes_fail(self):
        for mode, hidden in (("unknown", 16), ("full", 8), ("local_control", 32)):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                install_refinement(make_model(), mode, hidden)
        with self.assertRaises(ValueError):
            install_refinement(ResearchModel(architecture("p2"), nc=2, verbose=False))
        model = make_model()
        install_refinement(model, "full")
        with self.assertRaises(ValueError):
            install_refinement(model, "full")


if __name__ == "__main__":
    unittest.main()
