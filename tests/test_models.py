"""CPU synthetic forward/loss/gradient checks, no optimizer steps."""
import tempfile
import unittest
from pathlib import Path

from firesmoke.common import offline_runtime

offline_runtime()
import torch
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss

from firesmoke.models import (BackgroundLoss, ResearchModel, SemanticResidualDetailGate,
                              architecture, background_penalty, shared_transfer, style_augment)

torch.set_num_threads(2)


class ModelTests(unittest.TestCase):
    def test_hard_background_loss_only_empty_images(self):
        x = torch.zeros(2, 2, 10, requires_grad=True)
        loss = background_penalty(x, torch.tensor([0]), topk=3)
        loss.backward()
        self.assertEqual(x.grad[0].abs().sum().item(), 0)
        self.assertGreater(x.grad[1].abs().sum().item(), 0)
        self.assertEqual(background_penalty(x, torch.tensor([0, 1]), 3).item(), 0)
        self.assertTrue(torch.isfinite(background_penalty(x, torch.empty(0), 32)))

    def test_style_preserves_geometry_range_and_seed(self):
        x = torch.rand(2, 3, 16, 16)
        torch.manual_seed(1)
        a = style_augment(x)
        torch.manual_seed(1)
        b = style_augment(x)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(a.shape, x.shape)
        self.assertTrue((a >= 0).all() and (a <= 1).all())

    def test_srdg_is_identity_at_initialization_and_trainable(self):
        gate = SemanticResidualDetailGate(4, 4, gain=0.5)
        x = torch.randn(2, 8, 8, 8, requires_grad=True)
        y = gate(x)
        self.assertTrue(torch.equal(x, y))
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(gate.projection.weight.grad).all())
        self.assertGreater(gate.projection.weight.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_srdg_preserves_parent_initialization_stream(self):
        torch.manual_seed(17)
        parent = ResearchModel(architecture("p2"), nc=2, verbose=False)
        torch.manual_seed(17)
        candidate = ResearchModel(architecture("p2_srdg"), nc=2, verbose=False)
        candidate_state = candidate.state_dict()
        for key, value in parent.state_dict().items():
            fields = key.split(".")
            layer = int(fields[1])
            if layer >= 25:
                fields[1] = str(layer + 1)
            paired_key = ".".join(fields)
            self.assertIn(paired_key, candidate_state)
            self.assertTrue(torch.equal(value, candidate_state[paired_key]), paired_key)

    def test_baseline_and_p2_forward_loss_and_shared_initialization(self):
        source = DetectionModel(architecture("baseline"), nc=2, verbose=False)
        for arch, strides in (("baseline", [8, 16, 32]), ("p2", [4, 8, 16, 32]),
                              ("p2_srdg", [4, 8, 16, 32]), ("p2_dcbr", [4, 8, 16, 32]),
                              ("p2_dcbr_semantic", [4, 8, 16, 32]),
                              ("p2_dcbr_whole", [4, 8, 16, 32])):
            with self.subTest(arch=arch):
                model = ResearchModel(architecture(arch), nc=2, verbose=False)
                model.args = get_cfg()
                model.research_alpha = .25
                model.research_topk = 4
                transfer = shared_transfer(model, source)
                self.assertGreater(transfer["tensors"], 0)
                self.assertTrue(all(int(k.split(".")[1]) < 23 for k in transfer["keys"]))
                self.assertEqual(model.stride.tolist(), strides)
                batch = {"img": torch.rand(2, 3, 64, 64), "batch_idx": torch.tensor([0.]),
                         "cls": torch.tensor([[0.]]), "bboxes": torch.tensor([[.5, .5, .2, .2]])}
                preds = model(batch["img"])
                base, _ = v8DetectionLoss(model)(preds, batch)
                augmented, _ = BackgroundLoss(model)(preds, batch)
                self.assertTrue(torch.isfinite(augmented).all())
                self.assertGreater(float(augmented[1].detach()), float(base[1].detach()))
                self.assertAlmostEqual(float(base[0].detach()), float(augmented[0].detach()), places=5)
                augmented.sum().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                # Empty target batch must also remain finite.
                batch.update(batch_idx=torch.empty(0), cls=torch.empty(0, 1), bboxes=torch.empty(0, 4))
                empty, _ = BackgroundLoss(model)(preds, batch)
                self.assertTrue(torch.isfinite(empty).all())

    def test_canonical_data_works_with_upstream_loader(self):
        from PIL import Image
        from firesmoke.data import prepare
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for split in ("train", "val", "test"):
                (root / "source/images" / split).mkdir(parents=True)
                (root / "source/labels" / split).mkdir(parents=True)
                Image.new("RGB", (64, 64), "red").save(root / f"source/images/{split}/a.jpg")
                (root / f"source/labels/{split}/a.txt").write_text("0 .5 .5 .2 .2\n")
            cfg = {"prepared_root": str(root / "prepared"), "datasets": {"dfire": {
                "root": str(root / "source"), "layout": "dfire", "source_names": ["smoke", "fire"]}}}
            out = prepare(cfg, "dfire")
            dataset = YOLODataset(img_path=str(out / "train.txt"), imgsz=64, batch_size=1,
                                  augment=False, hyp=get_cfg(), data={"names": {0: "fire", 1: "smoke"}, "nc": 2, "channels": 3})
            self.assertEqual(float(dataset[0]["cls"][0]), 1)
            self.assertEqual(tuple(dataset[0]["img"].shape), (3, 64, 64))


if __name__ == "__main__":
    unittest.main()
