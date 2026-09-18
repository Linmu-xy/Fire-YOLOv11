"""Random-weight checkpoint roundtrip and tiny evaluation; never trains."""
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from firesmoke.common import NAMES, offline_runtime, read_json, write_json
from firesmoke.data import prepare

offline_runtime()
import torch
from ultralytics.cfg import get_cfg

from firesmoke.evaluation import evaluate, load_model
from firesmoke.models import ResearchModel, architecture


class EvaluationIntegration(unittest.TestCase):
    def test_srdg_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = ResearchModel(architecture("p2_srdg"), nc=2, verbose=False).eval()
            model.args = get_cfg()
            model.names = dict(enumerate(NAMES))
            weights = root / "srdg.pt"
            torch.save({"model": model, "train_args": {"task": "detect", "imgsz": 64},
                        "epoch": -1}, weights)
            loaded = load_model(weights)
            with torch.inference_mode():
                output = loaded.model(torch.zeros(1, 3, 64, 64))
            prediction = output[0] if isinstance(output, tuple) else output
            self.assertTrue(torch.isfinite(prediction).all())

    def test_checkpoint_evaluation_artifacts(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for split in ("train", "val", "test"):
                (root / "source/images" / split).mkdir(parents=True)
                (root / "source/labels" / split).mkdir(parents=True)
                for i in range(3):
                    Image.new("RGB", (64, 64), (60+i*30, 60, 60)).save(root / f"source/images/{split}/{i}.jpg")
                    text = f"{i} .5 .5 .2 .2\n" if i < 2 else ""
                    (root / f"source/labels/{split}/{i}.txt").write_text(text)
            cfg = {"prepared_root": str(root / "prepared"), "datasets": {"dfire": {
                "root": str(root / "source"), "layout": "dfire", "source_names": NAMES}}}
            out = prepare(cfg, "dfire")
            model = ResearchModel(architecture("baseline"), nc=2, verbose=False).eval()
            model.args = get_cfg()
            model.names = dict(enumerate(NAMES))
            (root / "run/weights").mkdir(parents=True)
            weights = root / "run/weights/best.pt"
            torch.save({"model": model, "train_args": {"task": "detect", "imgsz": 64}, "epoch": -1}, weights)
            write_json(root / "run/run.json", {"dataset": "dfire", "method": "synthetic-test", "seed": 0,
                                               "tag": "unit-test-only", "training": {}, "research": {},
                                               "data_metadata": read_json(out / "metadata.json")})
            result = evaluate(cfg, weights, "dfire", "test", root / "eval", device="cpu", imgsz=64, batch=2)
            self.assertEqual(result["image_count"], 3)
            self.assertEqual(set(result["per_class"]), {"fire", "smoke"})
            artifact = read_json(root / "eval/predictions.json")
            self.assertEqual(len(artifact["images"]), 3)
            self.assertTrue(any(not r["truth"] for r in artifact["images"]))


if __name__ == "__main__":
    unittest.main()
