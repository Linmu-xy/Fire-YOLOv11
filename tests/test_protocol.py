"""Small synthetic tests. No optimization, real-data training, or downloads."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from firesmoke.cli import main
from firesmoke.common import NAMES, read_json, write_json
from firesmoke.data import BKTree, audit, inventory, load_prepared, load_protocol, prepare, read_labels
from firesmoke.evaluation import summarize
from firesmoke.reliability import alarm_metrics, calibration_metrics, fit, report, select_threshold, temperature_scale


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg = {"prepared_root": str(self.root / "prepared"), "datasets": {"dfire": {
            "root": str(self.root / "source"), "layout": "dfire", "source_names": ["smoke", "fire"]}}}
        for s in ("train", "val", "test"):
            images = self.root / "source/images" / s
            labels = self.root / "source/labels" / s
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            Image.new("RGB", (32, 32), "red").save(images / "a.jpg")
            (labels / "a.txt").write_text("0 0.5 0.5 0.2 0.3\n")
            Image.new("RGB", (32, 32), "blue").save(images / "b.jpg")
            (labels / "b.txt").write_text("")

    def tearDown(self):
        self.temp.cleanup()

    def test_remap_empty_and_immutable_sources(self):
        source = self.root / "source/labels/train/a.txt"
        original = source.read_bytes()
        out = prepare(self.cfg, "dfire")
        self.assertEqual(read_labels(out / "labels/train/a.txt")[0][0], 1)
        self.assertEqual(read_labels(out / "labels/train/b.txt"), [])
        self.assertEqual(source.read_bytes(), original)
        self.assertTrue((out / "images/train/a.jpg").is_symlink())
        _, meta, rows = load_prepared(self.cfg, "dfire")
        self.assertEqual(meta["counts"], {"train": 2, "val": 2, "test": 2})
        self.assertEqual(len(rows), 6)
        with self.assertRaises(FileExistsError):
            prepare(self.cfg, "dfire")

    def test_manifest_label_and_split_tamper_detection(self):
        out = prepare(self.cfg, "dfire")
        f = out / "labels/train/a.txt"
        original = f.read_text()
        f.write_text("0 0.5 0.5 0.2 0.3\n")
        with self.assertRaisesRegex(ValueError, "label changed"):
            load_prepared(self.cfg, "dfire")
        f.write_text(original)
        (out / "train.txt").write_text((out / "test.txt").read_text())
        with self.assertRaisesRegex(ValueError, "split changed"):
            load_prepared(self.cfg, "dfire")

    def test_missing_label_is_not_background(self):
        (self.root / "source/labels/train/a.txt").unlink()
        with self.assertRaises(ValueError):
            inventory(self.cfg, "dfire")

    def test_degenerate_images_are_excluded_not_turned_into_negatives(self):
        self.cfg["datasets"]["dfire"]["invalid_box_policy"] = "exclude_invalid_images"
        f = self.root / "source/labels/train/a.txt"
        f.write_text("0 .5 .5 0 0\n")
        out = prepare(self.cfg, "dfire")
        self.assertEqual(len(read_json(out / "excluded.json")), 1)
        self.assertEqual(read_json(out / "metadata.json")["counts"]["train"], 1)
        self.assertNotIn("a.jpg", (out / "train.txt").read_text())
        self.assertEqual(f.read_text(), "0 .5 .5 0 0\n")

    def test_invalid_and_boundary_labels(self):
        f = self.root / "label.txt"
        for text in ("0 nan .5 .2 .3", "3 .5 .5 .2 .3", "0 .5 .5 0 .2", "0 .5 .5 .2"):
            f.write_text(text)
            with self.assertRaises(ValueError):
                read_labels(f)
        f.write_text("0 0.99 0.5 0.04 0.2")
        self.assertEqual(len(read_labels(f)), 1)  # retained and reported in audit

    def test_exact_duplicate_audit(self):
        result = audit(self.cfg, ["dfire"], self.root / "audit", hashes=True)
        self.assertTrue(result["hash_audit_completed"])
        self.assertGreater(result["candidate_pairs"], 0)
        pairs = [json.loads(s) for s in (self.root / "audit/duplicate_candidates.jsonl").read_text().splitlines()]
        self.assertTrue(any(r["byte_identical"] for r in pairs))

    def test_fasdd_crlf_and_split_overlap(self):
        root = self.root / "fasdd"
        ann = root / "annotations/YOLO_CV"
        (ann / "labels").mkdir(parents=True)
        (root / "images").mkdir()
        for s in ("train", "val", "test"):
            Image.new("RGB", (16, 16)).save(root / f"images/{s}.jpg")
            (ann / "labels" / f"{s}.txt").write_text("1 .5 .5 .2 .2\n")
            (ann / f"{s}.txt").write_bytes(f"./images/{s}.jpg\r\n".encode())
        self.cfg["datasets"]["fasdd"] = {"root": str(root), "layout": "fasdd", "source_names": NAMES}
        self.assertEqual(len(inventory(self.cfg, "fasdd")), 3)
        (ann / "test.txt").write_bytes((ann / "train.txt").read_bytes())
        with self.assertRaisesRegex(ValueError, "Repeated"):
            inventory(self.cfg, "fasdd")

    def test_bktree_matches_exhaustive_search(self):
        rng = np.random.default_rng(42)
        values = [int(v) for v in rng.integers(0, 256, 60)]
        tree = BKTree()
        for i, value in enumerate(values):
            tree.add(value, i)
        for value in (0, 1, 127, 255):
            expected = {(i, (v^value).bit_count()) for i, v in enumerate(values) if (v^value).bit_count() <= 2}
            self.assertEqual(set(tree.query(value, 2)), expected)


class ReliabilityTests(unittest.TestCase):
    def artifact(self, split="val", dataset="dfire"):
        return {"split": split, "dataset": dataset, "training_dataset": "dfire", "names": NAMES,
                "checkpoint_sha256": "checkpoint", "settings": {"imgsz": 640},
                "images": [{"id": str(i), "group": None,
                            "truth": [[c, .5, .5, .2, .2]] if i % 3 != 2 else [],
                            "predictions": [[c, .5, .5, .2, .2, .9 if i % 3 != 2 else .1]]}
                           for i, c in enumerate([0, 1, 0, 0, 1, 1])]}

    def test_threshold_ties_and_no_alarm_option(self):
        fitted = select_threshold([.9, .9, .8], [1, 0, 1], 0)
        self.assertGreater(fitted["threshold"], 1)
        self.assertEqual(fitted["validation_recall"], 0)
        fitted = select_threshold([.9, .7, .8], [1, 0, 1], 0)
        self.assertEqual(fitted["threshold"], .8)
        self.assertEqual(fitted["validation_recall"], 1)

    def test_temperature_and_metrics(self):
        np.testing.assert_allclose(temperature_scale([.2, .5, .8], 1), [.2, .5, .8])
        metrics = calibration_metrics([0, 1], [0, 1])
        self.assertEqual(metrics["image_Brier"], 0)
        self.assertEqual(metrics["image_ECE"], 0)
        m = alarm_metrics(np.array([[.8, .1], [.1, .1]]), np.array([[1, 0], [0, 0]]), [.5, .5])
        self.assertEqual(m["fire"]["recall"], 1)
        self.assertEqual(m["background_alarm_fraction"], 0)

    def test_calibration_blocks_test_target_and_checkpoint_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp)
            for split, dataset in (("test", "dfire"), ("val", "fasdd")):
                f = d / f"{split}-{dataset}.json"
                write_json(f, self.artifact(split, dataset))
                with self.assertRaises(ValueError):
                    fit(f, d / "bad.json")
            write_json(d / "val.json", self.artifact())
            fit(d / "val.json", d / "cal.json")
            target = self.artifact("test", "fasdd")
            write_json(d / "test.json", target)
            result = report(d / "test.json", d / "cal.json", d / "result.json", bootstrap=20)
            self.assertEqual(result["bootstrap_unit"], "image")
            target["checkpoint_sha256"] = "different"
            write_json(d / "mismatch.json", target)
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                report(d / "mismatch.json", d / "cal.json", d / "bad.json", bootstrap=20)

    def test_scene_bootstrap(self):
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp)
            write_json(d / "val.json", self.artifact())
            fit(d / "val.json", d / "cal.json")
            target = self.artifact("test")
            for i, row in enumerate(target["images"]):
                row["group"] = str(i // 3)
            write_json(d / "test.json", target)
            result = report(d / "test.json", d / "cal.json", d / "result.json", bootstrap=20)
            self.assertEqual(result["bootstrap_unit"], "scene")


class ProtocolTests(unittest.TestCase):
    def test_train_without_execute_never_imports_trainer(self):
        with patch("firesmoke.experiment.offline_runtime", side_effect=AssertionError("must not initialize runtime")):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                main(["train", "--dataset", "dfire", "--method", "full"])
            self.assertFalse(json.loads(output.getvalue())["training_started"])

    def test_all_ablation_plans(self):
        from firesmoke.experiment import plan
        cfg = load_protocol("configs/protocol.yaml")
        for ds in cfg["datasets"]:
            for method in cfg["methods"]:
                for seed in cfg["seeds"]:
                    self.assertFalse(plan(cfg, ds, method, seed)["training_started"])

    def test_duplicate_seeds_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp)
            files = []
            for i in range(3):
                f = d / f"{i}.json"
                write_json(f, {"tag": "v1", "training_dataset": "dfire", "dataset": "dfire", "method": "baseline",
                               "split": "test", "manifest_sha256": "x", "settings": {}, "seed": 0,
                               "checkpoint_sha256": str(i), "metrics": {"mAP": .5}})
                files.append(f)
            with self.assertRaisesRegex(ValueError, "seeds"):
                summarize(files, d / "summary.json")


if __name__ == "__main__":
    unittest.main()
