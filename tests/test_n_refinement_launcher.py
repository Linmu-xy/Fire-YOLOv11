"""Frozen-data identity remains strict while experiment methods change."""
import copy
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import yaml
from firesmoke.common import ROOT, digest
from firesmoke.data import load_protocol

spec = importlib.util.spec_from_file_location("n_refinement_launcher", ROOT / "scripts/train_fasdd_n_refinement.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def fixture(self, folder):
        cfg = load_protocol(ROOT / "configs/protocol-fasdd-n-refinement-v1.yaml")
        base = load_protocol(ROOT / "configs/protocol-fasdd-cv-clean-v1.yaml")
        weights = folder / "weights.pt"
        weights.write_bytes(b"fixture")
        base["weights"] = cfg["weights"] = str(weights)
        cfg["results_root"] = str(folder / "results")
        base_file = folder / "frozen.yaml"
        base_file.write_text(yaml.safe_dump(base))
        cfg["base_data_protocol"] = str(base_file)
        prepared = folder / "prepared"
        prepared.mkdir()
        sha = cfg["dataset_version"]["portable_content_sha256"]
        version = {"status": "built_and_integrity_verified", "portable_content_sha256": sha,
                   "protocol_sha256": digest(base_file), "verification": {"image_count": 1}}
        (prepared / "version.json").write_text(json.dumps(version))
        return cfg, base_file, prepared, {"portable_content_sha256": sha}, [{}]

    def call(self, fixture):
        cfg, _, prepared, metadata, rows = fixture
        with patch.object(launcher, "load_prepared", return_value=(prepared, metadata, rows)), \
             patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_name", return_value="fixture GPU"):
            return launcher.preflight(cfg, "detail", 0, "fixture")

    def test_separate_method_recipe_accepts_same_frozen_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(Path(temporary))
            result, checks = self.call(fixture)
            self.assertEqual(result["method"], "detail")
            self.assertEqual(checks["prepared_manifest_labels_splits"], "passed")
            self.assertEqual(checks["frozen_data_protocol_sha256"], digest(fixture[1]))

    def test_data_policy_and_training_changes_are_rejected(self):
        for mutation in ("source_names", "content", "training"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                fixture = self.fixture(Path(temporary))
                cfg = fixture[0]
                if mutation == "source_names":
                    cfg["datasets"][launcher.DATASET]["source_names"] = ["smoke", "fire"]
                elif mutation == "content":
                    cfg["dataset_version"]["portable_content_sha256"] = "wrong"
                else:
                    cfg["training"]["epochs"] = 160
                with self.assertRaises(ValueError):
                    self.call(fixture)

    def test_frozen_config_edit_content_mismatch_and_existing_run_rejected(self):
        for mutation in ("config", "metadata", "existing"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                fixture = self.fixture(Path(temporary))
                if mutation == "config":
                    with fixture[1].open("a") as stream:
                        stream.write("\n# changed\n")
                elif mutation == "metadata":
                    fixture[3]["portable_content_sha256"] = "wrong"
                else:
                    output = launcher.plan(fixture[0], launcher.DATASET, "detail", 0, "fixture")["output"]
                    Path(output).mkdir(parents=True)
                with self.assertRaises((ValueError, FileExistsError)):
                    self.call(fixture)

    def test_plan_does_not_start_training_or_require_data(self):
        with patch.object(launcher, "preflight", side_effect=AssertionError("unexpected preflight")), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(launcher.main(["--plan", "--method", "context", "--seed", "1"]), 0)
        self.assertFalse(json.loads(output.getvalue())["training_started"])


if __name__ == "__main__":
    unittest.main()
