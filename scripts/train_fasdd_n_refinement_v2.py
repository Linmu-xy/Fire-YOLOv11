#!/usr/bin/env python3
"""FASDD n640 clean context rerun and normalized detail screening, seed0 first."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_fasdd_n_refinement import main

if __name__ == "__main__":
    raise SystemExit(main(["--config", str(Path(__file__).resolve().parents[1] / "configs/protocol-fasdd-n-refinement-v2.yaml"),
                          "--tag", "fasdd-n-refinement-v2", "--method", "detail_norm", *sys.argv[1:]]))
