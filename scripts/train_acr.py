"""ACR paired screening launcher; defaults to preflight, --execute trains."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train_dcbr import preflight
from firesmoke.data import load_protocol
from firesmoke.experiment import plan, train

METHODS = ("p2_acr", "p2_acr_no_align", "p2_acr_uniform", "p2_acr_local")


def configuration():
    cfg = load_protocol(ROOT / "configs/protocol-v2.yaml")
    for method in METHODS:
        cfg["methods"][method] = dict(architecture=method, background_alpha=0., style=False)
    return cfg


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default="p2_acr")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--tag", default="module-acr-v1")
    parser.add_argument("--no-archive", action="store_true",
                        help="训练成功后不复制到 experiments/yolo11n/")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true")
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--plan", action="store_true", help="Print protocol without dataset/GPU checks")
    args = parser.parse_args(argv)
    cfg = configuration()
    spec = plan(cfg, "dfire", args.method, args.seed, args.tag)
    archive = ROOT / f"experiments/yolo11n/D-Fire_{args.method}_seed{args.seed}_{args.tag}"
    print(json.dumps(spec, ensure_ascii=False, indent=2))
    if not args.no_archive:
        print(f"成功后归档: {archive}")
    if args.plan:
        return 0
    # The existing launcher checks paired seed, data, weights, graph and optimizer.
    notes, errors = preflight(cfg, spec, None if args.no_archive else archive)
    for note in notes:
        print("[OK]", note)
    for error in errors:
        print("[FAIL]", error)
    if errors or not args.execute:
        print("Training not started.")
        return int(bool(errors))
    print(json.dumps(train(cfg, "dfire", args.method, args.seed, args.tag, execute=True), indent=2))
    if not args.no_archive:
        shutil.copytree(spec["output"], archive)
        print(f"归档完成: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
