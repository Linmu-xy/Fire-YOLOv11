"""ACR paired screening launcher; defaults to preflight, --execute trains."""
from __future__ import annotations

import argparse
import json
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
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true")
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--plan", action="store_true", help="Print protocol without dataset/GPU checks")
    args = parser.parse_args(argv)
    cfg = configuration()
    spec = plan(cfg, "dfire", args.method, args.seed, args.tag)
    print(json.dumps(spec, ensure_ascii=False, indent=2))
    if args.plan:
        return 0
    # The existing launcher checks paired seed, data, weights, graph and optimizer.
    notes, errors = preflight(cfg, spec, None)
    for note in notes:
        print("[OK]", note)
    for error in errors:
        print("[FAIL]", error)
    if errors or not args.execute:
        print("Training not started.")
        return int(bool(errors))
    print(json.dumps(train(cfg, "dfire", args.method, args.seed, args.tag, execute=True), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
