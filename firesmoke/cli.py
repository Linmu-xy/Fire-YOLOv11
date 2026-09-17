"""CLI defaults to configuration preview; training requires --execute."""
from __future__ import annotations

import argparse
import json

from .common import path
from .data import audit, load_protocol, prepare


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/protocol.yaml")
    commands = p.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Create canonical labels and symlinked dataset, sources unchanged")
    prep.add_argument("--dataset", required=True)
    aud = commands.add_parser("audit", help="Audit labels/splits; optionally decode/hash all images")
    aud.add_argument("--datasets", nargs="+", default=["dfire", "fasdd"])
    aud.add_argument("--output", required=True)
    aud.add_argument("--hashes", action="store_true")
    aud.add_argument("--radius", type=int, default=4)
    for name in ("plan", "train"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--dataset", required=True)
        cmd.add_argument("--method", default="baseline")
        cmd.add_argument("--seed", type=int, default=0)
        cmd.add_argument("--tag", default="v1")
        if name == "train":
            cmd.add_argument("--execute", action="store_true", help="Explicitly start optimization")
    check = commands.add_parser("check-model", help="CPU random-tensor forward; no training")
    check.add_argument("--method", default="baseline")
    check.add_argument("--output", required=True)
    check.add_argument("--imgsz", type=int, default=128)
    ev = commands.add_parser("evaluate")
    ev.add_argument("--weights", required=True)
    ev.add_argument("--dataset", required=True)
    ev.add_argument("--split", choices=["val", "test"], required=True)
    ev.add_argument("--output", required=True)
    ev.add_argument("--device", default="cpu")
    ev.add_argument("--imgsz", type=int, default=640)
    ev.add_argument("--batch", type=int, default=16)
    cal = commands.add_parser("calibrate")
    cal.add_argument("--predictions", required=True)
    cal.add_argument("--output", required=True)
    cal.add_argument("--max-fpr", type=float, default=0.01)
    rep = commands.add_parser("reliability")
    rep.add_argument("--predictions", required=True)
    rep.add_argument("--calibration", required=True)
    rep.add_argument("--output", required=True)
    rep.add_argument("--bootstrap", type=int, default=1000)
    rep.add_argument("--seed", type=int, default=0)
    bench = commands.add_parser("benchmark")
    bench.add_argument("--weights", required=True)
    bench.add_argument("--output", required=True)
    bench.add_argument("--device", default="cpu")
    bench.add_argument("--imgsz", type=int, default=640)
    bench.add_argument("--warmup", type=int, default=50)
    bench.add_argument("--iterations", type=int, default=200)
    bench.add_argument("--threads", type=int, default=2)
    summary = commands.add_parser("summarize")
    summary.add_argument("--metrics", nargs="+", required=True)
    summary.add_argument("--output", required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    cfg = load_protocol(args.config)
    if args.command == "prepare":
        result = {"prepared": str(prepare(cfg, args.dataset))}
    elif args.command == "audit":
        result = audit(cfg, args.datasets, args.output, args.hashes, args.radius)
    elif args.command in ("plan", "train"):
        from .experiment import train
        result = train(cfg, args.dataset, args.method, args.seed, args.tag, getattr(args, "execute", False))
    elif args.command == "check-model":
        from .experiment import check_model
        result = check_model(cfg, args.method, args.output, args.imgsz)
        result.pop("weight_transfer", None)  # full report is preserved in the file
    elif args.command == "evaluate":
        if args.batch < 1 or args.imgsz < 64 or args.imgsz % 32:
            raise ValueError("batch must be positive; imgsz must be >=64 and a multiple of 32")
        from .evaluation import evaluate
        result = evaluate(cfg, args.weights, args.dataset, args.split, args.output, args.device, args.imgsz, args.batch)
    elif args.command == "calibrate":
        from .reliability import fit
        result = fit(args.predictions, args.output, args.max_fpr)
    elif args.command == "reliability":
        from .reliability import report
        result = report(args.predictions, args.calibration, args.output, args.bootstrap, args.seed)
    elif args.command == "benchmark":
        from .evaluation import benchmark
        result = benchmark(args.weights, args.output, args.device, args.imgsz, args.warmup, args.iterations, args.threads)
        result.pop("samples_ms", None)
    elif args.command == "summarize":
        from .evaluation import summarize
        result = summarize(args.metrics, args.output, cfg["seeds"])
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
