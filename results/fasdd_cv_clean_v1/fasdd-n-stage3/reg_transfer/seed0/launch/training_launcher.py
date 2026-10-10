#!/usr/bin/env python3
"""Stage3 n640 initialization controls. Default previews; --execute trains.

Foreground execution is deliberate; an external supervisor may capture stdout.
Protocol, code snapshot, exact transfer decisions and terminal status are saved.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from firesmoke.common import digest, offline_runtime, provenance, write_json
from firesmoke.data import load_protocol
from firesmoke.experiment import plan
from train_fasdd_n_refinement import preflight

CONFIG = ROOT / "configs/protocol-fasdd-n-stage3.yaml"
TAG = "fasdd-n-stage3"
DATASET = "fasdd_cv_clean_v1"


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("baseline_control", "reg_transfer", "compatible"), default="reg_transfer")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--plan", action="store_true")
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    cfg = load_protocol(CONFIG)
    spec = plan(cfg, DATASET, args.method, args.seed, TAG)
    spec["initialization"] = spec["research"]["initialization"]
    spec["architecture_scope"] = "Original YOLO11n; 2,590,230 parameters; no new module"
    if args.plan or not (args.dry_run or args.execute):
        print(json.dumps(spec, indent=2))
        return 0
    _, checks = preflight(cfg, args.method, args.seed, TAG)
    print(json.dumps({"plan": spec, "preflight": checks}, indent=2), flush=True)
    if not args.execute:
        return 0
    import yaml
    from firesmoke.initialization_experiments import TransferExperimentTrainer
    from firesmoke.models import architecture
    output = Path(spec["output"])
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    runtime = ROOT.parent / (ROOT.name + "-runtime") / DATASET / TAG / args.method / f"seed{args.seed}" / stamp
    runtime.mkdir(parents=True, exist_ok=False)
    os.environ["MPLCONFIGDIR"] = str(runtime / "matplotlib")
    offline_runtime()
    overrides = {**spec["training"], "model": architecture("baseline"), "data": spec["data"],
                 "pretrained": spec["weights"], "project": str(output.parent), "name": output.name,
                 "exist_ok": False, "resume": False, "task": "detect", "mode": "train", "split": "val"}
    hashes = {str(f.relative_to(ROOT)): digest(f) for pattern in ("firesmoke/*.py", "scripts/*.py", "configs/**/*.yaml")
              for f in ROOT.glob(pattern) if f.is_file()}
    with tarfile.open(runtime / "source_snapshot.tar.gz", "w:gz") as bundle:
        for relative in sorted(hashes):
            bundle.add(ROOT / relative, arcname=relative, recursive=False)
    shutil.copy2(__file__, runtime / "training_launcher.py")
    shutil.copy2(CONFIG, runtime / "registered_protocol.yaml")
    record = {"started_at": now(), "training_pid": os.getpid(), "plan": spec, "preflight": checks,
              "snapshot_source_sha256": hashes, "runtime": str(runtime), "provenance": provenance()}
    write_json(runtime / "launch.json", record)
    state = {"status": "failed", "training_pid": os.getpid(), "started_at": now()}
    try:
        trainer = TransferExperimentTrainer(overrides=overrides, research=spec["research"])
        spec.update(training_started=True, provenance=record["provenance"],
                    pretrained_sha256=digest(spec["weights"]), resolved_overrides=overrides,
                    data_metadata={"portable_content_sha256": checks["portable_content_sha256"]})
        write_json(trainer.save_dir / "run.json", spec)
        (trainer.save_dir / "protocol.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        launch = trainer.save_dir / "launch"
        launch.mkdir(exist_ok=False)
        for name in ("source_snapshot.tar.gz", "training_launcher.py", "registered_protocol.yaml", "launch.json"):
            shutil.copy2(runtime / name, launch / name)
        trainer.train()
        state["status"] = "completed"
    except BaseException as error:
        state["error"] = repr(error)
        raise
    finally:
        state["ended_at"] = now()
        write_json(runtime / "status.json", state)
        if (output / "launch").is_dir():
            write_json(output / "launch/status.json", state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
