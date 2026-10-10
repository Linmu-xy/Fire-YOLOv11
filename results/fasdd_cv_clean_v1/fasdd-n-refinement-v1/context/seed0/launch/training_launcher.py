#!/usr/bin/env python3
"""Registered n640 screening launcher; preview by default, --execute detaches."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from firesmoke.common import digest, path, provenance, stable_digest
from firesmoke.data import load_prepared, load_protocol
from firesmoke.experiment import plan

CONFIG = ROOT / "configs/protocol-fasdd-n-refinement-v1.yaml"
DATASET = "fasdd_cv_clean_v1"
METHODS = ("baseline", "detail", "context", "full", "local_control", "full_uniform", "full_shared", "full_r32")


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(file, value):
    with Path(file).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def preflight(cfg, method, seed, tag):
    spec = plan(cfg, DATASET, method, seed, tag)
    if Path(spec["output"]).exists():
        raise FileExistsError("Existing experiment must be preserved: " + spec["output"])
    if not Path(spec["weights"]).is_file():
        raise FileNotFoundError(spec["weights"])
    base_file = path(cfg["base_data_protocol"])
    base = load_protocol(base_file)
    # Training methods have a separate recipe fingerprint. Freeze the original
    # data policy/config and content identity instead of demanding recipe equality.
    for key in ("prepared_root", "datasets", "dataset_version"):
        if cfg[key] != base[key]:
            raise ValueError("Frozen data policy differs: " + key)
    if cfg["training"] != base["training"] or cfg["weights"] != base["weights"]:
        raise ValueError("v1 screening must preserve the baseline recipe and weights")
    prepared, metadata, rows = load_prepared(cfg, DATASET)
    version = json.loads((prepared / "version.json").read_text())
    expected = cfg["dataset_version"]["portable_content_sha256"]
    if version["status"] != "built_and_integrity_verified":
        raise ValueError("Data integrity verification is incomplete")
    if not (expected == metadata["portable_content_sha256"] == version["portable_content_sha256"]):
        raise ValueError("Dataset content fingerprint mismatch")
    if digest(base_file) != version["protocol_sha256"]:
        raise ValueError("Original frozen data protocol was changed")
    if len(rows) != version["verification"]["image_count"]:
        raise ValueError("Dataset image count mismatch")
    import torch
    if str(spec["training"]["device"]) != "0" or not torch.cuda.is_available():
        raise RuntimeError("Registered screening requires CUDA GPU0")
    return spec, {"prepared_manifest_labels_splits": "passed", "images": len(rows),
                  "portable_content_sha256": expected, "frozen_data_protocol_sha256": digest(base_file),
                  "protocol_sha256": stable_digest(cfg),
                  "recipe_sha256": stable_digest({"training": spec["training"], "research": spec["research"],
                                                   "weights": spec["weights"]}),
                  "gpu": torch.cuda.get_device_name(0),
                  "weights_sha256": digest(spec["weights"])}


def archive(runtime, output):
    if not (output / "run.json").is_file():
        return
    destination = output / "launch"
    destination.mkdir(exist_ok=True)
    for name in ("training_launcher.py", "registered_protocol.yaml", "launch.json",
                 "source_snapshot.tar.gz", "process.json", "worker.json"):
        source, target = runtime / name, destination / name
        if source.is_file() and not target.exists():
            shutil.copy2(source, target)


def worker(runtime):
    launch = json.loads((runtime / "launch.json").read_text())
    spec = launch["plan"]
    output = Path(spec["output"])
    status = {"status": "failed", "worker_pid": os.getpid(), "started_at": now()}
    child = None
    try:
        for relative, expected in launch["snapshot_source_sha256"].items():
            if digest(ROOT / relative) != expected:
                raise RuntimeError("Source changed after launch snapshot: " + relative)
        config = runtime / "registered_protocol.yaml"
        preflight(load_protocol(config), spec["method"], spec["seed"], spec["tag"])
        os.environ["MPLCONFIGDIR"] = str(runtime / "matplotlib")
        command = [sys.executable, "-u", "-m", "firesmoke", "--config", str(config), "train",
                   "--dataset", DATASET, "--method", spec["method"], "--seed", str(spec["seed"]),
                   "--tag", spec["tag"], "--execute"]
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL)
        save(runtime / "worker.json", {"worker_pid": os.getpid(), "training_pid": child.pid,
                                       "started_at": now(), "command": command})
        while child.poll() is None:
            archive(runtime, output)
            time.sleep(2)
        archive(runtime, output)
        status.update(returncode=child.returncode, status="completed" if child.returncode == 0 else "failed")
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait()
        traceback.print_exc()
        status["error"] = repr(error)
    status["ended_at"] = now()
    save(runtime / "status.json", status)
    if (output / "run.json").is_file():
        archive(runtime, output)
        save(output / "launch/status.json", status)
    return int(status["status"] != "completed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default="detail")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--tag", default="fasdd-n-refinement-v1")
    parser.add_argument("--runtime-root", type=Path, default=ROOT.parent / (ROOT.name + "-runtime"))
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--plan", action="store_true")
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args.worker.resolve())
    cfg = load_protocol(CONFIG)
    if args.plan:
        print(json.dumps(plan(cfg, DATASET, args.method, args.seed, args.tag), indent=2))
        return 0
    spec, checks = preflight(cfg, args.method, args.seed, args.tag)
    print(json.dumps({"plan": spec, "preflight": checks}, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        print("Training not started. Use --execute to launch.")
        return 0
    runtime_root = args.runtime_root.expanduser().resolve()
    if runtime_root == ROOT or ROOT in runtime_root.parents:
        raise ValueError("Runtime logs must stay outside the repository")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    runtime = runtime_root / DATASET / args.tag / args.method / f"seed{args.seed}" / stamp
    runtime.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, runtime / "training_launcher.py")
    shutil.copy2(CONFIG, runtime / "registered_protocol.yaml")
    files = sorted(set([*ROOT.glob("firesmoke/*.py"), *ROOT.glob("scripts/*.py"),
                        *ROOT.glob("configs/**/*.yaml"), ROOT / "PROJECT_LAYOUT.md",
                        ROOT / "requirements.txt", ROOT / "pyproject.toml"]))
    hashes = {str(f.relative_to(ROOT)): digest(f) for f in files if f.is_file()}
    with tarfile.open(runtime / "source_snapshot.tar.gz", "w:gz") as bundle:
        for relative in hashes:
            bundle.add(ROOT / relative, arcname=relative, recursive=False)
    save(runtime / "launch.json", {"requested_at": now(), "plan": spec, "preflight": checks,
         "provenance": provenance(), "snapshot_source_sha256": hashes,
         "snapshot_archive_sha256": digest(runtime / "source_snapshot.tar.gz"),
         "runtime": str(runtime), "log": str(runtime / "train.log"),
         "purpose": "YOLO11n module screening on fixed FASDD clean v1 validation",
         "test_evaluation": "No automatic test evaluation"})
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "YOLO_OFFLINE": "true", "YOLO_AUTOINSTALL": "false",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    with (runtime / "train.log").open("xb") as log:
        process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--worker", str(runtime)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True, env=env)
    record = {"supervisor_pid": process.pid, "runtime": str(runtime), "log": str(runtime / "train.log"),
              "output": spec["output"], "launched_at": now()}
    save(runtime / "process.json", record)
    print("LAUNCHED " + json.dumps(record), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
