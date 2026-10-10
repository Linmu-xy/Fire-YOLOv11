#!/usr/bin/env python3
"""FASDD_CV clean v1 baseline seed0; preview by default, --execute detaches.

Results follow PROJECT_LAYOUT.md. Runtime logs live outside the repository.
No automatic resume, test evaluation, parameter changes, or overwrite.
"""
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
from firesmoke.common import digest, path, provenance
from firesmoke.data import load_prepared, load_protocol
from firesmoke.experiment import plan

CONFIG = ROOT / "configs/protocol-fasdd-cv-clean-v1.yaml"
DATASET = "fasdd_cv_clean_v1"
METHOD = "baseline"
SEED = 0
TAG = "fasdd-cv-clean-v1"


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(filename, value):
    with Path(filename).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def preflight(cfg, config_file, tag):
    spec = plan(cfg, DATASET, METHOD, SEED, tag)
    if Path(spec["output"]).exists():
        raise FileExistsError("Existing experiment must be preserved: " + spec["output"])
    if not Path(spec["weights"]).is_file():
        raise FileNotFoundError(spec["weights"])
    prepared, metadata, rows = load_prepared(cfg, DATASET)
    version = json.loads((prepared / "version.json").read_text())
    expected = cfg["dataset_version"]["portable_content_sha256"]
    if version["status"] != "built_and_integrity_verified":
        raise ValueError("Dataset build has not passed integrity verification")
    if not (expected == metadata["portable_content_sha256"] == version["portable_content_sha256"]):
        raise ValueError("Dataset version fingerprint mismatch")
    if digest(config_file) != version["protocol_sha256"]:
        raise ValueError("Frozen data protocol changed; register another experiment protocol")
    if len(rows) != version["verification"]["image_count"]:
        raise ValueError("Dataset image count mismatch")
    import torch
    if str(spec["training"]["device"]) != "0" or not torch.cuda.is_available():
        raise RuntimeError("This registered run requires CUDA GPU 0")
    return spec, {"prepared_manifest_and_labels": "passed", "images": len(rows),
                  "portable_content_sha256": expected,
                  "gpu": torch.cuda.get_device_name(0),
                  "weights_sha256": digest(spec["weights"]),
                  "perceptual_review": version["perceptual_audit"]["review_status"]}


def archive(runtime, output):
    if not (output / "run.json").is_file():
        return False
    destination = output / "launch"
    destination.mkdir(exist_ok=True)
    for name in ("training_launcher.py", "registered_protocol.yaml", "launch.json",
                 "source_snapshot.tar.gz", "process.json", "worker.json"):
        source = runtime / name
        target = destination / name
        if source.is_file() and not target.exists():
            shutil.copy2(source, target)
    return True


def worker(runtime):
    launch = json.loads((runtime / "launch.json").read_text())
    output = Path(launch["plan"]["output"])
    status = {"status": "failed", "worker_pid": os.getpid(), "started_at": now()}
    child = None
    try:
        # Fail before training if source changed between snapshot and worker start.
        for relative, expected in launch["snapshot_source_sha256"].items():
            if digest(ROOT / relative) != expected:
                raise RuntimeError("Source changed after launch snapshot: " + relative)
        config = runtime / "registered_protocol.yaml"
        preflight(load_protocol(config), config, launch["plan"]["tag"])
        os.environ["MPLCONFIGDIR"] = str(runtime / "matplotlib")
        command = [sys.executable, "-u", "-m", "firesmoke", "--config", str(config),
                   "train", "--dataset", DATASET, "--method", METHOD, "--seed", str(SEED),
                   "--tag", launch["plan"]["tag"], "--execute"]
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL)
        save(runtime / "worker.json", {"worker_pid": os.getpid(), "training_pid": child.pid,
                                       "started_at": now(), "command": command})
        print("TRAINING_PROCESS", child.pid, flush=True)
        while child.poll() is None:
            archive(runtime, output)
            time.sleep(2)
        archive(runtime, output)
        status.update(returncode=child.returncode,
                      status="completed" if child.returncode == 0 else "failed")
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
        save(output / "launch" / "status.json", status)
    return 0 if status["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true")
    action.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tag", default=TAG, help="Use a new tag for a separate experiment")
    parser.add_argument("--runtime-root", type=Path,
                        default=ROOT.parent / (ROOT.name + "-runtime"))
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker.resolve())
    cfg = load_protocol(CONFIG)
    spec, checks = preflight(cfg, CONFIG, args.tag)
    print(json.dumps({"plan": spec, "preflight": checks}, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        print("Training not started. Pass --execute to launch in background.")
        return 0
    runtime_root = args.runtime_root.expanduser().resolve()
    if runtime_root == ROOT or ROOT in runtime_root.parents:
        raise ValueError("PROJECT_LAYOUT.md requires runtime logs outside the repository")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    runtime = runtime_root / DATASET / args.tag / METHOD / "seed0" / stamp
    runtime.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, runtime / "training_launcher.py")
    shutil.copy2(CONFIG, runtime / "registered_protocol.yaml")
    files = sorted(set([*ROOT.glob("firesmoke/*.py"), *ROOT.glob("scripts/*.py"),
                        *ROOT.glob("configs/**/*.yaml"),
                        ROOT / "requirements.txt", ROOT / "pyproject.toml",
                        ROOT / "PROJECT_LAYOUT.md"]))
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in files if p.is_file()}
    with tarfile.open(runtime / "source_snapshot.tar.gz", "w:gz") as bundle:
        for relative in source_hashes:
            bundle.add(ROOT / relative, arcname=relative, recursive=False)
    save(runtime / "launch.json", {
        "requested_at": now(), "plan": spec, "preflight": checks,
        "provenance": provenance(), "snapshot_source_sha256": source_hashes,
        "snapshot_archive_sha256": digest(runtime / "source_snapshot.tar.gz"),
        "runtime": str(runtime), "log": str(runtime / "train.log"),
        "purpose": "seed0 exploratory baseline; perceptual scene review remains pending",
        "test_evaluation": "not requested; no automatic test evaluation",
    })
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "YOLO_OFFLINE": "true",
           "YOLO_AUTOINSTALL": "false", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    with (runtime / "train.log").open("xb") as log:
        process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                                    "--worker", str(runtime)],
                                   cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True,
                                   close_fds=True, env=env)
    record = {"supervisor_pid": process.pid, "runtime": str(runtime),
              "log": str(runtime / "train.log"), "output": spec["output"],
              "launched_at": now()}
    save(runtime / "process.json", record)
    print("LAUNCHED " + json.dumps(record, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
