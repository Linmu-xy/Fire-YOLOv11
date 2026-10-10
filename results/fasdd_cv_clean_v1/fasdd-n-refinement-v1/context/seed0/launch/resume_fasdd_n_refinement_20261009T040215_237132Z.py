#!/usr/bin/env python3
"""Resume an explicitly paused FASDD n-refinement run from its last.pt."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
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

DATASET = "fasdd_cv_clean_v1"
TAG = "fasdd-n-refinement-v1"
CONFIG = ROOT / "configs/protocol-fasdd-n-refinement-v1.yaml"


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(file: Path, value: dict, *, exclusive: bool = False) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with file.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        return
    temporary = file.with_suffix(file.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, file)


def sha256(file: Path) -> str:
    hasher = hashlib.sha256()
    with file.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _snapshot_files() -> list[Path]:
    files = [*ROOT.glob("firesmoke/*.py"), *ROOT.glob("scripts/*.py"),
             *ROOT.glob("configs/**/*.yaml"), ROOT / "PROJECT_LAYOUT.md",
             ROOT / "requirements.txt", ROOT / "pyproject.toml"]
    return sorted({file for file in files if file.is_file()})


def inspect_paused_run(method: str, seed: int, *, require_paused_status: bool = True) -> dict:
    output = ROOT / "results" / DATASET / TAG / method / f"seed{seed}"
    pause_file = output / "launch/pause.json"
    status_file = output / "launch/status.json"
    if not pause_file.is_file() or not status_file.is_file():
        raise FileNotFoundError(f"No recorded paused run at {output}")
    pause = json.loads(pause_file.read_text(encoding="utf-8"))
    status = json.loads(status_file.read_text(encoding="utf-8"))
    if pause.get("status") != "paused" or pause.get("reason") != "user_requested_pause":
        raise ValueError("Only a run explicitly recorded as user-paused can be resumed")
    if require_paused_status and status.get("status") != "paused":
        raise RuntimeError(f"Run status is {status.get('status')!r}; refusing duplicate resume")
    if pause.get("method") != method or int(pause.get("seed", -1)) != seed:
        raise ValueError("Pause record method/seed does not match this request")

    original_runtime = Path(pause["runtime_run"])
    registered_protocol = original_runtime / "registered_protocol.yaml"
    original_launch_file = original_runtime / "launch.json"
    if not registered_protocol.is_file() or not original_launch_file.is_file():
        raise FileNotFoundError("Original runtime protocol or launch record is missing")
    cfg = load_protocol(registered_protocol)
    spec = plan(cfg, DATASET, method, seed, TAG)
    if Path(spec["output"]).resolve() != output.resolve():
        raise ValueError("Registered protocol resolves to a different output directory")

    original_launch = json.loads(original_launch_file.read_text(encoding="utf-8"))
    for relative, expected in original_launch.get("snapshot_source_sha256", {}).items():
        source = ROOT / relative
        if not source.is_file() or digest(source) != expected:
            raise RuntimeError(f"Training source changed since the paused run: {relative}")

    prepared, metadata, manifest_rows = load_prepared(cfg, DATASET)
    version = json.loads((prepared / "version.json").read_text(encoding="utf-8"))
    expected_data_hash = cfg["dataset_version"]["portable_content_sha256"]
    if (version.get("status") != "built_and_integrity_verified"
            or metadata.get("portable_content_sha256") != expected_data_hash
            or version.get("portable_content_sha256") != expected_data_hash
            or len(manifest_rows) != version["verification"]["image_count"]):
        raise RuntimeError("Frozen FASDD dataset fingerprint/manifest verification failed")
    base_protocol = load_protocol(path(cfg["base_data_protocol"]))
    if digest(path(cfg["base_data_protocol"])) != version["protocol_sha256"]:
        raise RuntimeError("Frozen FASDD data protocol changed since dataset construction")

    checkpoint = output / "weights/last.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_hash = sha256(checkpoint)
    if checkpoint_hash != pause.get("checkpoint_sha256"):
        raise RuntimeError("last.pt changed since it was recorded as the pause checkpoint")
    if Path(pause["checkpoint"]).resolve() != checkpoint.resolve():
        raise ValueError("Pause record points to a different checkpoint")
    with (output / "results.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("No completed epoch is recorded in results.csv")
    csv_epoch = int(float(rows[-1]["epoch"]))

    # Import custom model classes before loading the Ultralytics checkpoint.
    from firesmoke.models import ResearchTrainer  # noqa: F401
    import torch

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_epoch = int(ckpt.get("epoch", -1))
    train_args = ckpt.get("train_args", {})
    next_epoch = checkpoint_epoch + 2  # checkpoint epoch is zero-based; UI epoch is one-based
    if checkpoint_epoch < 0 or checkpoint_epoch + 1 != csv_epoch:
        raise RuntimeError(f"Checkpoint/results epoch mismatch: ckpt={checkpoint_epoch}, csv={csv_epoch}")
    if csv_epoch != int(pause.get("last_completed_epoch", -1)):
        raise RuntimeError("results.csv differs from the recorded pause epoch")
    if output.resolve() != Path(train_args.get("save_dir", "")).resolve():
        raise RuntimeError("Checkpoint save_dir does not match the existing run directory")
    if int(train_args.get("epochs", -1)) != int(spec["training"]["epochs"]):
        raise RuntimeError("Checkpoint total epoch count differs from the registered protocol")
    if Path(train_args.get("data", "")).resolve() != Path(spec["data"]).resolve():
        raise RuntimeError("Checkpoint data path differs from the registered protocol")
    for key in ("seed", "batch", "imgsz", "workers", "deterministic", "optimizer", "amp"):
        expected = spec["training"].get(key)
        if expected is not None and train_args.get(key) != expected:
            raise RuntimeError(f"Checkpoint training argument {key}={train_args.get(key)!r} != {expected!r}")

    return {"config": cfg, "spec": spec, "output": output, "pause": pause,
            "original_runtime": original_runtime, "registered_protocol": registered_protocol,
            "checkpoint": checkpoint, "checkpoint_sha256": checkpoint_hash,
            "checkpoint_epoch_zero_based": checkpoint_epoch, "last_completed_epoch": csv_epoch,
            "next_epoch": next_epoch, "last_results_row": rows[-1],
            "data_hash": expected_data_hash, "data_image_count": len(manifest_rows),
            "train_args": train_args, "base_protocol": base_protocol}


def worker(runtime: Path) -> int:
    launch = json.loads((runtime / "launch.json").read_text(encoding="utf-8"))
    output = Path(launch["output"])
    status = {"status": "running", "started_at": now(), "runtime": str(runtime),
              "method": launch["method"], "seed": launch["seed"],
              "resume_from": launch["checkpoint"], "start_epoch": launch["next_epoch"]}
    write_json(runtime / "status.json", status)
    (runtime / "start.signal").touch(exist_ok=True)
    try:
        for relative, expected in launch["snapshot_source_sha256"].items():
            if digest(ROOT / relative) != expected:
                raise RuntimeError("Resume source changed after launch snapshot: " + relative)
        state = inspect_paused_run(launch["method"], launch["seed"], require_paused_status=False)
        if state["checkpoint_sha256"] != launch["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint changed after resume launch")
        os.environ["MPLCONFIGDIR"] = str(runtime / "matplotlib")
        from firesmoke.models import ResearchTrainer
        checkpoint = str(state["checkpoint"])
        trainer = ResearchTrainer(overrides={"model": checkpoint, "resume": checkpoint},
                                  research=state["spec"]["research"])
        if trainer.save_dir.resolve() != output.resolve():
            raise RuntimeError(f"Ultralytics selected unexpected save_dir: {trainer.save_dir}")
        if Path(str(trainer.args.resume)).resolve() != state["checkpoint"].resolve():
            raise RuntimeError("Ultralytics did not accept the intended resume checkpoint")
        if trainer.args.epochs != int(state["train_args"]["epochs"]):
            raise RuntimeError("Resume would change the original total epoch count")
        write_json(runtime / "trainer_preflight.json", {
            "save_dir": str(trainer.save_dir), "resume": str(trainer.args.resume),
            "start_epoch": state["next_epoch"], "total_epochs": trainer.args.epochs,
            "batch": trainer.args.batch, "imgsz": trainer.args.imgsz,
            "device": str(trainer.args.device), "dataset_sha256": state["data_hash"],
            "checkpoint_sha256": state["checkpoint_sha256"]})
        # The original run wrote these reports with exclusive-create semantics.
        # On resume, accept an existing report only when recomputation is identical.
        import firesmoke.models as model_module
        original_write_json = model_module.write_json
        immutable_resume_reports = {
            (output / "weight_transfer.json").resolve(),
            (output / "refinement.json").resolve(),
        }

        def write_json_resume_safe(file, value):
            target = Path(file)
            if target.exists():
                # These describe the original pretrained transfer and the
                # installed module. Resume reconstructs that graph from last.pt;
                # recomputing them from last.pt would describe a different source.
                if target.resolve() in immutable_resume_reports:
                    return
                previous = json.loads(target.read_text(encoding="utf-8"))
                if stable_digest(previous) == stable_digest(value):
                    return
                try:
                    relative = target.resolve().relative_to(output.resolve())
                    backup = runtime / "existing_artifacts" / relative
                except ValueError:
                    backup = runtime / "existing_artifacts" / target.name
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, backup)
                raise RuntimeError(f"Existing report differs; preserved copy at {backup}")
            original_write_json(target, value)

        model_module.write_json = write_json_resume_safe
        status.update({"status": "training", "trainer_pid": os.getpid(), "preflight_at": now()})
        write_json(runtime / "status.json", status)
        write_json(output / "launch/status.json", status)
        trainer.train()
        with (output / "results.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        status.update({"status": "completed", "returncode": 0, "ended_at": now(),
                       "last_completed_epoch": int(float(rows[-1]["epoch"])) if rows else None})
        code = 0
    except KeyboardInterrupt:
        status.update({"status": "paused", "reason": "resume_interrupted", "ended_at": now(),
                       "returncode": -2})
        code = 130
    except BaseException as error:
        traceback.print_exc()
        status.update({"status": "failed", "error": repr(error), "ended_at": now(), "returncode": 1})
        code = 1
    write_json(runtime / "status.json", status)
    write_json(output / "launch/status.json", status)
    return code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("context", "detail"), default="context")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--runtime-root", type=Path, default=ROOT.parent / (ROOT.name + "-runtime"))
    parser.add_argument("--execute", action="store_true", help="resume; otherwise only run preflight")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.tag != TAG:
        raise ValueError(f"This resume script is registered only for tag {TAG!r}")
    if args.worker:
        while not (args.worker / "start.signal").is_file():
            time.sleep(0.1)
        return worker(args.worker.resolve())

    state = inspect_paused_run(args.method, args.seed)
    summary = {key: state[key] for key in ("output", "checkpoint", "checkpoint_sha256",
               "last_completed_epoch", "next_epoch", "last_results_row", "data_hash", "data_image_count")}
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)
    if not args.execute:
        print("Preflight only; training not started. Add --execute to resume.")
        return 0

    runtime_root = args.runtime_root.expanduser().resolve()
    if runtime_root == ROOT or ROOT in runtime_root.parents:
        raise ValueError("Runtime logs must stay outside the repository")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    runtime = runtime_root / DATASET / TAG / args.method / f"seed{args.seed}" / ("resume-" + stamp)
    runtime.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, runtime / "training_launcher.py")
    shutil.copy2(state["registered_protocol"], runtime / "registered_protocol.yaml")
    files = _snapshot_files()
    hashes = {str(file.relative_to(ROOT)): digest(file) for file in files}
    with tarfile.open(runtime / "source_snapshot.tar.gz", "w:gz") as archive:
        for relative in hashes:
            archive.add(ROOT / relative, arcname=relative, recursive=False)
    launch = {"requested_at": now(), "method": args.method, "seed": args.seed, "tag": args.tag,
              "output": str(state["output"]), "checkpoint": str(state["checkpoint"]),
              "checkpoint_sha256": state["checkpoint_sha256"],
              "last_completed_epoch": state["last_completed_epoch"], "next_epoch": state["next_epoch"],
              "dataset_sha256": state["data_hash"], "plan": state["spec"],
              "original_runtime": str(state["original_runtime"]),
              "snapshot_source_sha256": hashes, "snapshot_archive_sha256": digest(runtime / "source_snapshot.tar.gz"),
              "runtime": str(runtime), "log": str(runtime / "train.log"),
              "provenance": provenance(), "purpose": "Resume user-paused FASDD n refinement training"}
    write_json(runtime / "launch.json", launch, exclusive=True)
    output_launch = state["output"] / "launch"
    output_launch.mkdir(exist_ok=True)
    shutil.copy2(__file__, output_launch / f"resume_fasdd_n_refinement_{stamp}.py")
    write_json(output_launch / f"resume_{stamp}.json", launch, exclusive=True)
    with (runtime / "train.log").open("xb") as log:
        process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                                    "--method", args.method, "--seed", str(args.seed),
                                    "--tag", args.tag, "--worker", str(runtime)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1", "YOLO_OFFLINE": "true",
                 "YOLO_AUTOINSTALL": "false", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    record = {"status": "starting", "supervisor_pid": process.pid, "runtime": str(runtime),
              "log": str(runtime / "train.log"), "resume_from": str(state["checkpoint"]),
              "last_completed_epoch": state["last_completed_epoch"],
              "next_epoch": state["next_epoch"], "launched_at": now()}
    write_json(runtime / "process.json", record, exclusive=True)
    write_json(runtime / "worker.json", {"worker_pid": process.pid, "training_pid": process.pid,
                                          "started_at": now(), "command": process.args}, exclusive=True)
    record["status"] = "running"
    write_json(state["output"] / "launch/status.json", record)
    (runtime / "start.signal").touch(exist_ok=True)
    print("RESUME_LAUNCHED " + json.dumps(record, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
