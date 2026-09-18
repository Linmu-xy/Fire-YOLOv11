"""Portable paths, immutable output files and experiment provenance."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["fire", "smoke"]


def path(value):
    p = Path(value).expanduser()
    return p if p.is_absolute() else ROOT / p


def digest(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def fresh_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=False)
    return p


def git_output(*args):
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None


def provenance():
    versions = {}
    for name in ("ultralytics", "torch", "torchvision", "numpy", "Pillow", "PyYAML"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    files = sorted([*ROOT.glob("firesmoke/*.py"), *ROOT.glob("configs/**/*.yaml")])
    return {
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--porcelain"),
        "python": sys.version, "platform": platform.platform(), "versions": versions,
        "argv": sys.argv,
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in files},
    }


def offline_runtime():
    # Set before importing Ultralytics; avoid implicit model downloads/installations.
    os.environ.setdefault("YOLO_OFFLINE", "true")
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    # Deliberately keep Ultralytics' own config dir. check_font() only probes
    # USER_CONFIG_DIR/Arial.ttf (== $YOLO_CONFIG_DIR/Ultralytics/Arial.ttf) and otherwise
    # downloads it from ultralytics.com, which fails offline. The font is installed once at
    # ~/.config/Ultralytics/Arial.ttf, so overriding this to a repo-local dir would force a
    # download on every fresh clone. Matplotlib stays repo-local: only a cache, always writable.
    os.environ.pop("YOLO_CONFIG_DIR", None)
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime" / "matplotlib"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    if importlib.metadata.version("ultralytics") != "8.4.42":
        raise RuntimeError("This research adapter requires ultralytics==8.4.42; review/test before upgrading.")
