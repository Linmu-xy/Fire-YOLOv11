#!/usr/bin/env python3
"""Read a saved YOLO11n checkpoint and sources; no forward pass or training."""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    from ultralytics import YOLO, __version__
    net = YOLO(str(args.checkpoint)).model
    head = net.model[-1]
    layers = []
    for index, module in enumerate(net.model):
        layers.append({"index": index, "from": module.f, "type": type(module).__name__,
                       "params": sum(p.numel() for p in module.parameters()),
                       "cv2_out": getattr(getattr(getattr(module, "cv2", None), "conv", None), "out_channels", None)})
    files = list((root / "firesmoke").glob("*.py")) + list((root / "configs/models").glob("*.yaml"))
    result = {"checkpoint": str(args.checkpoint),
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "ultralytics": __version__, "total_params": sum(p.numel() for p in net.parameters()),
              "save_indices": net.save, "layers": layers,
              "head": {"class": type(head).__name__, "stride": head.stride.tolist(),
                       "nc": head.nc, "reg_max": head.reg_max, "legacy": head.legacy,
                       "end2end": head.end2end,
                       "input_channels": [branch[0].conv.in_channels for branch in head.cv2],
                       "classification_widths": [branch[0][1].conv.out_channels for branch in head.cv3]},
              "source_sha256": {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
              "forward_run": False, "training_started": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"total_params": result["total_params"], "head": result["head"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
