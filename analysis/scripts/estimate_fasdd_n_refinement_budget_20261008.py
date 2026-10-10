#!/usr/bin/env python3
"""Arithmetic estimate of a proposed module's conv weights and MACs, not profiling."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--hidden", type=int, choices=(16, 32), default=16)
    args = p.parse_args()
    ops = []
    def conv(name, cin, cout, kernel, side, groups=1, bias=False, copies=1):
        weights = cin * cout * kernel * kernel // groups
        ops.append({"name": name, "cin": cin, "cout": cout, "kernel": kernel,
                    "side_at640": side, "groups": groups, "copies": copies,
                    "parameters": copies * (weights + (cout if bias else 0)),
                    "macs": copies * weights * side * side})
    r = args.hidden
    for c, s, name in [(64, 80, "P3"), (128, 40, "P4"), (256, 20, "P5")]:
        conv(name + " context reduction", c, r, 1, s)
        conv(name + " context routing", 3*r, 3, 1, s, bias=True)
        conv(name + " context depthwise", r, r, 5, s, groups=r)
        conv(name + " independent cls/reg projections", r, c, 1, s, copies=2)
    conv("P2 detail reduction", 64, r, 1, 160)
    conv("detail after space-to-depth", 4*r, r, 1, 80)
    conv("detail depthwise", r, r, 3, 80, groups=r)
    conv("detail gate", 2*r, r, 1, 80, bias=True)
    conv("detail regression projection", r, 64, 1, 80)
    params = sum(o["parameters"] for o in ops)
    macs = sum(o["macs"] for o in ops)
    baseline = 2590230
    result = {"proposal": "P3 detail regression + P3/P4/P5 context with independent cls/reg projections",
              "input_imgsz": 640, "hidden_channels": r, "baseline_params": baseline,
              "estimated_added_params": params, "estimated_total_params": baseline + params,
              "estimated_parameter_growth_percent": 100*params/baseline,
              "added_conv_macs": macs, "added_conv_gflops_using_2_per_mac": 2*macs/1e9,
              "operations": ops,
              "excluded": ["activations", "sigmoid/softmax", "feature multiplication/addition",
                           "resizing/pooling", "space-to-depth rearrangement", "feature caching/memory traffic"],
              "notes": ["These are arithmetic estimates for a design with bias-free convs except gates, and no added normalization.",
                        "No module implementation, model forward, latency measurement, or accuracy experiment was run."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("operations", "excluded", "notes")}))


if __name__ == "__main__":
    main()
