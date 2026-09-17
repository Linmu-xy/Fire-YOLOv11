"""Image-level alarm calibration, operating points, and uncertainty intervals.

This does NOT measure calibration of localization or certify deployment safety.
"""
from __future__ import annotations

import math

import numpy as np

from .common import digest, path, read_json, write_json


def temperature_scale(scores, temperature):
    p = np.clip(np.asarray(scores, dtype=float), 1e-6, 1-1e-6)
    logit = (np.log(p) - np.log1p(-p)) / temperature
    return 1 / (1 + np.exp(-np.clip(logit, -60, 60)))


def fit_temperature(scores, labels):
    # Bounded 1-D log-grid; deterministic and dependency-light.
    candidates = np.exp(np.linspace(math.log(0.1), math.log(10), 401))
    losses = []
    for t in candidates:
        p = np.clip(temperature_scale(scores, t), 1e-12, 1-1e-12)
        losses.append(float(-(labels*np.log(p)+(1-labels)*np.log1p(-p)).mean()))
    return float(candidates[int(np.argmin(losses))])


def calibration_metrics(scores, labels, bins=15):
    p, y = np.asarray(scores, dtype=float), np.asarray(labels, dtype=float)
    if not len(y):
        raise ValueError("No observations")
    membership = np.minimum((p*bins).astype(int), bins-1)
    ece = sum(float(np.mean(membership == i)) * abs(float(p[membership == i].mean()-y[membership == i].mean()))
              for i in range(bins) if np.any(membership == i))
    return {"image_ECE": ece, "image_Brier": float(np.mean((p-y)**2)), "bins": bins}


def select_threshold(scores, labels, max_fpr):
    """Maximum empirical recall under a validation negative-image FPR budget.

    Compare using >= at application time, include a no-alarm threshold >1.
    Ties: lower FPR, then higher threshold. O(N log N), no quadratic sweeps.
    """
    p, y = np.asarray(scores), np.asarray(labels, dtype=int)
    positives, negatives = int(y.sum()), int((1-y).sum())
    if not positives or not negatives:
        raise ValueError("Calibration needs both positive and negative images for every class")
    order = np.argsort(-p, kind="stable")
    p, y = p[order], y[order]
    tp, fp = np.cumsum(y), np.cumsum(1-y)
    ends = np.r_[np.flatnonzero(p[:-1] != p[1:]), len(p)-1]
    best = (0.0, 0.0, float(np.nextafter(1.0, 2.0)))
    for i in ends:
        fpr, recall = float(fp[i]/negatives), float(tp[i]/positives)
        key = (recall, -fpr, float(p[i]))
        if fpr <= max_fpr and key > best:
            best = key
    return {"threshold": best[2], "validation_recall": best[0], "validation_fpr": -best[1]}


def arrays(artifact):
    scores, labels = [], []
    for row in artifact["images"]:
        scores.append([max((b[5] for b in row["predictions"] if int(b[0]) == c), default=0.0) for c in (0, 1)])
        labels.append([int(any(int(b[0]) == c for b in row["truth"])) for c in (0, 1)])
    return np.array(scores, dtype=float), np.array(labels, dtype=int)


def fit(predictions, output, max_fpr=0.01):
    if not 0 <= max_fpr < 1:
        raise ValueError("max-fpr must be in [0,1)")
    artifact = read_json(path(predictions))
    if artifact["split"] != "val":
        raise ValueError("Calibration/threshold fitting is allowed on validation only")
    if artifact["training_dataset"] != artifact["dataset"]:
        raise ValueError("v1 uses source validation only; target calibration invalidates zero-shot protocol")
    scores, labels = arrays(artifact)
    temperatures, thresholds, diagnostics = [], [], []
    for c in (0, 1):
        if len(np.unique(labels[:, c])) != 2:
            raise ValueError("Each class needs positive and negative validation images")
        t = fit_temperature(scores[:, c], labels[:, c])
        scaled = temperature_scale(scores[:, c], t)
        selected = select_threshold(scaled, labels[:, c], max_fpr)
        temperatures.append(t)
        thresholds.append(selected["threshold"])
        diagnostics.append({**selected, "before": calibration_metrics(scores[:, c], labels[:, c]),
                            "after": calibration_metrics(scaled, labels[:, c])})
    result = {"schema": 1, "task": "image_class_presence", "names": artifact["names"],
              "checkpoint_sha256": artifact["checkpoint_sha256"], "settings": artifact["settings"],
              "source_dataset": artifact["dataset"], "fit_split": "val",
              "fit_predictions_sha256": digest(path(predictions)),
              "temperatures": temperatures, "thresholds": thresholds, "max_validation_fpr": max_fpr,
              "fit_diagnostics_not_test_results": diagnostics,
              "warning": "In-sample calibration diagnostics; no guaranteed target-domain FPR control."}
    write_json(path(output), result)
    return result


def alarm_metrics(scores, labels, thresholds):
    predicted = scores >= np.asarray(thresholds)[None, :]
    result = {}
    for c, name in enumerate(("fire", "smoke")):
        positive = labels[:, c] == 1
        result[name] = {"recall": float(predicted[positive, c].mean()) if positive.any() else None,
                        "negative_image_fpr": float(predicted[~positive, c].mean()) if (~positive).any() else None}
    background = labels.sum(1) == 0
    result["background_alarm_fraction"] = float(predicted[background].any(1).mean()) if background.any() else None
    return result


def report(predictions, calibration, output, bootstrap=1000, seed=0):
    artifact, fitted = read_json(path(predictions)), read_json(path(calibration))
    for key in ("checkpoint_sha256", "settings", "names"):
        if artifact[key] != fitted[key]:
            raise ValueError(f"Calibration does not match {key}")
    if artifact["split"] != "test":
        raise ValueError("Final reliability report requires test predictions")
    if bootstrap < 1:
        raise ValueError("bootstrap must be positive")
    scores, labels = arrays(artifact)
    scaled = np.column_stack([temperature_scale(scores[:, c], fitted["temperatures"][c]) for c in (0, 1)])
    metrics = alarm_metrics(scaled, labels, fitted["thresholds"])
    background_rows = [r for r in artifact["images"] if not r["truth"]]
    false_boxes = 0
    for row in background_rows:
        for box in row["predictions"]:
            c = int(box[0])
            false_boxes += bool(temperature_scale([box[5]], fitted["temperatures"][c])[0] >= fitted["thresholds"][c])
    metrics["background_FP_per_image_at_alarm_thresholds"] = false_boxes/len(background_rows) if background_rows else None
    metrics["background_image_count"] = len(background_rows)
    metrics["calibration"] = {name: {"before": calibration_metrics(scores[:, c], labels[:, c]),
                                    "after": calibration_metrics(scaled[:, c], labels[:, c])}
                              for c, name in enumerate(("fire", "smoke"))}
    groups = [r.get("group") for r in artifact["images"]]
    grouped = all(g is not None for g in groups)
    if any(g is not None for g in groups) and not grouped:
        raise ValueError("Partial scene groups are ambiguous; annotate all groups or none")
    units = {}
    for i, g in enumerate(groups):
        units.setdefault(str(g) if grouped else str(i), []).append(i)
    unit_indices = list(units.values())
    rng = np.random.default_rng(seed)
    values = {"fire_recall": [], "smoke_recall": [], "background_alarm_fraction": []}
    for _ in range(bootstrap):
        idx = np.concatenate([unit_indices[i] for i in rng.integers(0, len(units), len(units))])
        sampled = alarm_metrics(scaled[idx], labels[idx], fitted["thresholds"])
        for key, value in (("fire_recall", sampled["fire"]["recall"]),
                           ("smoke_recall", sampled["smoke"]["recall"]),
                           ("background_alarm_fraction", sampled["background_alarm_fraction"])):
            if value is not None:
                values[key].append(value)
    intervals = {k: {"95_percentile_CI": np.quantile(v, [0.025, 0.975]).tolist() if v else None,
                     "valid_replicates": len(v)} for k, v in values.items()}
    result = {"metrics": metrics, "intervals": intervals, "bootstrap_unit": "scene" if grouped else "image",
              "bootstrap_replicates": bootstrap, "seed": seed, "dataset": artifact["dataset"],
              "checkpoint_sha256": artifact["checkpoint_sha256"], "thresholds": fitted["thresholds"],
              "predictions_sha256": digest(path(predictions)), "calibration_sha256": digest(path(calibration)),
              "warning": "Image bootstrap can underestimate uncertainty for correlated frames; zero observed alarms do not prove zero risk."}
    write_json(path(output), result)
    return result
