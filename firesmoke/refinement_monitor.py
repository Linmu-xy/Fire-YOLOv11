"""Sparse real-batch activation/gradient diagnostics without changing training."""
from __future__ import annotations

import json
from pathlib import Path

import torch


def rms(value):
    return float(value.detach().float().square().mean().sqrt())


class RefinementMonitor:
    def __init__(self, trainer, samples_per_epoch=2):
        from ultralytics.utils.torch_utils import unwrap_model
        self.trainer = trainer
        self.module = unwrap_model(trainer.model).model[-1].refinement
        self.samples_per_epoch = samples_per_epoch
        self.active = False
        self.index = 0
        self.current = {}
        self.file = Path(trainer.save_dir) / "refinement_learning.jsonl"
        self.hooks = [self.module.register_forward_hook(self.forward)]
        for name, parameter in self.module.named_parameters():
            if parameter.requires_grad:
                self.hooks.append(parameter.register_hook(self.gradient_hook(name)))

    def gradient_hook(self, name):
        def capture(gradient):
            if self.active:
                self.current.setdefault("task_gradient_rms", {})[name] = rms(gradient)
        return capture

    def forward(self, module, inputs, output):
        if not self.active:
            return
        features = inputs[0]
        regression, classification = output
        self.current["feature_rms"] = [rms(x) for x in features]
        for name, values in (("regression", regression), ("classification", classification)):
            delta = [rms(value - feature) for value, feature in zip(values, features)]
            self.current[name + "_delta_rms"] = delta
            self.current[name + "_delta_over_feature_rms"] = [d / max(f, 1e-12) for d, f in zip(delta, self.current["feature_rms"])]

    def batch_start(self):
        self.active = self.index < self.samples_per_epoch
        self.current = {"epoch": self.trainer.epoch + 1, "sampled_batch": self.index,
                        "task_gradient_note": "Raw loss-backward parameter hook, before optimizer decay/clipping; AMP=false"}

    def batch_end(self):
        if self.active:
            self.current["parameter_rms"] = {name: rms(p) for name, p in self.module.named_parameters()}
            self.current["finite"] = all(torch.isfinite(torch.tensor(value)).all().item()
                                         for group in (self.current.get("task_gradient_rms", {}), self.current["parameter_rms"])
                                         for value in group.values())
            with self.file.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(self.current, allow_nan=False) + "\n")
        self.index += 1
        self.active = False


def attach_refinement_monitor(trainer):
    trainer.refinement_monitor = RefinementMonitor(trainer)


def start_epoch(trainer):
    trainer.refinement_monitor.index = 0


def start_batch(trainer):
    trainer.refinement_monitor.batch_start()


def end_batch(trainer):
    trainer.refinement_monitor.batch_end()
