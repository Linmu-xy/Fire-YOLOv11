#!/usr/bin/env python3
"""Validate identity, CUDA gradients, export, and actual Trainer resume on a tiny fixture."""
from __future__ import annotations
import argparse
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_OFFLINE", "true")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", choices=("context", "detail_norm"), required=True)
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    runtime = ROOT.parent / (ROOT.name + "-runtime") / ("validate_v2_" + args.pilot + "_" + stamp)
    runtime.mkdir(parents=True, exist_ok=False)
    os.environ["MPLCONFIGDIR"] = str(runtime / "matplotlib")
    report = {"status": "running", "created_at": stamp, "runtime": str(runtime),
              "pilot_method": args.pilot, "checks": [],
              "note": "Engineering fixture only; no benchmark accuracy claims."}
    def save():
        (args.output / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    save()
    try:
        import numpy as np
        import torch
        import yaml
        from ultralytics import YOLO
        from ultralytics.cfg import get_cfg
        from ultralytics.nn.tasks import load_checkpoint
        from ultralytics.utils.loss import v8DetectionLoss
        from ultralytics.utils.torch_utils import init_seeds, unwrap_model, get_flops
        from firesmoke.models import ResearchModel, ResearchTrainer, architecture, shared_transfer
        from firesmoke.detail_refinement_v2 import install_normalized_detail
        from firesmoke.checkpoint_state import restore_full_model_state

        init_seeds(0, deterministic=True)
        torch.set_num_threads(2)
        parent = ResearchModel(architecture("baseline"), nc=2, verbose=False).eval()
        shared_transfer(parent, YOLO(str(ROOT / "weights/yolo11n.pt")).model)
        parent.args = get_cfg()
        parent.names = {0: "fire", 1: "smoke"}
        candidate = copy.deepcopy(parent)
        rng = torch.get_rng_state().clone()
        spec = install_normalized_detail(candidate, 16)
        assert torch.equal(rng, torch.get_rng_state()), "Module installation changed initialization RNG"
        assert spec["new_parameters"] == 5328 and spec["total_parameters"] == 2595558
        assert candidate.model[-1].nl == 3 and candidate.model[-1].f == [2, 16, 19, 22]
        for key, tensor in parent.state_dict().items():
            assert torch.equal(tensor, candidate.state_dict()[key]), key
        for shape in ((1, 3, 64, 64), (2, 3, 96, 128)):
            x = torch.rand(shape)
            with torch.inference_mode():
                assert torch.equal(parent(x)[0], candidate(x)[0]), "Initial output differs from baseline"
        report["checks"].append({"name": "initial_identity_rng_original_head", "passed": True, "spec": spec,
                                 "gflops640": get_flops(candidate, 640)})
        save()
        train_model = copy.deepcopy(candidate).to("cuda:0").train()
        batch = {"img": torch.rand(32, 3, 640, 640, device="cuda:0"),
                 "batch_idx": torch.arange(32, device="cuda:0").float(),
                 "cls": (torch.arange(32, device="cuda:0") % 2).float().view(-1, 1),
                 "bboxes": torch.tensor([.5, .5, .08, .10], device="cuda:0").repeat(32, 1)}
        torch.cuda.reset_peak_memory_stats()
        losses, _ = v8DetectionLoss(train_model)(train_model(batch["img"]), batch)
        losses.sum().backward()
        assert torch.isfinite(losses).all()
        assert all(torch.isfinite(p.grad).all() for p in train_model.parameters() if p.grad is not None)
        output_grad = float(train_model.model[-1].refinement.output.weight.grad.float().norm())
        assert output_grad > 0, "Detail output projection has no task gradient"
        report["checks"].append({"name": "cuda_batch32_640_loss_backward", "passed": True,
                                 "loss": losses.detach().tolist(), "output_task_gradient_l2": output_grad,
                                 "peak_allocated_bytes": torch.cuda.max_memory_allocated()})
        save()
        del train_model, batch, losses
        torch.cuda.empty_cache()
        with torch.no_grad():
            torch.nn.init.normal_(candidate.model[-1].refinement.output.weight, std=.005)
        x = torch.rand(1, 3, 128, 160)
        checkpoint = runtime / "detail_norm_export.pt"
        torch.save({"model": candidate, "train_args": vars(candidate.args), "version": "8.4.42"}, checkpoint)
        restored, _ = load_checkpoint(str(checkpoint), fuse=False)
        with torch.inference_mode():
            expected = candidate(x)[0]
            assert torch.equal(expected, restored.eval()(x)[0])
            fused = copy.deepcopy(restored).fuse(verbose=False).eval()
            error = float((expected - fused(x)[0]).abs().max())
            assert torch.allclose(expected, fused(x)[0], rtol=1e-4, atol=1e-4)
        stub = ResearchTrainer.__new__(ResearchTrainer)
        stub.research = {"background_alpha": 0., "topk": 32, "style": False,
                         "refinement": "detail_norm", "refinement_hidden": 16}
        stub.resume = True
        stub.save_dir = runtime / "reconstruction"
        reconstructed = stub.get_model(cfg=restored.yaml, weights=restored, verbose=False).eval()
        with torch.inference_mode():
            assert torch.equal(restored(x)[0], reconstructed(x)[0])
        report["checks"].append({"name": "checkpoint_reload_fusion_and_core_resume", "passed": True,
                                 "fused_max_abs_error": error, "restore": stub.model_restore_report})
        save()
        if args.export:
            import onnxruntime as ort
            wrapper = YOLO(str(checkpoint))
            exported = wrapper.export(format="onnx", imgsz=(128, 160), batch=1, device="cpu",
                                      simplify=False, dynamic=False, opset=17, verbose=False)
            session = ort.InferenceSession(str(exported), providers=["CPUExecutionProvider"])
            sample = torch.rand(1, 3, 128, 160)
            with torch.inference_mode():
                actual = wrapper.model(sample)
                if isinstance(actual, tuple):
                    actual = actual[0]
            observed = session.run(None, {session.get_inputs()[0].name: sample.numpy()})[0]
            error = float(np.abs(observed - actual.numpy()).max())
            assert np.allclose(observed, actual.numpy(), rtol=1e-3, atol=1e-3)
            report["checks"].append({"name": "onnx_parity", "passed": True, "max_abs_error": error})
            save()
            del wrapper, session
        del parent, candidate, restored, fused, reconstructed
        research = {"background_alpha": 0., "topk": 32, "style": False,
                    "refinement": args.pilot, "refinement_hidden": 16, "monitor_refinement": True}
        fixture = runtime / "fixture"
        rows = [json.loads(line) for line in (ROOT / "data/prepared/fasdd_cv_clean_v1/manifest.jsonl").open()]
        selection = []
        for split in ("train", "val"):
            groups = {name: [] for name in ("background", "fire", "smoke", "both")}
            for row in rows:
                if row["split"] != split:
                    continue
                classes = {int(box[0]) for box in row["boxes"]}
                group = "background" if not classes else "both" if classes == {0, 1} else "fire" if classes == {0} else "smoke"
                groups[group].append(row)
            for group, members in groups.items():
                for row in sorted(members, key=lambda value: hashlib.sha256(value["id"].encode()).hexdigest())[:4]:
                    image = fixture / "images" / split / Path(row["prepared_image"]).name
                    label = fixture / "labels" / split / (image.stem + ".txt")
                    image.parent.mkdir(parents=True, exist_ok=True)
                    label.parent.mkdir(parents=True, exist_ok=True)
                    image.symlink_to(row["prepared_image"])
                    label.write_text(Path(row["prepared_label"]).read_text())
                    selection.append({"id": row["id"], "split": split, "group": group})
        assert len(selection) == 32
        data = fixture / "data.yaml"
        data.write_text(yaml.safe_dump({"path": str(fixture), "train": "images/train", "val": "images/val",
                                       "names": {0: "fire", 1: "smoke"}}, sort_keys=False))
        (runtime / "selection.json").write_text(json.dumps(selection, indent=2))
        overrides = {"model": architecture("baseline"), "pretrained": str(ROOT / "weights/yolo11n.pt"),
                     "data": str(data), "device": "0", "epochs": 3, "imgsz": 128, "batch": 4,
                     "workers": 0, "amp": False, "seed": 0, "deterministic": True,
                     "optimizer": "SGD", "momentum": .937, "weight_decay": .0005,
                     "lr0": .01, "warmup_epochs": 0, "nbs": 4,
                     "mosaic": 0., "close_mosaic": 0, "plots": False,
                     "project": str(runtime), "name": "trainer_pilot", "exist_ok": False, "verbose": False}
        first = ResearchTrainer(overrides=overrides, research=research)
        pause_checkpoint = runtime / "pause_input.pt"
        def snapshot(trainer):
            if trainer.epoch == 0:
                shutil.copy2(trainer.last, pause_checkpoint)
        def stop(trainer):
            if trainer.epoch == 0:
                trainer.stop = True
        first.add_callback("on_model_save", snapshot)
        first.add_callback("on_fit_epoch_end", stop)
        first.train()
        paused = torch.load(pause_checkpoint, map_location="cpu", weights_only=False)
        assert paused["epoch"] == 0 and paused["optimizer"] is not None and paused["ema"] is not None
        paused_weights = paused["ema"].float().state_dict()
        del first
        resumed = ResearchTrainer(overrides={"model": str(pause_checkpoint), "resume": str(pause_checkpoint)}, research=research)
        resume_proof = {}
        def verify_start(trainer):
            state = unwrap_model(trainer.model).state_dict()
            assert all(torch.equal(tensor.detach().cpu(), paused_weights[key]) for key, tensor in state.items())
            ema_state = trainer.ema.ema.state_dict()
            assert all(torch.equal(tensor.detach().cpu(), paused_weights[key]) for key, tensor in ema_state.items())
            optimizer_state = trainer.optimizer.state_dict()
            matched = 0
            for key, expected in paused["optimizer"]["state"].items():
                for item, tensor in expected.items():
                    observed = optimizer_state["state"][key][item]
                    if isinstance(tensor, torch.Tensor):
                        assert torch.equal(observed.detach().cpu().float(), tensor.float()), (key, item)
                        matched += 1
                    else:
                        assert observed == tensor
            assert trainer.start_epoch == 1 and trainer.ema.updates == paused["updates"]
            resume_proof.update({"model_tensors_exact": len(state), "ema_tensors_exact": len(ema_state),
                                 "optimizer_tensors_exact": matched, "ema_updates": trainer.ema.updates,
                                 "start_epoch_zero_based": trainer.start_epoch, "passed": True})
        resumed.add_callback("on_train_start", verify_start)
        resumed.train()
        assert resume_proof.get("passed") and resumed.epoch == 2
        diagnostic = [json.loads(line) for line in (resumed.save_dir / "refinement_learning.jsonl").open()]
        assert len(diagnostic) == 6 and all(row["finite"] for row in diagnostic)
        internal_grad = max(value for row in diagnostic for name, value in row.get("task_gradient_rms", {}).items() if "output" not in name)
        output_grad = max(value for row in diagnostic for name, value in row.get("task_gradient_rms", {}).items() if "output" in name)
        assert internal_grad > 0 and output_grad > 0
        report["checks"].append({"name": "actual_trainer_pause_copy_resume", "passed": True,
                                 "restore": resume_proof, "fixture_train_images": 16, "fixture_val_images": 16,
                                 "completed_epochs": 3, "diagnostic_rows": len(diagnostic),
                                 "max_internal_task_gradient_rms": internal_grad,
                                 "max_output_task_gradient_rms": output_grad,
                                 "note": "Checks checkpoint state continuity; does not claim bitwise equality of whole interrupted/uninterrupted trajectories."})
        report["status"] = "passed"
        report["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        save()
        print("VALIDATION_PASSED " + str(args.output / "validation.json"), flush=True)
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        save()
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
