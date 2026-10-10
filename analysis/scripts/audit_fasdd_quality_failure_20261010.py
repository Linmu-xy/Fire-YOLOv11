#!/usr/bin/env python3
"""Read-only quality failure audit; new artifacts, unchanged parent/readout files.

Full-val score variants are exploratory diagnostics, never new confirmatory
model results. Within-TP50 experiments require GT identity and cannot deploy.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import shutil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def association(p, q, target):
    import numpy as np
    from scipy.stats import spearmanr
    def corr(x, y):
        return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and min(x.std(), y.std()) > 1e-8 else None
    def partial(x, y, z):
        if len(x) < 4:
            return None
        matrix = np.column_stack((np.ones(len(z)), z))
        return corr(x - matrix @ np.linalg.lstsq(matrix, x, rcond=None)[0],
                    y - matrix @ np.linalg.lstsq(matrix, y, rcond=None)[0])
    return {"count": len(p), "mean_target": float(target.mean()) if len(p) else None,
            "mean_quality": float(q.mean()) if len(p) else None,
            "q_target_MAE": float(np.abs(q-target).mean()) if len(p) else None,
            "p_q_Pearson": corr(p, q), "p_IoU_Pearson": corr(p, target), "q_IoU_Pearson": corr(q, target),
            "q_IoU_partial_Pearson_given_p": partial(q, target, p),
            "q_IoU_Spearman": float(spearmanr(q, target).statistic) if len(p)>2 and min(q.std(), target.std())>1e-8 else None}


def stratified(p, q, target, classes):
    import numpy as np
    result = {}
    for c, name in enumerate(("fire", "smoke")):
        base = classes == c
        result[name] = {}
        for group, keep in {"all": base, "IoU_zero": base & (target == 0),
                            "IoU_gt_zero": base & (target > 0), "IoU_ge50": base & (target >= .5),
                            "IoU_ge75": base & (target >= .75)}.items():
            result[name][group] = association(p[keep], q[keep], target[keep])
        result[name]["target_bins"] = [{"range": [low, high], "count": int(keep.sum()),
            "mean_p": float(p[keep].mean()) if keep.any() else None,
            "mean_q": float(q[keep].mean()) if keep.any() else None}
            for low, high in ((0, .000001), (.000001, .1), (.1, .5), (.5, .75), (.75, .9), (.9, 1.000001))
            for keep in [base & (target >= low) & (target < high)]]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runtime = ROOT.parent / (ROOT.name + "-runtime") / "quality_failure_audit" / dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    runtime.mkdir(parents=True)
    os.environ["MPLCONFIGDIR"] = str(runtime / "matplotlib")
    from firesmoke.common import digest, offline_runtime, write_json
    shutil.copy2(__file__, output / "audit_source.py")
    write_json(output / "audit_source_sha256.json", {"sha256": digest(__file__)})
    offline_runtime()
    import numpy as np
    import torch
    import yaml
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics.utils import nms
    from ultralytics.utils.metrics import ap_per_class, box_iou
    from firesmoke.localization_quality import LocalizationQualityReadout, QualityTap, sampled_quality_batch
    from train_fasdd_n_quality_stage3 import prepare, cuda_batch
    import firesmoke.models
    torch.set_num_threads(4)
    cfg = yaml.safe_load((ROOT / "configs/protocol-fasdd-n-quality-stage3.yaml").read_text())
    run = ROOT / "results" / cfg["dataset"] / cfg["tag"] / cfg["method"] / "seed0"
    weight = run / "weights/best_readout.pt"
    selected = torch.load(weight, map_location="cpu", weights_only=False)
    parent_hash, quality_hash = digest(ROOT/cfg["parent"]), digest(weight)
    assert parent_hash == selected["parent_sha256"]
    parent, readout, tap, criterion, loaders, prepared, checks = prepare(cfg, runtime)
    readout.load_state_dict(selected["readout"], strict=True)
    readout.eval()
    selected_rows = []
    with torch.no_grad():
        for step, cpu in enumerate(loaders["heldout"], 1):
            batch = cuda_batch(cpu)
            parent(batch["img"])
            inputs, target, mask = sampled_quality_batch(tap, criterion, batch, cfg["training"]["topk"])
            p, q = inputs[2].sigmoid(), readout(*inputs).sigmoid()
            for c in range(2):
                selected_rows.append(np.column_stack((np.full(int(mask.sum()), c), p[:, c][mask].cpu().numpy(),
                    q[:, c][mask].cpu().numpy(), target[:, c][mask].cpu().numpy())).astype(np.float32))
        del parent, tap, criterion, loaders, inputs, p, q, target, mask, batch, cpu
    selected_array = np.concatenate(selected_rows)
    selected_summary = stratified(selected_array[:, 1], selected_array[:, 2], selected_array[:, 3], selected_array[:, 0])
    write_json(output/"train_internal_selected_distribution.json", selected_summary)
    print("TRAIN_INTERNAL " + json.dumps(selected_summary), flush=True)
    np.savez_compressed(output/"train_internal_selected.npz", values=selected_array,
                        columns=np.array(["class", "original_p", "quality_q", "target_IoU"]))
    del selected_rows, selected_array
    torch.cuda.empty_cache()
    parent = YOLO(str(ROOT/cfg["parent"])).model.float().cuda().eval()
    tap = QualityTap(parent, readout, "post_nms")
    collected = []
    risk = {"total_baseline_predictions": 0, "below_001_after_product": 0,
            "baseline_TP50_below001": 0, "baseline_TP75_below001": 0, "baseline_TP90_below001": 0}
    dense = {"anchors": 0, "class_scores_above001_original": 0, "class_scores_above001_product": 0}
    scenes = {name: {"images": 0, "gt": [0,0]} for name in ("negative", "fire_only", "smoke_only", "mixed")}
    operational = {name: {"negative_images": 0, "negative_alarm_images": 0,
                          "fire_16to64_gt": 0, "fire_16to64_recalled75": 0,
                          "mixed_smoke_gt": 0, "mixed_smoke_recalled50": 0}
                   for name in ("baseline_p", "product_pq")}
    statistic_rows = []
    edge_rows = []

    def pairs(iou, pc, gc, cutoff):
        ij = np.argwhere((gc[:, None] == pc[None, :]) & (iou >= cutoff))
        if not len(ij):
            return np.empty((0,2), dtype=int)
        ij = ij[np.argsort(iou[ij[:,0], ij[:,1]])[::-1]]
        ij = ij[np.unique(ij[:,1], return_index=True)[1]]
        return ij[np.unique(ij[:,0], return_index=True)[1]]

    class DiagnosticValidator(DetectionValidator):
        def postprocess(self, preds):
            dense["anchors"] += int(tap.raw["scores"].shape[0] * tap.raw["scores"].shape[2])
            original = tap.raw["scores"].sigmoid()
            quality = tap.quality_logits.sigmoid()
            prob = tap.raw["boxes"].reshape(original.shape[0],4,16,-1).softmax(2)
            top4 = prob.topk(4,dim=2).values
            offsets = (prob * torch.arange(16,device=prob.device).view(1,1,16,1)).sum(2).mean(1)/15
            entropy = -(prob*prob.clamp_min(1e-12).log()).sum(2).mean(1)/np.log(16)
            statistics = torch.stack((top4[:,:,0].mean(1),top4.mean((1,2)),offsets,entropy,
                                     tap.level_ids[None].expand(original.shape[0],-1).float()),2)
            dense["class_scores_above001_original"] += int((original > .001).sum())
            dense["class_scores_above001_product"] += int((original * quality > .001).sum())
            outputs, indices = nms.non_max_suppression(preds, self.args.conf, self.args.iou,
                nc=2, multi_label=True, agnostic=False, max_det=300, return_idxs=True)
            self.candidate_quality = [quality[i, x[:,5].long(), k.long()].cpu().numpy() if len(x) else np.empty(0)
                                      for i,(x,k) in enumerate(zip(outputs, indices))]
            self.candidate_statistics = [statistics[i,k.long()].cpu().numpy() if len(x) else np.empty((0,5))
                                         for i,(x,k) in enumerate(zip(outputs,indices))]
            return [{"bboxes": x[:,:4], "conf": x[:,4], "cls": x[:,5], "extra": x[:,6:]} for x in outputs]

        def update_metrics(self, preds, batch):
            super().update_metrics(preds, batch)
            for i, pred in enumerate(preds):
                truth = self._prepare_batch(i, batch)
                pc, gc = pred["cls"].cpu().numpy(), truth["cls"].cpu().numpy()
                p, q = pred["conf"].cpu().numpy(), self.candidate_quality[i]
                iou = box_iou(truth["bboxes"], pred["bboxes"]).cpu().numpy()
                same_iou = iou * (gc[:,None] == pc[None,:])
                max_iou = same_iou.max(0) if len(gc) else np.zeros(len(pc))
                tp = self._process_batch(pred, truth)["tp"]
                if isinstance(tp, torch.Tensor): tp = tp.cpu().numpy()
                collected.append((tp, p, pc, gc, q, max_iou))
                statistic_rows.append(self.candidate_statistics[i])
                lost = p*q < .001
                risk["total_baseline_predictions"] += len(p)
                risk["below_001_after_product"] += int(lost.sum())
                for index, cutoff in ((0,50),(5,75),(8,90)):
                    risk[f"baseline_TP{cutoff}_below001"] += int((lost & tp[:,index]).sum())
                scene = "negative" if not len(gc) else "mixed" if len(set(gc)) == 2 else "fire_only" if gc[0] == 0 else "smoke_only"
                scenes[scene]["images"] += 1
                for c in range(2):scenes[scene]["gt"][c] += int((gc==c).sum())
                gt_long = (truth["bboxes"][:,2:] - truth["bboxes"][:,:2]).amax(1).cpu().numpy()
                matches = pairs(iou[:,p>=.25],pc[p>=.25],gc,.1)
                if len(matches):
                    pred_ids = np.flatnonzero(p>=.25)[matches[:,1]]
                    gt_ids = matches[:,0]
                    gt = truth["bboxes"].cpu().numpy()[gt_ids]
                    pb = pred["bboxes"].cpu().numpy()[pred_ids]
                    widths = np.maximum(gt[:,2:]-gt[:,:2],1e-6)
                    errors = (pb-gt)/np.tile(widths,(1,2))
                    edge_rows.append(np.column_stack((gc[gt_ids],iou[gt_ids,pred_ids],errors)).astype(np.float32))
                small_fire = (gc==0) & (gt_long>=16) & (gt_long<64)
                for name, scores in (("baseline_p", p), ("product_pq", p*q)):
                    item = operational[name]
                    keep = scores >= .25
                    if scene=="negative":
                        item["negative_images"] += 1
                        item["negative_alarm_images"] += int(keep.any())
                    match75 = pairs(iou[:,keep], pc[keep], gc, .75)[:,0]
                    item["fire_16to64_gt"] += int(small_fire.sum())
                    item["fire_16to64_recalled75"] += int(small_fire[match75].sum())
                    if scene=="mixed":
                        item["mixed_smoke_gt"] += int((gc==1).sum())
                        match50 = pairs(iou[:,keep], pc[keep], gc, .5)[:,0]
                        item["mixed_smoke_recalled50"] += int((gc[match50]==1).sum())

    settings = {**cfg["evaluation"], "task": "detect", "mode": "val", "data": str(prepared/"data.yaml")}
    validator = DiagnosticValidator(args=settings, save_dir=output/"native_baseline")
    try:
        validator(model=parent)
    finally:
        tap.close()
    tp, p, pc, gc, q, target = [np.concatenate([r[i] for r in collected]) for i in range(6)]
    statistics=np.concatenate(statistic_rows)
    edge=np.concatenate(edge_rows)
    np.savez_compressed(output/"val_original_candidates.npz", tp=tp, p=p, pc=pc, gc=gc, q=q, target_iou=target,
                        statistics=statistics, statistic_names=np.array(['top1_mean','top4_probability_mean','expected_bin_mean_normalized','entropy_mean','level']))
    np.savez_compressed(output/"edge_errors_conf025.npz",values=edge,columns=np.array(['class','IoU','left_error_by_width','top_error_by_height','right_error_by_width','bottom_error_by_height']))
    write_json(output/"collection_metadata.json",{"risk":risk,"dense":dense,"operational":operational,"scenes":scenes})
    names={0:"fire",1:"smoke"}
    def calculate(scores):
        ap=ap_per_class(tp, scores, pc, gc, plot=False, names=names)
        return {"mAP50_95":float(ap[5].mean()), "mAP50":float(ap[5][:,0].mean()),
                "classes":{names[int(c)]:{"AP50_95":float(ap[5][i].mean()),"AP75":float(ap[5][i,5]),"AP50":float(ap[5][i,0])}
                           for i,c in enumerate(ap[6])}}
    def within_tp(rank):
        scores=p.copy()
        for c in (0,1):
            ids=np.flatnonzero((pc==c)&tp[:,0])
            order=ids[np.argsort(-rank[ids],kind="stable")]
            scores[order]=np.sort(p[ids])[::-1]
        return scores
    variants={"baseline_p":calculate(p),"product_pq":calculate(p*q),
              "product_pq_gamma025":calculate(p*q**.25),"product_pq_gamma050":calculate(p*q**.5),
              "quality_only_q":calculate(q),"GT_required_TP50_reorder_by_q":calculate(within_tp(q)),
              "GT_required_TP50_reorder_by_IoU":calculate(within_tp(target))}
    reference=json.loads((run/"evaluation/comparison.json").read_text())
    assert abs(variants["baseline_p"]["mAP50_95"] - reference["modes"]["disabled"]["mAP50_95"]) < 1e-10
    # q has FP32 rounding differences under deterministic versus default CUDA
    # kernels. Reproduced AP must agree within 0.00001 percentage points.
    assert abs(variants["product_pq"]["mAP50_95"] - reference["modes"]["post_nms"]["mAP50_95"]) < 1e-7
    for value in variants.values():value["gain_pp"] = 100*(value["mAP50_95"]-variants["baseline_p"]["mAP50_95"])
    associations = stratified(p,q,target,pc)
    associations_tp = {names[c]:association(p[keep],q[keep],target[keep]) for c in (0,1) for keep in [(pc==c)&tp[:,0]]}
    stat_tp={names[c]:{name:association(p[keep],statistics[keep,i],target[keep])
                      for i,name in enumerate(['top1_mean','top4_probability_mean','expected_bin_mean_normalized','entropy_mean'])}
             for c in (0,1) for keep in [(pc==c)&tp[:,0]]}
    edge_summary={names[c]:{'count':int((edge[:,0]==c).sum()),'median_IoU':float(np.median(edge[edge[:,0]==c,1])),
                           'median_absolute_normalized_edge_errors':np.median(np.abs(edge[edge[:,0]==c,2:]),axis=0).tolist()}
                  for c in (0,1)}
    for item in operational.values():
        item["fire_16to64_recall75"] = item["fire_16to64_recalled75"]/item["fire_16to64_gt"]
        item["mixed_smoke_recall50"] = item["mixed_smoke_recalled50"]/item["mixed_smoke_gt"]
        item["negative_alarm_rate"] = item["negative_alarm_images"]/item["negative_images"]
    result={"parent_sha256":parent_hash,"quality_sha256":quality_hash,"selected_epoch":selected["epoch"],
            "images":validator.seen,"exploratory_val_diagnostics":True,"GT_oracles_not_deployable":True,
            "train_internal_selected":selected_summary,"val_candidate_associations":associations,
            "val_TP50_associations":associations_tp,"fixed_candidate_AP":variants,
            "val_TP50_DFL_statistic_associations":stat_tp,"edge_errors_conf025":edge_summary,
            "threshold_risk_not_net_recall_loss":risk,"dense_threshold_counts":dense,
            "scenes":scenes,"operational_fixed_candidates_conf025":operational,
            "weights_unchanged":digest(ROOT/cfg["parent"])==parent_hash and digest(weight)==quality_hash}
    write_json(output/"summary.json",result)
    print("SUMMARY " + json.dumps(result),flush=True)


if __name__=="__main__":
    main()
