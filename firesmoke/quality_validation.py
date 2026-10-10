"""Native validation for frozen quality scoring, including a fixed-box control."""
from __future__ import annotations

import numpy as np
import torch
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import nms
from ultralytics.utils.metrics import box_iou


class QualityValidator(DetectionValidator):
    def __init__(self, tap, *args, **kwargs):
        self.tap = tap
        self.quality_pairs = [[], []]
        super().__init__(*args, **kwargs)

    def postprocess(self, preds):
        if self.tap.mode != "post_nms":
            return super().postprocess(preds)
        # Original scores decide every candidate, NMS winner and max_det slot.
        outputs, indices = nms.non_max_suppression(
            preds, self.args.conf, self.args.iou, nc=2, multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det, end2end=False, return_idxs=True)
        quality = self.tap.quality_logits.sigmoid().pow(self.tap.exponent)
        for i, (detections, anchors) in enumerate(zip(outputs, indices)):
            if len(detections):
                detections[:, 4] *= quality[i, detections[:, 5].long(), anchors.long()]
        return [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], "extra": x[:, 6:]}
                for x in outputs]

    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)
        if self.tap.mode == "disabled":
            return
        # This GT comparison is a reporting diagnostic after inference, and
        # never changes scores, checkpoint selection or quality inputs.
        for i, pred in enumerate(preds):
            truth = self._prepare_batch(i, batch)
            iou = box_iou(truth["bboxes"], pred["bboxes"])
            for c in range(2):
                keep = pred["cls"] == c
                same = truth["cls"] == c
                if not keep.any():
                    continue
                target = iou[same][:, keep].amax(0) if same.any() else torch.zeros_like(pred["conf"][keep])
                # Report final score versus IoU; post/pre modes have changed
                # candidate sets, so compare each separately, not as paired AP.
                self.quality_pairs[c].append(torch.stack((pred["conf"][keep], target), 1).cpu().numpy())

    def summary(self):
        box = self.metrics.box
        classes = {self.names[int(c)]: {"AP50": float(box.all_ap[i, 0]),
                    "AP75": float(box.all_ap[i, 5]), "AP50_95": float(box.all_ap[i].mean()),
                    "AP_by_IoU": box.all_ap[i].tolist()}
                   for i, c in enumerate(box.ap_class_index)}
        correlation = {}
        for c, chunks in enumerate(self.quality_pairs):
            if not chunks:
                continue
            values = np.concatenate(chunks)
            value = float(np.corrcoef(values.T)[0, 1]) if min(values.std(0)) > 0 else None
            correlation[self.names[c]] = {"prediction_count": len(values), "score_IoU_Pearson": value}
        return {"mode": self.tap.mode, "mAP50": float(box.map50), "mAP50_95": float(box.map),
                "classes": classes, "score_IoU_diagnostic": correlation, "speed_ms": self.speed,
                "images": self.seen,
                "candidate_policy": "Original NMS candidates retained" if self.tap.mode == "post_nms"
                                    else "Native NMS on inference scores"}
