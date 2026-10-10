"""Read existing fixed-confidence baseline predictions; no model/data changes."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def overlaps(gt, pred):
    area_g = np.maximum(gt[:, 2:] - gt[:, :2], 0).prod(1)
    area_p = np.maximum(pred[:, 2:] - pred[:, :2], 0).prod(1)
    lo = np.maximum(gt[:, None, :2], pred[None, :, :2])
    hi = np.minimum(gt[:, None, 2:], pred[None, :, 2:])
    inter = np.maximum(hi - lo, 0).prod(2)
    return inter / np.maximum(area_g[:, None] + area_p[None, :] - inter, 1e-9), inter, area_g, area_p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    counts = defaultdict(Counter)
    values = defaultdict(lambda: defaultdict(list))
    image_count = 0
    for line in args.audit.open():
        r = json.loads(line)
        image_count += 1
        gt = np.asarray(r['gt_xyxy'], dtype=float).reshape(-1, 4)
        pred = np.asarray(r['pred_xyxy'], dtype=float).reshape(-1, 4)
        gc = np.asarray(r['gt_classes']); pc = np.asarray(r['pred_classes'])
        conf = np.asarray(r['pred_conf'])
        iou, inter, ga, pa = overlaps(gt, pred)
        for c, name in enumerate(('fire', 'smoke')):
            count = counts[name]
            gids, pids = np.flatnonzero(gc == c), np.flatnonzero(pc == c)
            count['gt'] += len(gids)
            count['predictions'] += len(pids)
            count['missed_gt_one_to_one_iou50'] += sum(gc[i] == c for i in r['missed_gt'])
            for gi in gids:
                same = iou[gi, pids]
                best = float(same.max()) if same.size else 0.
                count['best_retained_iou_ge50'] += best >= .5
                count['best_retained_iou_ge75'] += best >= .75
                count['best_retained_iou_below10'] += best < .1
                count['best_retained_iou_10to50'] += .1 <= best < .5
                eligible = pids[same >= .1]
                if best >= .75:
                    count['quality_ranking_denominator_gt_with_iou75_candidate'] += 1
                    highest = eligible[np.argmax(conf[eligible])]
                    count['quality_ranking_conflict_gt'] += iou[gi, highest] < .75
                if gi not in r['missed_gt'] or best < .1:
                    continue
                pi = pids[int(np.argmax(same))]
                gw, gh = gt[gi, 2:] - gt[gi, :2]
                pw, ph = pred[pi, 2:] - pred[pi, :2]
                gcenter = (gt[gi, :2] + gt[gi, 2:]) / 2
                pcenter = (pred[pi, :2] + pred[pi, 2:]) / 2
                center = np.abs(pcenter - gcenter) / np.maximum([gw, gh], 1e-9)
                values[name]['missed_best_width_ratio'].append(float(pw / max(gw, 1e-9)))
                values[name]['missed_best_height_ratio'].append(float(ph / max(gh, 1e-9)))
                values[name]['missed_best_center_x_error_normalized'].append(float(center[0]))
                values[name]['missed_best_center_y_error_normalized'].append(float(center[1]))
                if best < .5:
                    count['missed_best_weak_geometry'] += 1
                    count['missed_weak_center_within10pct_both_axes'] += bool((center < .1).all())
                    count['missed_weak_any_pred_covers80pct_gt'] += bool((inter[gi, pids] / max(ga[gi], 1e-9) >= .8).any())
                    count['missed_weak_any_pred80pct_contained_in_gt'] += bool((inter[gi, pids] / np.maximum(pa[pids], 1e-9) >= .8).any())
            for pi in pids:
                same_gt = gc == c
                count['predictions_covering80pct_of_multiple_same_class_gts'] += int((inter[same_gt, pi] / np.maximum(ga[same_gt], 1e-9) >= .8).sum()) >= 2
                if same_gt.any():
                    values[name]['prediction_conf'].append(float(conf[pi]))
                    values[name]['prediction_best_sameclass_iou'].append(float(iou[same_gt, pi].max()))
    out = {'audit': str(args.audit), 'audit_sha256': hashlib.sha256(args.audit.read_bytes()).hexdigest(),
           'images': image_count, 'scope': 'Completed baseline seed1, val, 640, post-NMS retained predictions at conf>=0.25. Geometry/ranking diagnostics, not AP or oracle AP.', 'classes': {}}
    for name in ('fire', 'smoke'):
        count = {k: int(v) for k, v in counts[name].items()}
        summaries = {k: {'n': len(v), 'median': float(np.median(v)), 'p25': float(np.percentile(v, 25)), 'p75': float(np.percentile(v, 75))}
                     for k, v in values[name].items() if v and k not in ('prediction_conf', 'prediction_best_sameclass_iou')}
        den = count.get('quality_ranking_denominator_gt_with_iou75_candidate', 0)
        count['quality_ranking_conflict_fraction'] = count.get('quality_ranking_conflict_gt', 0) / max(den, 1)
        x, y = values[name]['prediction_conf'], values[name]['prediction_best_sameclass_iou']
        corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else None
        out['classes'][name] = {'counts': count, 'geometry': summaries, 'conf_vs_best_iou_pearson_retained_positive_images': corr}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    print(json.dumps(out, separators=(',', ':')))


if __name__ == '__main__':
    main()
