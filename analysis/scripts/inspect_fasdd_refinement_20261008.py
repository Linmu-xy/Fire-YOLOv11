"""CPU-only checkpoint perturbation on a fixed 32-image diagnostic sample."""
import argparse
from collections import Counter, defaultdict
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--mode', choices=('detail', 'context'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, help='Optional frozen checkpoint from an earlier probe')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.repo))
    import cv2
    import numpy as np
    import torch
    torch.set_num_threads(2)
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils.metrics import box_iou
    from ultralytics.utils.nms import non_max_suppression
    import firesmoke.models
    original = args.checkpoint or args.repo / f'results/fasdd_cv_clean_v1/fasdd-n-refinement-v1/{args.mode}/seed0/weights/best.pt'
    before = original.stat()
    snapshot = args.output / 'checkpoint_snapshot.pt'
    snapshot.write_bytes(original.read_bytes())
    after = original.stat()
    if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
        raise RuntimeError('Checkpoint changed during snapshot; use a new output directory and retry')
    yolo = YOLO(str(snapshot))
    model = yolo.model.cpu().float().eval()
    head = model.model[-1]
    refinement = head.refinement
    rows = [json.loads(line) for line in (args.repo / 'data/prepared/fasdd_cv_clean_v1/manifest.jsonl').open()]
    selected = []
    for group in ('background', 'fire', 'smoke', 'both'):
        eligible = []
        for row in rows:
            classes = {int(b[0]) for b in row['boxes']}
            actual = 'background' if not classes else 'both' if classes == {0, 1} else 'fire' if classes == {0} else 'smoke'
            if row['split'] == 'val' and actual == group:
                eligible.append(row)
        eligible.sort(key=lambda r: hashlib.sha256(r['id'].encode()).hexdigest())
        selected.extend((group, row) for row in eligible[:8])
    if len(selected) != 32:
        raise RuntimeError('Diagnostic sample must contain eight images per group')
    captured = {}
    handle = head.register_forward_pre_hook(lambda module, x: captured.update(inputs=x[0]))
    ratios = defaultdict(list)
    routes = defaultdict(list)
    activations = defaultdict(list)
    changes = defaultdict(list)
    activation_handles = []
    if refinement.detail is not None:
        def activation_hook(name):
            def hook(module, inputs, output):
                activations[name + '/input_rms'].append(float(inputs[0].square().mean().sqrt()))
                activations[name + '/output_rms'].append(float(output.square().mean().sqrt()))
            return hook
        for name in ('reduce', 'mix', 'local', 'gate', 'output'):
            activation_handles.append(getattr(refinement.detail, name).register_forward_hook(activation_hook(name)))
    totals = defaultdict(Counter)
    per_image = []
    saved_forward = refinement.forward

    def count_matches(gt, cls, pred, threshold):
        if not len(gt) or not len(pred):
            return Counter()
        iou = box_iou(gt, pred[:, :4]).numpy()
        ij = np.argwhere((iou >= threshold) & (cls.numpy()[:, None] == pred[:, 5].numpy()[None, :]))
        if len(ij) == 0:
            return Counter()
        match = ij[np.argsort(iou[ij[:, 0], ij[:, 1]])[::-1]]
        match = match[np.unique(match[:, 1], return_index=True)[1]]
        match = match[np.unique(match[:, 0], return_index=True)[1]]
        return Counter(int(cls[i]) for i in match[:, 0])

    with torch.inference_mode():
        for group, row in selected:
            raw = cv2.imread(row['prepared_image'])
            if raw is None:
                raise FileNotFoundError(row['prepared_image'])
            h, w = raw.shape[:2]
            gain = min(640 / h, 640 / w)
            nw, nh = round(w * gain), round(h * gain)
            left, top = round((640 - nw) / 2 - .1), round((640 - nh) / 2 - .1)
            image = LetterBox(new_shape=(640, 640), auto=False)(image=raw)
            tensor = torch.from_numpy(np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1))).float().unsqueeze(0) / 255
            boxes = torch.tensor(row['boxes'], dtype=torch.float32).reshape(-1, 5)
            gt, cls = torch.empty(0, 4), boxes[:, 0]
            if len(boxes):
                centers = boxes[:, 1:3] * torch.tensor([w, h]) * gain + torch.tensor([left, top])
                wh = boxes[:, 3:5] * torch.tensor([w, h]) * gain
                gt = torch.cat((centers - wh / 2, centers + wh / 2), dim=1)
            prediction_on = model(tensor)[0]
            inputs = captured['inputs']
            shallow, features = (inputs[0], inputs[1:]) if refinement.needs_detail else (None, inputs)
            reg, clas = saved_forward(features, shallow)
            for level, feature in enumerate(features):
                fh, fw = feature.shape[-2:]
                foreground = torch.zeros((fh, fw), dtype=torch.bool)
                valid = torch.zeros_like(foreground)
                valid[int(top * fh / 640):int(np.ceil((top + nh) * fh / 640)), int(left * fw / 640):int(np.ceil((left + nw) * fw / 640))] = True
                for box in gt:
                    x1, y1, x2, y2 = (box * torch.tensor([fw, fh, fw, fh]) / 640).tolist()
                    foreground[max(0, int(y1)):min(fh, int(np.ceil(y2))), max(0, int(x1)):min(fw, int(np.ceil(x2)))] = True
                masks = {'gt': foreground & valid, 'background': ~foreground & valid}
                for task, refined in (('reg', reg[level]), ('cls', clas[level])):
                    for region, mask in masks.items():
                        if mask.any():
                            delta_rms = (refined - feature)[0, :, mask].square().mean().sqrt()
                            feature_rms = feature[0, :, mask].square().mean().sqrt().clamp_min(1e-9)
                            ratios[f'P{level+3}/{task}/{region}'].append(float(delta_rms / feature_rms))
                if refinement.context is not None:
                    context = refinement.context
                    reduced = [torch.nn.functional.silu(layer(x)) for layer, x in zip(context.reduce, features)]
                    aligned = [context.resize(z, j, level) for j, z in enumerate(reduced)]
                    weights = context.routes[level](torch.cat(aligned, dim=1)).softmax(1)
                    entropy = -(weights * weights.clamp_min(1e-9).log()).sum(1)[0] / np.log(3)
                    for region, mask in masks.items():
                        if mask.any():
                            routes[f'P{level+3}/{region}/normalized_entropy'].append(float(entropy[mask].mean()))
            # Perturb only this loaded checkpoint; backbone/head remain coadapted.
            refinement.forward = lambda features, detail=None: (list(features), list(features))
            prediction_off = model(tensor)[0]
            refinement.forward = saved_forward
            difference = (prediction_on - prediction_off).abs()
            changes['box_coordinate_max_abs_px'].append(float(difference[:, :4].max()))
            changes['class_probability_max_abs'].append(float(difference[:, 4:].max()))
            record = {'id': row['id'], 'group': group, 'gt': Counter(int(c) for c in cls)}
            for state, prediction in (('on', prediction_on), ('off', prediction_off)):
                pred = non_max_suppression(prediction.clone(), conf_thres=.25, iou_thres=.7, multi_label=True, nc=2, max_time_img=2.)[0]
                record[state] = {}
                for c in cls:
                    totals[f'{group}/{int(c)}']['gt'] += int(state == 'on')
                for threshold in (.5, .75):
                    matched = count_matches(gt, cls, pred, threshold)
                    record[state][f'tp{threshold}'] = matched
                    for c in (0, 1):
                        totals[f'{group}/{c}'][f'{state}_tp{threshold}'] += matched[c]
            per_image.append(record)
    handle.remove()
    for handle in activation_handles:
        handle.remove()
    report = {'created_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'mode': args.mode,
              'checkpoint_epoch_index0': yolo.ckpt.get('epoch'), 'checkpoint_sha256': hashlib.sha256(snapshot.read_bytes()).hexdigest(),
              'device': 'cpu', 'threads': 2, 'images': 32, 'selection': '8 val images per background/fire/smoke/both, ordered by SHA256(id)',
              'scope': '32-image balanced mechanism probe at square640/conf.25/NMS.7. ON/OFF perturbation of a coadapted current checkpoint; not official AP, not a retrained ablation or baseline comparison.',
              'residual_to_parent_feature_rms': {k: {'n': len(v), 'median': float(np.median(v)), 'p25': float(np.percentile(v, 25)), 'p75': float(np.percentile(v, 75))} for k, v in ratios.items()},
              'routing_entropy_normalized': {k: {'n': len(v), 'mean': float(np.mean(v))} for k, v in routes.items()},
              'detail_activation_rms': {k: {'calls': len(v), 'median': float(np.median(v))} for k, v in activations.items()},
              'prediction_on_off_difference': {k: {'images': len(v), 'median_of_per_image_max': float(np.median(v)), 'max': max(v)} for k, v in changes.items()},
              'matched_counts': dict(totals), 'per_image': per_image}
    (args.output / 'diagnostic.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'per_image'}, separators=(',', ':')))


if __name__ == '__main__':
    main()
