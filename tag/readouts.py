"""Transparent, inference-only semantic/instance/panoptic SAM3 readouts.

Presence probabilities weight native mask predictions exactly once. These public
single-view reference presets are independent of the withheld TAG implementation.
"""

import numpy as np
import torch
import torch.nn.functional as F


def _probabilities(raw, size, confidence):
    presence = raw['presence'].float().sigmoid().reshape(())
    scores = raw['logits'].float().sigmoid().reshape(-1) * presence
    keep = scores > confidence
    scores = scores[keep]
    if scores.numel():
        masks = F.interpolate(
            raw['masks'][keep, None].float(), size, mode='bilinear', align_corners=False
        )[:, 0].sigmoid()
    else:
        masks = raw['masks'].new_empty((0, *size), dtype=torch.float32)
    semantic = F.interpolate(
        raw['semantic'].float().reshape(1, 1, *raw['semantic'].shape[-2:]),
        size, mode='bilinear', align_corners=False,
    )[0, 0].sigmoid() * presence
    return scores, masks, semantic


def mask_nms(scores, masks, threshold):
    """Classwise greedy mask NMS; IoUs are measured at at most 512 pixels/side."""
    if not scores.numel():
        return []
    height, width = masks.shape[-2:]
    scale = min(1.0, 512 / max(height, width))
    size = (max(1, round(height * scale)), max(1, round(width * scale)))
    flat = F.interpolate(masks[:, None].float(), size, mode='nearest')[:, 0].flatten(1)
    intersection = flat @ flat.T
    area = flat.sum(1)
    iou = intersection / (area[:, None] + area[None, :] - intersection).clamp(min=1)
    order = scores.argsort(descending=True, stable=True)
    kept = []
    while order.numel():
        index = order[0]
        kept.append(int(index))
        order = order[1:][iou[index, order[1:]] <= threshold]
    return kept


def encode_mask(mask):
    from pycocotools import mask as mask_utils

    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    rle = mask_utils.encode(np.asfortranarray(mask, dtype=np.uint8))
    rle['counts'] = rle['counts'].decode('ascii')
    return rle


def panoptic_merge(candidates, config, shape):
    """Merge score-weighted masks; category IDs are class index + 1, void is 0."""
    policy = config['panoptic']
    best = torch.zeros(shape, dtype=torch.float32)
    winner = torch.full(shape, -1, dtype=torch.int64)
    for index, candidate in enumerate(candidates):
        value = candidate['probability'].float() * candidate['score']
        better = value > best
        best[better], winner[better] = value[better], index
    panoptic = np.zeros(shape, dtype=np.int32)
    segments = []
    for index, candidate in enumerate(candidates):
        full = candidate['probability'].float() > policy['mask_threshold']
        mask = (winner == index) & full
        area, original_area = int(mask.sum()), int(full.sum())
        if not area or area / max(original_area, 1) < policy['overlap']:
            continue
        segment_id = len(segments) + 1
        panoptic[mask.numpy()] = segment_id
        segments.append({'id': segment_id, 'category_id': candidate['category'] + 1})
    return panoptic, segments


@torch.inference_mode()
def predict(model, image, config, task):
    """Return semantic indices, COCO RLE instances, or a panoptic map/metadata."""
    if task not in ('semantic', 'instance', 'panoptic'):
        raise ValueError(f'Unknown task: {task}')
    size = (image.height, image.width)
    policy = config[task]
    things = set(config['thing_indices'])
    best = torch.zeros(size, dtype=torch.float32, device=model.device)
    labels = torch.full(size, config['semantic']['background'], dtype=torch.int64, device=model.device)
    instances, candidates = [], []
    classes_seen = 0
    for category, raw in enumerate(model.predict_queries(image)):
        classes_seen += 1
        if category >= len(config['class_names']):
            raise ValueError('The model returned too many class predictions.')
        scores, masks, semantic = _probabilities(raw, size, policy['confidence'])
        if task == 'semantic':
            heat = semantic if policy['use_semantic'] else torch.zeros_like(semantic)
            for score, probability in zip(scores, masks):
                heat = torch.maximum(heat, probability * score)
            better = heat > best
            best[better], labels[better] = heat[better], category
            continue
        if category not in things:
            if task == 'panoptic':
                # One native semantic region per stuff class, with predicted presence.
                confidence = float(raw['presence'].float().sigmoid())
                if confidence > policy['confidence']:
                    # _probabilities already includes the presence weighting.
                    candidates.append({
                        'category': category, 'score': confidence,
                        'probability': (semantic / confidence).half().cpu(),
                    })
            continue
        binary = masks > policy['mask_threshold']
        nonempty = binary.flatten(1).any(1) if len(binary) else torch.zeros(0, dtype=torch.bool, device=masks.device)
        scores, masks, binary = scores[nonempty], masks[nonempty], binary[nonempty]
        for index in mask_nms(scores, binary, policy['nms']):
            if task == 'instance':
                instances.append({
                    'category_id': category, 'score': float(scores[index]),
                    'segmentation': encode_mask(binary[index]),
                })
            else:
                candidates.append({
                    'category': category, 'score': float(scores[index]),
                    'probability': masks[index].half().cpu(),
                })
    if classes_seen != len(config['class_names']):
        raise ValueError('The model must predict the full configured vocabulary.')
    if task == 'semantic':
        labels[best < policy['rejection']] = policy['background']
        return labels.cpu().numpy().astype(np.uint8)
    if task == 'instance':
        instances.sort(key=lambda item: -item['score'])
        return instances[:policy['max_dets']]
    return panoptic_merge(candidates, config, size)
