"""Dataset-level mIoU, COCO mask AP, and panoptic quality."""

import json
from pathlib import Path

import numpy as np
from PIL import Image


class SemanticMetrics:
    def __init__(self, class_names):
        self.class_names = class_names
        self.intersection = np.zeros(len(class_names), dtype=np.int64)
        self.predicted = np.zeros_like(self.intersection)
        self.target = np.zeros_like(self.intersection)
        self.images = 0

    def update(self, prediction, target):
        prediction, target = np.asarray(prediction), np.asarray(target)
        if prediction.shape != target.shape:
            raise ValueError("Prediction and ground truth must have the same shape.")
        if prediction.ndim != 2 or not all(
            np.issubdtype(array.dtype, np.integer) for array in (prediction, target)
        ):
            raise ValueError("Semantic maps must be two-dimensional integer arrays.")
        classes = len(self.class_names)
        valid = (target >= 0) & (target < classes)
        predicted, target = prediction[valid], target[valid]
        self.intersection += np.bincount(target[predicted == target], minlength=classes)
        self.predicted += np.bincount(
            predicted[(predicted >= 0) & (predicted < classes)], minlength=classes
        )
        self.target += np.bincount(target, minlength=classes)
        self.images += 1

    def compute(self):
        union = self.predicted + self.target - self.intersection
        iou = np.divide(
            self.intersection,
            union,
            out=np.full_like(union, np.nan, dtype=float),
            where=union > 0,
        )
        return {
            "images": self.images,
            "mIoU": float(np.nanmean(iou) * 100) if np.isfinite(iou).any() else None,
            "per_class": {
                name: float(value * 100) if np.isfinite(value) else None
                for name, value in zip(self.class_names, iou)
            },
            "intersection": self.intersection.tolist(),
            "union": union.tolist(),
        }


class PanopticMetrics:
    def __init__(self, class_names, thing_indices):
        self.class_names = class_names
        self.things = {index + 1 for index in thing_indices}
        self.stuff = set(range(1, len(class_names) + 1)) - self.things
        self.stats = np.zeros((len(class_names) + 1, 4), dtype=np.float64)
        self.images = 0

    def update(self, prediction, segments, target, target_segments):
        prediction, target = np.asarray(prediction), np.asarray(target)
        if prediction.shape != target.shape:
            raise ValueError("Prediction and ground truth must have the same shape.")
        for array in (prediction, target):
            if (
                array.ndim != 2
                or not np.issubdtype(array.dtype, np.integer)
                or np.any(array < 0)
                or np.any(array >= (1 << 32))
            ):
                raise ValueError("Panoptic maps must contain nonnegative 32-bit integer IDs.")
        target = target.astype(np.int64, copy=False)
        prediction = prediction.astype(np.int64, copy=False)
        gt_ids, gt_counts = np.unique(target, return_counts=True)
        pred_ids, pred_counts = np.unique(prediction, return_counts=True)
        gt = self._validate_segments(target_segments, gt_ids, "ground truth")
        pred = self._validate_segments(segments, pred_ids, "prediction")
        gt_area, pred_area = dict(zip(gt_ids, gt_counts)), dict(
            zip(pred_ids, pred_counts)
        )
        offset = 1 << 32
        pairs, counts = np.unique(
            target.astype(np.uint64) * np.uint64(offset) + prediction.astype(np.uint64),
            return_counts=True,
        )
        # Python integer arithmetic avoids uint64 scalar promotion to float in NumPy 1.x.
        intersection = {
            (int(pair) // offset, int(pair) % offset): int(count)
            for pair, count in zip(pairs, counts)
        }
        matched_gt, matched_pred = set(), set()
        for (gid, pid), area in intersection.items():
            if gid not in gt or pid not in pred:
                continue
            category = gt[gid]["category_id"]
            if gt[gid].get("iscrowd", 0) or category != pred[pid]["category_id"]:
                continue
            union = gt_area[gid] + pred_area[pid] - area - intersection.get((0, pid), 0)
            iou = area / union if union else 0
            if iou > 0.5:
                self.stats[category, :2] += [iou, 1]
                matched_gt.add(gid)
                matched_pred.add(pid)
        crowds = {}
        for gid, segment in gt.items():
            if segment.get("iscrowd", 0):
                crowds.setdefault(segment["category_id"], []).append(gid)
            elif gid not in matched_gt:
                self.stats[segment["category_id"], 3] += 1
        for pid, segment in pred.items():
            if pid in matched_pred:
                continue
            category = segment["category_id"]
            ignored = intersection.get((0, pid), 0)
            if category in crowds:
                ignored += sum(intersection.get((gid, pid), 0) for gid in crowds[category])
            if pred_area.get(pid, 0) and ignored / pred_area[pid] > 0.5:
                continue
            self.stats[category, 2] += 1
        self.images += 1

    def _validate_segments(self, segments, map_ids, source):
        indexed = {}
        for segment in segments:
            segment_id, category = segment["id"], segment["category_id"]
            if (
                not isinstance(segment_id, (int, np.integer))
                or not 0 < segment_id < (1 << 32)
                or segment_id in indexed
                or not isinstance(category, (int, np.integer))
                or not 1 <= category <= len(self.class_names)
            ):
                raise ValueError(f"Invalid or duplicate {source} panoptic segment: {segment}")
            indexed[segment_id] = segment
        if set(map(int, map_ids)) - {0} != set(indexed):
            raise ValueError(f"Panoptic {source} map IDs do not match segment metadata.")
        return indexed

    def _summarize(self, categories):
        stats = self.stats[sorted(categories)]
        stats = stats[stats[:, 1:].sum(1) > 0]
        if not len(stats):
            return {"PQ": 0.0, "SQ": 0.0, "RQ": 0.0, "classes": 0}
        iou, tp, fp, fn = stats.T
        denominator = tp + 0.5 * (fp + fn)
        sq = np.divide(iou, tp, out=np.zeros_like(iou), where=tp > 0)
        return {
            "PQ": float(np.mean(iou / denominator) * 100),
            "SQ": float(np.mean(sq) * 100),
            "RQ": float(np.mean(tp / denominator) * 100),
            "classes": len(stats),
        }

    def compute(self):
        return {
            "images": self.images,
            "All": self._summarize(self.things | self.stuff),
            "Things": self._summarize(self.things),
            "Stuff": self._summarize(self.stuff),
            "per_class": {
                name: self._summarize({index + 1})
                for index, name in enumerate(self.class_names)
            },
            "statistics": self.stats.tolist(),
        }


def load_panoptic_annotations(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "segments.json").read_text())
    return metadata


def read_panoptic(directory, key):
    with Image.open(Path(directory) / f"{key}.png") as image:
        labels = np.asarray(image, dtype=np.int64)
    if labels.ndim == 3 and labels.shape[-1] == 3:
        return labels[..., 0] + 256 * labels[..., 1] + 65536 * labels[..., 2]
    if labels.ndim != 2:
        raise ValueError("Panoptic PNG must be an RGB segment-ID map or a single-channel integer map.")
    if np.any(labels < 0):
        raise ValueError("Panoptic segment IDs must be nonnegative.")
    return labels


def instance_category_mapping(annotation, class_names, dataset):
    if dataset == "isaid":
        mapping = {category["id"]: category["id"] for category in annotation["categories"]}
        if any(not 0 < category < len(class_names) for category in mapping):
            raise ValueError("iSAID instance category IDs must match foreground class indices.")
        return mapping
    by_name = {name: index for index, name in enumerate(class_names)}
    unknown = [category["name"] for category in annotation["categories"] if category["name"] not in by_name]
    if unknown:
        raise ValueError(f"Unknown instance categories for the selected dataset: {unknown}")
    return {
        by_name[category["name"]]: category["id"]
        for category in annotation["categories"]
    }


def evaluate_instances(
    predictions, gt_json, class_names, dataset, max_dets, image_ids=None
):
    """Predictions use model class indices and COCO image IDs."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if max_dets < 10:
        raise ValueError("COCO max_dets must be at least 10.")
    ground_truth = COCO(str(gt_json))
    mapping = instance_category_mapping(ground_truth.dataset, class_names, dataset)
    converted = [
        {**item, "category_id": mapping[item["category_id"]]} for item in predictions
    ]
    if converted:
        detections = ground_truth.loadRes(converted)
    else:
        detections = COCO()
        detections.dataset = {
            "images": ground_truth.dataset["images"],
            "categories": ground_truth.dataset["categories"],
            "annotations": [],
        }
        detections.createIndex()
    evaluator = COCOeval(ground_truth, detections, iouType="segm")
    evaluator.params.maxDets = [1, 10, max_dets]
    evaluator.params.imgIds = sorted(
        image_ids if image_ids is not None else ground_truth.getImgIds()
    )
    evaluator.evaluate()
    evaluator.accumulate()
    precision = evaluator.eval["precision"]
    scores = {}
    for name, threshold, area in [
        ("AP", None, 0),
        ("AP50", 0.5, 0),
        ("AP75", 0.75, 0),
        ("APs", None, 1),
        ("APm", None, 2),
        ("APl", None, 3),
    ]:
        values = precision[:, :, :, area, -1]
        if threshold is not None:
            values = values[np.isclose(evaluator.params.iouThrs, threshold)]
        values = values[values > -1]
        scores[name] = float(values.mean() * 100) if values.size else None
    return {"images": len(evaluator.params.imgIds), "max_dets": max_dets, **scores}
