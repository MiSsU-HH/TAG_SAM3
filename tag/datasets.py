"""Validation images and semantic labels in the released dataset layouts."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

CITY_LABEL_IDS = [
    7,
    8,
    11,
    12,
    13,
    17,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    31,
    32,
    33,
]


@dataclass(frozen=True)
class ImageRecord:
    key: str
    path: Path
    semantic_path: Path


def load_dataset(root, dataset, split="val"):
    root = Path(root)
    if dataset == "cityscapes":
        image_dir = root / "leftImg8bit" / split
        images = sorted(image_dir.glob("*/*_leftImg8bit.png"))
        records = []
        for path in images:
            stem = path.name.replace("_leftImg8bit.png", "")
            label = (
                root
                / "gtFine"
                / split
                / path.parent.name
                / f"{stem}_gtFine_labelTrainIds.png"
            )
            if not label.is_file():
                label = label.with_name(f"{stem}_gtFine_labelIds.png")
            records.append(ImageRecord(path.stem, path, label))
    elif dataset == "ade20k":
        split_dir = "validation" if split == "val" else split
        image_dir = root / "images" / split_dir
        records = [
            ImageRecord(
                path.stem, path, root / "annotations" / split_dir / f"{path.stem}.png"
            )
            for path in sorted(image_dir.glob("*.jpg"))
        ]
    elif dataset == "isaid":
        image_dir = root / "img_dir" / split
        records = [
            ImageRecord(
                path.stem,
                path,
                root / "ann_dir" / split / f"{path.stem}_instance_color_RGB.png",
            )
            for path in sorted(image_dir.glob("*.png"))
        ]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    if not records:
        raise FileNotFoundError(f"No validation images found in {image_dir}")
    return records


def read_semantic(record, dataset):
    with Image.open(record.semantic_path) as image:
        labels = np.asarray(image, dtype=np.int64)
    if labels.ndim != 2:
        raise ValueError(
            f"Expected a single-channel label map: {record.semantic_path.name}"
        )
    if dataset == "ade20k":
        labels = np.where((labels > 0) & (labels != 255), labels - 1, 255)
    elif dataset == "cityscapes" and record.semantic_path.name.endswith(
        "_labelIds.png"
    ):
        mapped = np.full_like(labels, 255)
        for index, label_id in enumerate(CITY_LABEL_IDS):
            mapped[labels == label_id] = index
        labels = mapped
    return labels


def instance_image_ids(gt_json):
    import json

    annotation = json.loads(Path(gt_json).read_text())
    result = {}
    for item in annotation["images"]:
        key = Path(item["file_name"]).stem
        if key in result:
            raise ValueError(f"Duplicate image filename stem in {gt_json}: {key}")
        result[key] = item["id"]
    return result
