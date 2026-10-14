"""Convert validation annotations to COCO instances and RGB panoptic-ID maps."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tag.datasets import CITY_LABEL_IDS, load_dataset


ROOT = Path(__file__).resolve().parents[1]


def encode(mask):
    rle = mask_utils.encode(np.asfortranarray(mask, dtype=np.uint8))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def save_panoptic(labels, path):
    """Save exact 24-bit segment IDs in the standard RGB panoptic PNG format."""
    labels = np.asarray(labels)
    if (
        labels.ndim != 2
        or not np.issubdtype(labels.dtype, np.integer)
        or np.any(labels < 0)
        or np.any(labels >= (1 << 24))
    ):
        raise ValueError("Panoptic IDs must be a two-dimensional integer map in [0, 2^24).")
    rgb = np.stack([labels % 256, labels // 256 % 256, labels // 65536], axis=-1)
    Image.fromarray(rgb.astype(np.uint8)).save(path)


def convert_city(record, root, split, things):
    stem = record.key.replace("_leftImg8bit", "")
    path = (
        root
        / "gtFine"
        / split
        / record.path.parent.name
        / f"{stem}_gtFine_instanceIds.png"
    )
    raw = np.asarray(Image.open(path), dtype=np.int64)
    mapping = {label: index for index, label in enumerate(CITY_LABEL_IDS)}
    regions = []
    for value in np.unique(raw):
        category = mapping.get(int(value // 1000 if value >= 1000 else value))
        if category is not None:
            regions.append(
                (raw == value, category, int(category in things and value < 1000))
            )
    return raw.shape, regions


def convert_ade(record, root, split, things, mapping, num_classes):
    split_dir = "validation" if split == "val" else split
    semantic = np.asarray(Image.open(record.semantic_path), dtype=np.int64) - 1
    instance = np.asarray(
        Image.open(root / "annotations_instance" / split_dir / f"{record.key}.png")
    )
    if instance.ndim != 3 or instance.shape[:2] != semantic.shape:
        raise ValueError(f"Invalid ADE20K instance annotation: {record.key}")
    regions = [
        (semantic == category, int(category), 0)
        for category in np.unique(semantic)
        if 0 <= category < num_classes and category not in things
    ]
    for instance_id in np.unique(instance[..., 1]):
        if instance_id == 0:
            continue
        mask = instance[..., 1] == instance_id
        categories = np.unique(instance[..., 0][mask])
        if len(categories) != 1:
            raise ValueError(f"Instance has multiple categories: {record.key}")
        category = mapping[str(categories[0])]
        if category not in things:
            raise ValueError(f"Instance category is not a thing class: {category}")
        regions.append((mask, category, 0))
    return semantic.shape, regions


def write_regions(records, root, output, dataset, split, config):
    things = set(config["thing_indices"])
    mapping = json.loads(
        (ROOT / "configs" / "ade20k_instance_mapping.json").read_text()
    )
    images, annotations, metadata = [], [], {}
    for image_id, record in enumerate(records, 1):
        if dataset == "cityscapes":
            shape, regions = convert_city(record, root, split, things)
        else:
            shape, regions = convert_ade(
                record, root, split, things, mapping, len(config["class_names"])
            )
        height, width = shape
        images.append(
            {
                "id": image_id,
                "file_name": record.path.name,
                "height": height,
                "width": width,
            }
        )
        panoptic = np.zeros(shape, dtype=np.int32)
        segments = []
        for segment_id, (mask, category, crowd) in enumerate(regions, 1):
            panoptic[mask] = segment_id
            segments.append(
                {"id": segment_id, "category_id": category + 1, "iscrowd": crowd}
            )
            if category in things:
                rle = encode(mask)
                annotations.append(
                    {
                        "id": len(annotations) + 1,
                        "image_id": image_id,
                        "category_id": category + 1,
                        "iscrowd": crowd,
                        "segmentation": rle,
                        "area": int(mask.sum()),
                        "bbox": mask_utils.toBbox(rle).tolist(),
                    }
                )
        save_panoptic(panoptic, output / "panoptic" / f"{record.key}.png")
        visible_ids = set(map(int, np.unique(panoptic))) - {0}
        metadata[record.key] = [segment for segment in segments if segment["id"] in visible_ids]
        if image_id % 100 == 0:
            print(f"Converted {image_id}/{len(records)} images", flush=True)
    categories = [
        {"id": index + 1, "name": config["class_names"][index]}
        for index in sorted(things)
    ]
    (output / "instances.json").write_text(
        json.dumps(
            {"images": images, "annotations": annotations, "categories": categories}
        )
    )
    return metadata


def convert_isaid(root, output, split, instance_json, panoptic_json, panoptic_dir):
    annotation_root = root / "annotations"
    instance_json = instance_json or annotation_root / f"iSAID_{split}_instance.json"
    panoptic_json = panoptic_json or annotation_root / f"iSAID_{split}_panoptic.json"
    panoptic_dir = panoptic_dir or annotation_root / f"iSAID_{split}_panoptic"
    instances = json.loads(Path(instance_json).read_text())
    # Only evaluation fields are carried into the generated annotations.
    instances = {key: instances[key] for key in ("images", "annotations", "categories")}
    (output / "instances.json").write_text(json.dumps(instances))
    panoptic = json.loads(Path(panoptic_json).read_text())
    metadata = {}
    for index, annotation in enumerate(panoptic["annotations"], 1):
        name = annotation["file_name"]
        rgb = np.asarray(Image.open(Path(panoptic_dir) / name), dtype=np.int64)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"Official iSAID panoptic PNG must have exactly three RGB channels: {name}")
        ids = rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]
        key = Path(name).stem
        save_panoptic(ids, output / "panoptic" / f"{key}.png")
        metadata[key] = [
            {
                "id": segment["id"],
                "category_id": segment["category_id"] + 1,
                "iscrowd": segment.get("iscrowd", 0),
            }
            for segment in annotation["segments_info"]
        ]
        if index % 100 == 0:
            print(
                f'Converted {index}/{len(panoptic["annotations"])} images', flush=True
            )
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=["ade20k", "cityscapes", "isaid"], required=True
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--instance-json", type=Path)
    parser.add_argument("--panoptic-json", type=Path)
    parser.add_argument("--panoptic-dir", type=Path)
    args = parser.parse_args()
    config = json.loads((ROOT / "configs" / f"sam3_{args.dataset}.json").read_text())
    (args.output / "panoptic").mkdir(parents=True, exist_ok=True)
    if args.dataset == "isaid":
        segments = convert_isaid(
            args.data_root,
            args.output,
            args.split,
            args.instance_json,
            args.panoptic_json,
            args.panoptic_dir,
        )
    else:
        records = load_dataset(args.data_root, args.dataset, args.split)
        segments = write_regions(
            records, args.data_root, args.output, args.dataset, args.split, config
        )
    metadata = {
        "class_names": config["class_names"],
        "num_classes": len(config["class_names"]),
        "thing_trainids": config["thing_indices"],
        "segments": segments,
    }
    (args.output / "panoptic" / "segments.json").write_text(json.dumps(metadata))
    print(f"Wrote annotations for {len(segments)} images to {args.output}")


if __name__ == "__main__":
    main()
