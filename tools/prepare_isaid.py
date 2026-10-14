"""Create the iSAID validation image and semantic-label tiles."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


COLORS = np.array(
    [
        [0, 0, 0],
        [0, 0, 63],
        [0, 63, 63],
        [0, 63, 0],
        [0, 63, 127],
        [0, 63, 191],
        [0, 63, 255],
        [0, 127, 63],
        [0, 127, 127],
        [0, 0, 127],
        [0, 0, 191],
        [0, 0, 255],
        [0, 191, 127],
        [0, 127, 191],
        [0, 127, 255],
        [0, 100, 155],
    ],
    dtype=np.uint8,
)


def windows(length):
    length = max(length, 896)
    return sorted(
        {max(0, min(start + 896, length) - 896) for start in range(0, length, 512)}
    )


def semantic_labels(rgb):
    labels = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for category, color in enumerate(COLORS):
        labels[np.all(rgb == color, axis=-1)] = category
    return labels


def prepare(images, semantic, output):
    image_dir = output / "img_dir/val"
    label_dir = output / "ann_dir/val"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for label_path in sorted(semantic.glob("*_instance_color_RGB.png")):
        stem = label_path.name.removesuffix("_instance_color_RGB.png")
        rgb = np.asarray(Image.open(images / f"{stem}.png").convert("RGB"))
        labels = semantic_labels(np.asarray(Image.open(label_path).convert("RGB")))
        height, width = labels.shape
        if rgb.shape[:2] != (height, width):
            raise ValueError(f"Image and semantic mask dimensions differ: {stem}")
        padding = ((0, max(0, 896 - height)), (0, max(0, 896 - width)))
        rgb = np.pad(rgb, (*padding, (0, 0)), constant_values=0)
        labels = np.pad(labels, padding, constant_values=255)
        for y in windows(height):
            for x in windows(width):
                name = f"{stem}_{y}_{y + 896}_{x}_{x + 896}"
                Image.fromarray(rgb[y : y + 896, x : x + 896]).save(
                    image_dir / f"{name}.png"
                )
                Image.fromarray(labels[y : y + 896, x : x + 896]).save(
                    label_dir / f"{name}_instance_color_RGB.png"
                )
                total += 1
    if not total:
        raise ValueError("No validation semantic masks were found.")
    print(f"Prepared {total} validation tiles in {output}")


def polygon_mask(annotation):
    from pycocotools import mask as mask_utils

    rings = [
        np.asarray(ring, dtype=np.float64)
        for ring in annotation["segmentation"]
        if len(ring) >= 6
    ]
    if not rings:
        return None
    xs = np.concatenate([ring[0::2] for ring in rings])
    ys = np.concatenate([ring[1::2] for ring in rings])
    x, y = max(0, int(np.floor(xs.min()))), max(0, int(np.floor(ys.min())))
    width, height = int(np.ceil(xs.max())) - x + 2, int(np.ceil(ys.max())) - y + 2
    for ring in rings:
        ring[0::2] -= x
        ring[1::2] -= y
    rle = mask_utils.merge(
        mask_utils.frPyObjects([ring.tolist() for ring in rings], height, width)
    )
    return x, y, mask_utils.decode(rle).astype(bool)


def prepare_annotations(output, instances):
    from pycocotools import mask as mask_utils

    names = [
        "ship",
        "storage_tank",
        "baseball_diamond",
        "tennis_court",
        "basketball_court",
        "Ground_Track_Field",
        "Bridge",
        "Large_Vehicle",
        "Small_Vehicle",
        "Helicopter",
        "Swimming_pool",
        "Roundabout",
        "Soccer_ball_field",
        "plane",
        "Harbor",
    ]
    source = json.loads(instances.read_text())
    category_ids = {
        item["id"]: names.index(item["name"]) + 1 for item in source["categories"]
    }
    by_image = defaultdict(list)
    for annotation in source["annotations"]:
        by_image[annotation["image_id"]].append(annotation)
    destination = output / "annotations"
    panoptic_dir = destination / "iSAID_val_panoptic"
    panoptic_dir.mkdir(parents=True, exist_ok=True)
    images, annotations, panoptic_annotations = [], [], []
    for image in sorted(source["images"], key=lambda item: item["file_name"]):
        stem = Path(image["file_name"]).stem
        rasters = [
            (annotation, polygon_mask(annotation))
            for annotation in sorted(
                by_image[image["id"]], key=lambda item: item["area"], reverse=True
            )
        ]
        tiles = []
        for path in (output / "ann_dir/val").glob(f"{stem}_*_instance_color_RGB.png"):
            name = path.name.removesuffix("_instance_color_RGB.png")
            y0, y1, x0, x1 = map(int, name.removeprefix(stem + "_").split("_"))
            tiles.append((y0, y1, x0, x1, name, path))
        for y0, y1, x0, x1, name, path in sorted(tiles):
            labels = np.asarray(Image.open(path))
            height, width = labels.shape
            image_id = len(images) + 1
            images.append(
                {
                    "id": image_id,
                    "file_name": name + ".png",
                    "width": width,
                    "height": height,
                }
            )
            panoptic = np.zeros((height, width), dtype=np.uint32)
            panoptic[labels != 255] = 1
            segments = [{"id": 1, "category_id": 0, "iscrowd": 0}]
            for annotation, raster in rasters:
                if raster is None:
                    continue
                x, y, mask = raster
                mh, mw = mask.shape
                top, bottom = max(y, y0), min(y + mh, y1, image["height"])
                left, right = max(x, x0), min(x + mw, x1, image["width"])
                if top >= bottom or left >= right:
                    continue
                crop = mask[top - y : bottom - y, left - x : right - x]
                if not crop.any():
                    continue
                tile = np.zeros((height, width), dtype=np.uint8)
                tile[top - y0 : bottom - y0, left - x0 : right - x0] = crop
                category = category_ids[annotation["category_id"]]
                crowd = int(annotation.get("iscrowd", 0))
                rle = mask_utils.encode(np.asfortranarray(tile))
                rle["counts"] = rle["counts"].decode("ascii")
                annotations.append(
                    {
                        "id": len(annotations) + 1,
                        "image_id": image_id,
                        "category_id": category,
                        "iscrowd": crowd,
                        "segmentation": rle,
                        "bbox": mask_utils.toBbox(rle).tolist(),
                        "area": int(mask_utils.area(rle)),
                    }
                )
                visible = tile.astype(bool)
                if visible.any():
                    segment_id = len(segments) + 1
                    panoptic[visible] = segment_id
                    segments.append(
                        {"id": segment_id, "category_id": category, "iscrowd": crowd}
                    )
            visible_segments = []
            for segment in segments:
                yy, xx = np.nonzero(panoptic == segment["id"])
                if not len(xx):
                    continue
                segment["area"] = len(xx)
                segment["bbox"] = [
                    int(xx.min()),
                    int(yy.min()),
                    int(xx.max() - xx.min() + 1),
                    int(yy.max() - yy.min() + 1),
                ]
                visible_segments.append(segment)
            rgb = np.stack(
                [panoptic % 256, panoptic // 256 % 256, panoptic // 65536], -1
            )
            Image.fromarray(rgb.astype(np.uint8)).save(panoptic_dir / f"{name}.png")
            panoptic_annotations.append(
                {
                    "image_id": image_id,
                    "file_name": name + ".png",
                    "segments_info": visible_segments,
                }
            )
    categories = [
        {"id": index + 1, "name": name, "isthing": 1}
        for index, name in enumerate(names)
    ]
    (destination / "iSAID_val_instance.json").write_text(
        json.dumps(
            {"images": images, "annotations": annotations, "categories": categories}
        )
    )
    panoptic_categories = [{"id": 0, "name": "background", "isthing": 0}] + categories
    (destination / "iSAID_val_panoptic.json").write_text(
        json.dumps(
            {
                "images": images,
                "annotations": panoptic_annotations,
                "categories": panoptic_categories,
            }
        )
    )
    print(f"Prepared {len(annotations)} instance annotations for {len(images)} tiles")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Directory containing original validation images.",
    )
    parser.add_argument(
        "--semantic",
        type=Path,
        required=True,
        help="Directory containing original validation semantic RGB masks.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--instances",
        type=Path,
        help="Official original-resolution validation COCO polygon JSON.",
    )
    args = parser.parse_args()
    prepare(args.images, args.semantic, args.output)
    if args.instances:
        prepare_annotations(args.output, args.instances)


if __name__ == "__main__":
    main()
