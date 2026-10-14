import argparse
import json
from pathlib import Path

from tag.runtime import SPLIT_SIZES, add_model_arguments, build_model, load_config, save_prediction


def main():
    parser = argparse.ArgumentParser(description="Evaluate frozen SAM3 on a validation split.")
    add_model_arguments(parser)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--annotations",
        type=Path,
        help="Directory containing instances.json and panoptic/.",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N images.")
    args = parser.parse_args()
    if args.task != "semantic" and args.annotations is None:
        parser.error("--annotations is required for instance and panoptic evaluation")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.prompt_batch_size is not None and args.prompt_batch_size < 1:
        parser.error("--prompt-batch-size must be positive")
    if not args.backbone.is_file():
        parser.error(f"Official SAM3 checkpoint not found: {args.backbone}")

    from PIL import Image
    from tqdm import tqdm
    from tag.datasets import instance_image_ids, load_dataset, read_semantic
    from tag.metrics import (
        PanopticMetrics,
        SemanticMetrics,
        evaluate_instances,
        load_panoptic_annotations,
        read_panoptic,
    )
    from tag.readouts import predict

    config = load_config(args.dataset, args.config)
    records = load_dataset(args.data_root, args.dataset)
    split_size = SPLIT_SIZES[args.dataset]
    if args.limit is not None:
        records = records[: args.limit]
    if args.task == "semantic":
        missing = [record.semantic_path for record in records if not record.semantic_path.is_file()]
        if missing:
            raise FileNotFoundError(f"Semantic annotation not found: {missing[0]}")
        metric = SemanticMetrics(config["class_names"])
    elif args.task == "panoptic":
        panoptic_dir = args.annotations / "panoptic"
        metadata = load_panoptic_annotations(panoptic_dir)
        if metadata["class_names"] != config["class_names"]:
            raise ValueError(
                "Panoptic annotation classes do not match the selected dataset."
            )
        metric = PanopticMetrics(config["class_names"], config["thing_indices"])
        for record in records:
            if record.key not in metadata["segments"] or not (panoptic_dir / f"{record.key}.png").is_file():
                raise FileNotFoundError(f"Panoptic ground truth not found: {record.key}")
    else:
        gt_json = args.annotations / "instances.json"
        image_ids = instance_image_ids(gt_json)
        missing = [record.key for record in records if record.key not in image_ids]
        if missing:
            raise ValueError(f"Instance annotation image key not found: {missing[0]}")
        predictions = []

    model = build_model(
        config, args.backbone, args.device, args.prompt_batch_size
    )
    for record in tqdm(records, desc=f"Frozen SAM3 {args.dataset} {args.task}"):
        with Image.open(record.path) as image:
            prediction = predict(model, image.convert("RGB"), config, args.task)
        if args.task == "semantic":
            metric.update(prediction, read_semantic(record, args.dataset))
        elif args.task == "panoptic":
            metric.update(
                *prediction,
                read_panoptic(panoptic_dir, record.key),
                metadata["segments"][record.key],
            )
        else:
            predictions.extend(
                {**item, "image_id": image_ids[record.key]} for item in prediction
            )
        if args.save_predictions:
            save_prediction(
                prediction, args.task, args.output / "predictions" / record.key
            )

    if args.task == "instance":
        result = evaluate_instances(
            predictions,
            gt_json,
            config["class_names"],
            args.dataset,
            config["instance"]["max_dets"],
            [image_ids[record.key] for record in records],
        )
    else:
        result = metric.compute()
    result = {
        "dataset": args.dataset,
        "baseline": "Frozen SAM3",
        "task": args.task,
        "complete_split": len(records) == split_size,
        "expected_images": split_size,
        "checkpoint": args.backbone.name,
        "prompt_batch_size": model.prompt_batch_size,
        "config": config,
        "environment": model.environment,
        **result,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    headline = {
        key: value
        for key, value in result.items()
        if key not in {"per_class", "statistics", "intersection", "union", "config"}
    }
    print(json.dumps(headline, indent=2))


if __name__ == "__main__":
    main()
