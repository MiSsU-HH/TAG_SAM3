"""Configuration and file I/O for the frozen SAM3 evaluation release."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("ade20k", "cityscapes", "isaid")
TASKS = ("semantic", "instance", "panoptic")
SPLIT_SIZES = {"ade20k": 2000, "cityscapes": 500, "isaid": 11644}


def load_config(dataset, path=None):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    path = Path(path) if path is not None else ROOT / "configs" / f"sam3_{dataset}.json"
    config = json.loads(path.read_text())
    if config.get("dataset") != dataset or config.get("baseline") != "sam3":
        raise ValueError("Use a frozen SAM3 configuration for the selected dataset.")
    names = config["class_names"]
    if not names or len(names) > 255 or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Provide 1 to 255 nonempty class names.")
    if len(names) != len(set(names)):
        raise ValueError("Class names must be unique.")
    things = config["thing_indices"]
    if len(things) != len(set(things)) or any(
        not isinstance(index, int) or not 0 <= index < len(names) for index in things
    ):
        raise ValueError("Thing indices must be unique valid class indices.")
    if not isinstance(config["prompt_batch_size"], int) or config["prompt_batch_size"] < 1:
        raise ValueError("Prompt batch size must be a positive integer.")
    for task in TASKS:
        settings = config[task]
        keys = {
            "semantic": ("confidence", "rejection"),
            "instance": ("confidence", "mask_threshold", "nms"),
            "panoptic": ("confidence", "mask_threshold", "nms", "overlap"),
        }[task]
        for key in keys:
            if not isinstance(settings[key], (int, float)) or not 0 <= settings[key] <= 1:
                raise ValueError(f"{task}.{key} must be in [0, 1].")
    if not isinstance(config["semantic"]["use_semantic"], bool):
        raise ValueError("semantic.use_semantic must be a boolean.")
    if not isinstance(config["semantic"]["background"], int) or not 0 <= config["semantic"]["background"] < len(names):
        raise ValueError("semantic.background must be a valid class index.")
    if not isinstance(config["instance"]["max_dets"], int) or config["instance"]["max_dets"] < 10:
        raise ValueError("instance.max_dets must be an integer of at least 10 for COCO evaluation.")
    return config


def build_model(config, backbone, device, prompt_batch_size):
    from .sam3_baseline import FrozenSAM3

    return FrozenSAM3(
        backbone_path=backbone,
        class_names=config["class_names"],
        device=device,
        prompt_batch_size=prompt_batch_size or config["prompt_batch_size"],
    )


def add_model_arguments(parser):
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--task", choices=TASKS, default="semantic")
    parser.add_argument("--config", type=Path, help="Optional frozen SAM3 readout preset.")
    parser.add_argument("--backbone", type=Path, default=ROOT / "weights" / "sam3.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--prompt-batch-size",
        type=int,
        help="Positive prompt chunk size.",
    )


def save_prediction(prediction, task, output):
    import numpy as np
    from PIL import Image

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if task == "semantic":
        Image.fromarray(prediction.astype(np.uint8)).save(output.with_suffix(".png"))
    elif task == "instance":
        output.with_suffix(".json").write_text(json.dumps(prediction) + "\n")
    else:
        labels, segments = prediction
        rgb = np.stack([labels % 256, labels // 256 % 256, labels // 65536], axis=-1)
        Image.fromarray(rgb.astype(np.uint8)).save(output.with_suffix(".png"))
        output.with_suffix(".json").write_text(json.dumps(segments) + "\n")
