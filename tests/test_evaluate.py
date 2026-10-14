"""Exercise all evaluation tasks on one tiny validation image without SAM3 weights."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import evaluate
from tag.runtime import load_config


ROOT = Path(__file__).resolve().parents[1]
HAS_INFERENCE_DEPS = all(importlib.util.find_spec(name) for name in ("torch", "pycocotools", "tqdm"))


class EvaluationCliTests(unittest.TestCase):
    def test_invalid_requests_fail_before_importing_torch(self):
        # A fresh process makes the import-order check independent of other tests.
        script = """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise AssertionError('torch was imported before CLI validation')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import evaluate
evaluate.main()
"""
        cases = [
            (["--variant", "tag_i"], "unrecognized arguments"),
            (["--checkpoint", "private.safetensors"], "unrecognized arguments"),
            (["--limit", "0"], "--limit must be positive"),
            (["--limit", "-1"], "--limit must be positive"),
            (["--prompt-batch-size", "0"], "--prompt-batch-size must be positive"),
            (["--prompt-batch-size", "-1"], "--prompt-batch-size must be positive"),
            (["--task", "instance"], "--annotations is required"),
            (["--task", "panoptic"], "--annotations is required"),
        ]
        for arguments, message in cases:
            with self.subTest(arguments=arguments):
                completed = subprocess.run(
                    [sys.executable, "-c", script, "--dataset", "cityscapes", "--data-root", "unused", *arguments],
                    cwd=ROOT, text=True, capture_output=True, timeout=30,
                )
                self.assertEqual(completed.returncode, 2, completed.stderr)
                self.assertIn(message, completed.stderr)
                self.assertNotIn("Traceback", completed.stderr)


@unittest.skipUnless(HAS_INFERENCE_DEPS, "torch, pycocotools and tqdm are needed for readout integration")
class EvaluationIntegrationTests(unittest.TestCase):
    def test_three_tasks_save_predictions_and_reproducibility_metadata(self):
        from pycocotools import mask as mask_utils
        import torch

        config = load_config("cityscapes")
        semantic = np.array([[0, 11], [0, 11]], dtype=np.uint8)
        key = "tiny_leftImg8bit"

        class FakeFrozenSAM3:
            def __init__(self, prompt_batch_size):
                self.prompt_batch_size = prompt_batch_size
                self.device = torch.device("cpu")
                self.environment = {"torch": torch.__version__, "sam3": "synthetic-test", "device": "cpu", "bfloat16": False}

            def predict_queries(self, image):
                if image.size != (2, 2):
                    raise AssertionError("Unexpected test image dimensions")
                for category in range(len(config["class_names"])):
                    present = category in (0, 11)
                    mask = torch.where(torch.from_numpy(semantic == category), 20.0, -20.0)
                    yield {
                        "presence": torch.tensor(20.0 if present else -20.0),
                        "logits": torch.tensor([20.0 if present else -20.0]),
                        "masks": mask[None], "semantic": mask,
                    }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "data/leftImg8bit/val/testcity"
            label_dir = root / "data/gtFine/val/testcity"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(image_dir / f"{key}.png")
            Image.fromarray(semantic).save(label_dir / "tiny_gtFine_labelTrainIds.png")
            backbone = root / "sam3.pt"
            backbone.write_bytes(b"synthetic checkpoint placeholder; model construction is mocked")
            annotations = root / "annotations"
            (annotations / "panoptic").mkdir(parents=True)
            panoptic = np.array([[1, 2], [1, 2]], dtype=np.int64)
            segments = [{"id": 1, "category_id": 1}, {"id": 2, "category_id": 12}]
            panoptic_rgb = np.stack([panoptic % 256, panoptic // 256 % 256, panoptic // 65536], axis=-1)
            Image.fromarray(panoptic_rgb.astype(np.uint8)).save(annotations / "panoptic" / f"{key}.png")
            (annotations / "panoptic/segments.json").write_text(json.dumps({
                "class_names": config["class_names"], "segments": {key: segments},
            }))
            rle = mask_utils.encode(np.asfortranarray(semantic == 11, dtype=np.uint8))
            rle["counts"] = rle["counts"].decode("ascii")
            (annotations / "instances.json").write_text(json.dumps({
                "info": {},
                "images": [{"id": 7, "file_name": f"{key}.png", "height": 2, "width": 2}],
                "categories": [{"id": 12, "name": "person"}],
                "annotations": [{"id": 1, "image_id": 7, "category_id": 12, "iscrowd": 0,
                                 "segmentation": rle, "area": 2, "bbox": [1, 0, 1, 2]}],
            }))

            for task in ("semantic", "instance", "panoptic"):
                with self.subTest(task=task):
                    output = root / task
                    arguments = [
                        "evaluate.py", "--dataset", "cityscapes", "--task", task,
                        "--data-root", str(root / "data"), "--annotations", str(annotations),
                        "--backbone", str(backbone), "--device", "cpu", "--prompt-batch-size", "7",
                        "--limit", "1", "--output", str(output), "--save-predictions",
                    ]
                    model = FakeFrozenSAM3(7)
                    with patch.object(evaluate, "build_model", return_value=model) as build:
                        with patch.object(sys, "argv", arguments), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                            evaluate.main()
                    build.assert_called_once_with(config, backbone, "cpu", 7)

                    def reject_nonfinite(value):
                        raise AssertionError(f"Nonfinite value in metrics JSON: {value}")

                    result = json.loads((output / "metrics.json").read_text(), parse_constant=reject_nonfinite)
                    self.assertEqual(result["baseline"], "Frozen SAM3")
                    self.assertEqual(result["dataset"], "cityscapes")
                    self.assertEqual(result["task"], task)
                    self.assertEqual(result["images"], 1)
                    self.assertFalse(result["complete_split"])
                    self.assertEqual(result["expected_images"], 500)
                    self.assertEqual(result["checkpoint"], "sam3.pt")
                    self.assertEqual(result["prompt_batch_size"], 7)
                    self.assertEqual(result["config"], config)
                    self.assertEqual(result["environment"], model.environment)
                    prediction = output / "predictions" / key
                    if task == "semantic":
                        self.assertEqual(result["mIoU"], 100.0)
                        self.assertIsNone(result["per_class"]["sky"])
                        with Image.open(prediction.with_suffix(".png")) as image:
                            self.assertEqual(image.mode, "L")
                            np.testing.assert_array_equal(np.asarray(image), semantic)
                    elif task == "instance":
                        self.assertAlmostEqual(result["AP"], 100.0)
                        self.assertIsNone(result["APm"])
                        saved = json.loads(prediction.with_suffix(".json").read_text())
                        self.assertEqual(len(saved), 1)
                        self.assertEqual(saved[0]["category_id"], 11)
                        self.assertIsInstance(saved[0]["segmentation"]["counts"], str)
                        np.testing.assert_array_equal(mask_utils.decode(saved[0]["segmentation"]), semantic == 11)
                    else:
                        self.assertEqual(result["All"]["PQ"], 100.0)
                        saved = json.loads(prediction.with_suffix(".json").read_text())
                        self.assertEqual(saved, segments)
                        with Image.open(prediction.with_suffix(".png")) as image:
                            self.assertEqual(image.mode, "RGB")
                            rgb = np.asarray(image, dtype=np.int64)
                        np.testing.assert_array_equal(rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2], panoptic)


if __name__ == "__main__":
    unittest.main()
