"""CPU contract tests; model weights and the SAM3 package are not required.

The wrapper is exercised against a small model exposing SAM3's native grounding
interface. Readout tests use controlled masks and real torch / COCO RLE routines.
Run from the repository root with ``python -m unittest discover -s tests``.
"""

from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
import torch
from torch import nn

from tag.readouts import mask_nms, predict
from tag.sam3_baseline import FrozenSAM3


@dataclass
class FakeFindStage:
    img_ids: torch.Tensor
    text_ids: torch.Tensor
    input_boxes: object
    input_boxes_mask: object
    input_boxes_label: object
    input_points: object
    input_points_mask: object


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.text_calls = []
        self.image_calls = []
        self.text_output = None

    def forward_text(self, prompts, device):
        self.text_calls.append((list(prompts), torch.is_grad_enabled()))
        classes = torch.arange(len(prompts), dtype=torch.float32, device=device)
        features = classes[None, :, None].expand(4, -1, 3).clone()
        features += torch.arange(4, device=device)[:, None, None] * 100
        self.text_output = {
            "language_features": features,
            "language_mask": torch.arange(4, device=device)[None] > classes[:, None],
            "language_embeds": (features + 10).clone(),
        }
        return self.text_output

    def forward_image(self, image):
        self.image_calls.append((image.clone(), torch.is_grad_enabled()))
        return {"vision_features": image.mean((-1, -2)), "vision_pos_enc": [image[:, :, :1, :1]]}


class FakeSAM3(nn.Module):
    supervise_joint_box_scores = False

    def __init__(self):
        super().__init__()
        self.backbone = FakeBackbone()
        self.head = nn.Linear(1, 1)
        self.grounding_calls = []

    def _get_dummy_prompt(self, count):
        return ("dummy", count)

    def forward_grounding(self, *, backbone_out, find_input, find_target, geometric_prompt):
        self.grounding_calls.append({
            "context": backbone_out,
            "input": find_input,
            "target": find_target,
            "geometry": geometric_prompt,
            "grad_enabled": torch.is_grad_enabled(),
        })
        # Class identity is carried by cached text, independently of chunk indices.
        identities = backbone_out["language_features"][0, :, 0].to(torch.float64)
        count = len(identities)
        return {
            "pred_logits": identities[:, None, None].expand(count, 2, 1).clone(),
            "pred_masks": identities[:, None, None, None].expand(count, 2, 3, 4).clone(),
            "presence_logit_dec": (identities + 0.25)[:, None],
            "semantic_seg": (identities + 0.5)[:, None, None, None].expand(count, 1, 3, 4).clone(),
        }


class FrozenSAM3Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.checkpoint = Path(self.temp.name) / "sam3.pt"
        self.checkpoint.touch()
        self.fake = FakeSAM3()
        self.build_calls = []

        def build_model(**kwargs):
            self.build_calls.append(kwargs)
            return self.fake

        sam3 = types.ModuleType("sam3")
        model_package = types.ModuleType("sam3.model")
        builder = types.ModuleType("sam3.model_builder")
        builder.build_sam3_image_model = build_model
        data_misc = types.ModuleType("sam3.model.data_misc")
        data_misc.FindStage = FakeFindStage
        modules = {
            "sam3": sam3,
            "sam3.model": model_package,
            "sam3.model_builder": builder,
            "sam3.model.data_misc": data_misc,
        }
        self.module_patch = patch.dict(sys.modules, modules)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def make_model(self, classes=None, batch_size=2):
        return FrozenSAM3(
            self.checkpoint,
            classes if classes is not None else ["road", "tree", "car", "person", "sky"],
            device="cpu",
            prompt_batch_size=batch_size,
        )

    def test_official_checkpoint_is_loaded_and_every_module_stays_frozen(self):
        model = self.make_model()
        self.assertEqual(len(self.build_calls), 1)
        call = self.build_calls[0]
        self.assertEqual(call["checkpoint_path"], str(self.checkpoint))
        self.assertFalse(call["load_from_HF"])
        self.assertTrue(call["eval_mode"])
        self.assertTrue(call["enable_segmentation"])
        self.assertFalse(call["enable_inst_interactivity"])
        self.assertFalse(any(module.training for module in model.model.modules()))
        self.assertFalse(any(parameter.requires_grad for parameter in model.model.parameters()))
        self.assertEqual(self.fake.backbone.text_calls, [(model.prompts, False)])
        self.assertEqual(model.environment["device"], "cpu")
        self.assertFalse(model.environment["bfloat16"])

    def test_text_is_cached_and_chunks_preserve_class_order_and_native_contract(self):
        model = self.make_model()
        image = Image.new("RGB", (7, 5), (40, 70, 90))
        results = list(model.predict_queries(image))
        self.assertEqual(len(results), 5)
        self.assertEqual(len(self.fake.backbone.image_calls), 1)
        self.assertEqual(len(self.fake.backbone.text_calls), 1)
        self.assertEqual([len(c["input"].text_ids) for c in self.fake.grounding_calls], [2, 2, 1])
        cached = self.fake.backbone.text_output
        start = 0
        for call in self.fake.grounding_calls:
            inputs, context = call["input"], call["context"]
            count = len(inputs.text_ids)
            self.assertIsNone(call["target"])
            self.assertFalse(call["grad_enabled"])
            self.assertEqual(call["geometry"], ("dummy", count))
            self.assertEqual(set(context), {
                "vision_features", "vision_pos_enc", "language_features", "language_mask", "language_embeds",
            })
            torch.testing.assert_close(inputs.img_ids, torch.zeros(count, dtype=torch.long))
            torch.testing.assert_close(inputs.text_ids, torch.arange(count))
            self.assertTrue(all(getattr(inputs, field) is None for field in (
                "input_boxes", "input_boxes_mask", "input_boxes_label", "input_points", "input_points_mask",
            )))
            torch.testing.assert_close(context["language_features"], cached["language_features"][:, start:start + count])
            torch.testing.assert_close(context["language_mask"], cached["language_mask"][start:start + count])
            torch.testing.assert_close(context["language_embeds"], cached["language_embeds"][:, start:start + count])
            start += count
        for category, raw in enumerate(results):
            self.assertEqual(set(raw), {"logits", "masks", "presence", "semantic"})
            self.assertEqual(raw["logits"].shape, (2,))
            self.assertEqual(raw["masks"].shape, (2, 3, 4))
            self.assertEqual(raw["presence"].shape, ())
            self.assertEqual(raw["semantic"].shape, (3, 4))
            self.assertTrue(all(value.dtype == torch.float32 for value in raw.values()))
            self.assertTrue(all(not value.requires_grad for value in raw.values()))
            torch.testing.assert_close(raw["logits"], torch.full((2,), float(category)))
            self.assertAlmostEqual(raw["presence"].item(), category + 0.25)

        # Another view has its own image features, but does not re-encode the text.
        second = list(model.predict_queries(image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), prompt_batch_size=3))
        self.assertEqual(len(second), 5)
        self.assertEqual(len(self.fake.backbone.image_calls), 2)
        self.assertEqual(len(self.fake.backbone.text_calls), 1)
        self.assertEqual([len(c["input"].text_ids) for c in self.fake.grounding_calls], [2, 2, 1, 3, 2])

    def test_rgba_is_converted_to_rgb_resized_and_normalized(self):
        model = self.make_model(classes=["car"])
        list(model.predict_queries(Image.new("RGBA", (4, 6), (0, 128, 255, 17))))
        tensor, grad_enabled = self.fake.backbone.image_calls[0]
        self.assertFalse(grad_enabled)
        self.assertEqual(tensor.shape, (1, 3, 1008, 1008))
        expected = torch.tensor([-1.0, 128 / 255 * 2 - 1, 1.0])
        torch.testing.assert_close(tensor[0, :, 500, 500], expected)
        torch.testing.assert_close(tensor[0], expected[:, None, None].expand(3, 1008, 1008))

    def test_invalid_batch_sizes_and_empty_vocabulary_fail_before_inference(self):
        for batch_size in (0, -1):
            with self.subTest(batch_size=batch_size):
                with self.assertRaises(ValueError):
                    self.make_model(batch_size=batch_size)
        with self.assertRaises(ValueError):
            self.make_model(classes=[])
        self.assertEqual(self.build_calls, [])
        model = self.make_model()
        for batch_size in (0, -1):
            with self.subTest(override=batch_size):
                with self.assertRaises(ValueError):
                    list(model.predict_queries(Image.new("RGB", (2, 2)), prompt_batch_size=batch_size))
        self.assertEqual(self.fake.backbone.image_calls, [])

    def test_missing_checkpoint_and_joint_scores_are_rejected(self):
        with self.assertRaises(FileNotFoundError):
            FrozenSAM3(Path(self.temp.name) / "missing.pt", ["car"], device="cpu")
        self.fake.supervise_joint_box_scores = True
        with self.assertRaises(RuntimeError):
            self.make_model()


def logit(probability):
    return torch.logit(torch.as_tensor(probability, dtype=torch.float32))


def raw_prediction(query_probabilities=(), presence=0.9, masks=None, semantic=0.01):
    queries = torch.as_tensor(query_probabilities, dtype=torch.float32)
    if masks is None:
        masks = torch.full((len(queries), 4, 5), 0.99)
    else:
        masks = torch.as_tensor(masks, dtype=torch.float32)
    semantic_map = torch.full((4, 5), semantic) if np.isscalar(semantic) else torch.as_tensor(semantic)
    return {"logits": logit(queries), "presence": logit(presence), "masks": logit(masks), "semantic": logit(semantic_map)}


class RawModel:
    device = torch.device("cpu")

    def __init__(self, predictions):
        self.predictions = predictions
        self.calls = 0

    def predict_queries(self, image):
        self.calls += 1
        yield from self.predictions


def readout_config(classes=2, things=None):
    return {
        "class_names": [f"class {i}" for i in range(classes)],
        "thing_indices": list(range(classes)) if things is None else things,
        "semantic": {"confidence": 0.01, "use_semantic": False, "rejection": 0.0, "background": 0},
        "instance": {"confidence": 0.01, "mask_threshold": 0.5, "nms": 0.5, "max_dets": 100},
        "panoptic": {"confidence": 0.01, "mask_threshold": 0.5, "nms": 0.5, "overlap": 0.5},
    }


class ReadoutTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (5, 4))

    def test_semantic_instance_and_panoptic_weight_presence_once(self):
        # A second presence multiplication would reverse the winning category.
        raws = [raw_prediction([0.9], presence=0.3), raw_prediction([0.6], presence=0.4)]
        config = readout_config()
        labels = predict(RawModel(raws), self.image, config, "semantic")
        np.testing.assert_array_equal(labels, np.zeros((4, 5), dtype=np.uint8))
        instances = predict(RawModel(raws), self.image, config, "instance")
        self.assertEqual([item["category_id"] for item in instances], [0, 1])
        self.assertAlmostEqual(instances[0]["score"], 0.27, places=6)
        self.assertAlmostEqual(instances[1]["score"], 0.24, places=6)
        panoptic, segments = predict(RawModel(raws), self.image, config, "panoptic")
        np.testing.assert_array_equal(panoptic, np.ones((4, 5), dtype=np.int32))
        self.assertEqual(segments, [{"id": 1, "category_id": 1}])

    def test_native_semantic_maps_and_stuff_presence_are_weighted_once(self):
        raws = [raw_prediction(presence=0.3, semantic=0.9), raw_prediction([0.6], presence=0.4)]
        config = readout_config(things=[1])
        config["semantic"]["use_semantic"] = True
        labels = predict(RawModel(raws), self.image, config, "semantic")
        np.testing.assert_array_equal(labels, np.zeros((4, 5), dtype=np.uint8))
        panoptic, segments = predict(RawModel(raws), self.image, config, "panoptic")
        np.testing.assert_array_equal(panoptic, np.ones((4, 5), dtype=np.int32))
        self.assertEqual(segments, [{"id": 1, "category_id": 1}])

    def test_low_presence_filters_confident_queries(self):
        config = readout_config()
        config["instance"]["confidence"] = 0.3
        raws = [raw_prediction([0.99], presence=0.2), raw_prediction([0.8], presence=0.8)]
        instances = predict(RawModel(raws), self.image, config, "instance")
        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0]["category_id"], 1)
        self.assertAlmostEqual(instances[0]["score"], 0.64, places=6)

    def test_empty_queries_return_background_and_empty_instance_panoptic_outputs(self):
        config = readout_config()
        config["semantic"]["rejection"] = 0.5
        raws = [raw_prediction(), raw_prediction()]
        labels = predict(RawModel(raws), self.image, config, "semantic")
        np.testing.assert_array_equal(labels, np.zeros((4, 5), dtype=np.uint8))
        self.assertEqual(predict(RawModel(raws), self.image, config, "instance"), [])
        panoptic, segments = predict(RawModel(raws), self.image, config, "panoptic")
        np.testing.assert_array_equal(panoptic, np.zeros((4, 5), dtype=np.int32))
        self.assertEqual(segments, [])
        self.assertEqual(mask_nms(torch.empty(0), torch.empty(0, 4, 5), 0.5), [])

    def test_full_vocabulary_is_required_for_every_task(self):
        for task in ("semantic", "instance", "panoptic"):
            for count in (0, 1, 3):
                with self.subTest(task=task, class_predictions=count):
                    with self.assertRaises(ValueError):
                        predict(RawModel([raw_prediction()] * count), self.image, readout_config(), task)

    def test_nms_removes_duplicate_masks_and_rle_preserves_pixels(self):
        left = torch.full((4, 5), 0.01)
        left[:, :2] = 0.99
        right = torch.full((4, 5), 0.01)
        right[:, 3:] = 0.99
        raw = raw_prediction([0.9, 0.8, 0.7], masks=torch.stack([left, left, right]))
        model = RawModel([raw])
        config = readout_config(classes=1)
        instances = predict(model, self.image, config, "instance")
        self.assertEqual(model.calls, 1)
        self.assertEqual(len(instances), 2)
        np.testing.assert_array_equal(mask_utils.decode(instances[0]["segmentation"]), (left.numpy() > 0.5))
        np.testing.assert_array_equal(mask_utils.decode(instances[1]["segmentation"]), (right.numpy() > 0.5))
        config["instance"]["max_dets"] = 1
        limited = predict(RawModel([raw]), self.image, config, "instance")
        self.assertEqual(len(limited), 1)
        self.assertAlmostEqual(limited[0]["score"], 0.81, places=6)

    def test_panoptic_overlap_and_metadata_agree(self):
        left = torch.full((4, 5), 0.01)
        left[:, :3] = 0.99
        right = torch.full((4, 5), 0.01)
        right[:, 2:] = 0.99
        raws = [raw_prediction([0.9], masks=left[None]), raw_prediction([0.8], masks=right[None])]
        config = readout_config()
        panoptic, segments = predict(RawModel(raws), self.image, config, "panoptic")
        self.assertEqual(segments, [{"id": 1, "category_id": 1}, {"id": 2, "category_id": 2}])
        self.assertEqual(set(np.unique(panoptic)), {segment["id"] for segment in segments})
        self.assertEqual(int((panoptic == 1).sum()), 12)
        self.assertEqual(int((panoptic == 2).sum()), 8)
        # The second region retains 8/12 pixels and is rejected at 80% overlap.
        config["panoptic"]["overlap"] = 0.8
        rejected, segments = predict(RawModel(raws), self.image, config, "panoptic")
        self.assertEqual(segments, [{"id": 1, "category_id": 1}])
        self.assertEqual(set(np.unique(rejected)), {0, 1})
        self.assertEqual(int((rejected == 0).sum()), 8)

    def test_unknown_task_is_rejected(self):
        with self.assertRaises(ValueError):
            predict(RawModel([]), self.image, readout_config(), "boxes")


if __name__ == "__main__":
    unittest.main()
