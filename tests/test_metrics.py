"""Small, deterministic checks for dataset labels and evaluation protocols."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from tag.datasets import ImageRecord, instance_image_ids, read_semantic
from tag.metrics import PanopticMetrics, SemanticMetrics, evaluate_instances, instance_category_mapping, read_panoptic
from tools.prepare_isaid import prepare, semantic_labels, windows


HAS_COCO = importlib.util.find_spec("pycocotools") is not None


class SemanticTests(unittest.TestCase):
    def test_ignore_pixels_do_not_contribute_to_union(self):
        metric = SemanticMetrics(["a", "b"])
        metric.update(np.array([[0, 0, 1]]), np.array([[0, 1, 255]]))
        result = metric.compute()
        self.assertEqual(result["intersection"], [1, 0])
        self.assertEqual(result["union"], [2, 1])
        self.assertEqual(result["mIoU"], 25.0)

    def test_void_predictions_are_false_negatives(self):
        metric = SemanticMetrics(["a"])
        metric.update(np.array([[-1, 0]]), np.array([[0, 0]]))
        self.assertEqual(metric.compute()["mIoU"], 50.0)

    def test_all_ignored_is_json_serializable(self):
        metric = SemanticMetrics(["a"])
        metric.update(np.array([[0]]), np.array([[255]]))
        result = metric.compute()
        self.assertIsNone(result["mIoU"])
        self.assertIsNone(result["per_class"]["a"])
        json.dumps(result, allow_nan=False)


class PanopticTests(unittest.TestCase):
    def test_perfect_thing_and_stuff(self):
        metric = PanopticMetrics(["stuff", "thing"], [1])
        segments = [{"id": 1, "category_id": 1}, {"id": 2, "category_id": 2}]
        metric.update(np.array([[1, 2]]), segments, np.array([[1, 2]]), segments)
        for group in ("All", "Things", "Stuff"):
            self.assertEqual(metric.compute()[group]["PQ"], 100.0)

    def test_unmatched_segment_is_false_negative(self):
        metric = PanopticMetrics(["thing"], [0])
        metric.update(np.array([[0]]), [], np.array([[1]]), [{"id": 1, "category_id": 1}])
        self.assertEqual(metric.compute()["statistics"][1], [0.0, 0.0, 0.0, 1.0])

    def test_void_area_is_excluded_from_match_union(self):
        metric = PanopticMetrics(["thing"], [0])
        metric.update(
            np.array([[1, 1, 1]]), [{"id": 1, "category_id": 1}],
            np.array([[2, 0, 0]]), [{"id": 2, "category_id": 1}],
        )
        self.assertEqual(metric.compute()["All"]["PQ"], 100.0)

    def test_multiple_crowds_of_same_category_are_ignored(self):
        metric = PanopticMetrics(["thing"], [0])
        metric.update(
            np.array([[1, 1, 1, 0, 0]]), [{"id": 1, "category_id": 1}],
            np.array([[2, 3, 4, 4, 4]]),
            [{"id": 2, "category_id": 1, "iscrowd": 1},
             {"id": 3, "category_id": 1, "iscrowd": 1},
             {"id": 4, "category_id": 1}],
        )
        self.assertEqual(metric.compute()["statistics"][1], [0.0, 0.0, 0.0, 1.0])

    def test_metadata_must_match_map(self):
        metric = PanopticMetrics(["thing"], [0])
        with self.assertRaisesRegex(ValueError, "map IDs"):
            metric.update(np.array([[1]]), [], np.array([[0]]), [])


class DatasetTests(unittest.TestCase):
    def test_ade_and_cityscapes_label_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ade.png"
            Image.fromarray(np.array([[0, 1, 150, 255]], dtype=np.uint8)).save(path)
            record = ImageRecord("sample", path, path)
            np.testing.assert_array_equal(read_semantic(record, "ade20k"), [[255, 0, 149, 255]])
            path = Path(directory) / "sample_gtFine_labelIds.png"
            Image.fromarray(np.array([[0, 7, 8, 33]], dtype=np.uint8)).save(path)
            record = ImageRecord("sample", path, path)
            np.testing.assert_array_equal(read_semantic(record, "cityscapes"), [[255, 0, 1, 18]])

    def test_ambiguous_coco_image_stems_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "instances.json"
            path.write_text(json.dumps({"images": [
                {"id": 1, "file_name": "a/image.png"},
                {"id": 2, "file_name": "b/image.png"},
            ]}))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                instance_image_ids(path)

    def test_panoptic_reader_keeps_legacy_single_channel_maps(self):
        with tempfile.TemporaryDirectory() as directory:
            labels = np.array([[0, 42, 65535]], dtype=np.uint16)
            Image.fromarray(labels).save(Path(directory) / "sample.png")
            np.testing.assert_array_equal(read_panoptic(directory, "sample"), labels)

    def test_panoptic_reader_rejects_rgba_maps(self):
        with tempfile.TemporaryDirectory() as directory:
            Image.new("RGBA", (2, 2)).save(Path(directory) / "sample.png")
            with self.assertRaisesRegex(ValueError, "RGB segment-ID map"):
                read_panoptic(directory, "sample")

    def test_iSAID_palette_and_small_image_padding(self):
        colors = np.array([[[0, 0, 0], [0, 0, 63], [1, 2, 3]]], dtype=np.uint8)
        np.testing.assert_array_equal(semantic_labels(colors), [[0, 1, 255]])
        self.assertEqual(windows(32), [0])
        self.assertEqual(windows(1600), [0, 512, 704])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "labels").mkdir()
            Image.fromarray(np.zeros((1, 3, 3), dtype=np.uint8)).save(root / "images/sample.png")
            Image.fromarray(colors).save(root / "labels/sample_instance_color_RGB.png")
            prepare(root / "images", root / "labels", root / "output")
            path = root / "output/ann_dir/val/sample_0_896_0_896_instance_color_RGB.png"
            with Image.open(path) as image:
                labels = np.asarray(image)
            self.assertEqual(labels.shape, (896, 896))
            np.testing.assert_array_equal(labels[0, :3], [0, 1, 255])
            self.assertEqual(int(labels[-1, -1]), 255)


class InstanceCategoryTests(unittest.TestCase):
    def test_coco_categories_are_mapped_by_name(self):
        annotation = {"categories": [{"id": 18, "name": "thing"}]}
        self.assertEqual(instance_category_mapping(annotation, ["stuff", "thing"], "ade20k"), {1: 18})

    def test_isaid_keeps_foreground_ids(self):
        annotation = {"categories": [{"id": 1, "name": "ship"}]}
        self.assertEqual(instance_category_mapping(annotation, ["background", "ship"], "isaid"), {1: 1})


@unittest.skipUnless(HAS_COCO, "pycocotools is needed for COCO AP and annotation conversion")
class CocoTests(unittest.TestCase):
    def test_rgb_panoptic_roundtrip_preserves_large_ids(self):
        from tools.prepare_annotations import save_panoptic

        with tempfile.TemporaryDirectory() as directory:
            labels = np.array([[0, 70000, (1 << 24) - 1]], dtype=np.int64)
            path = Path(directory) / "sample.png"
            save_panoptic(labels, path)
            with Image.open(path) as image:
                self.assertEqual(image.mode, "RGB")
            decoded = read_panoptic(directory, "sample")
            np.testing.assert_array_equal(decoded, labels)
            metric = PanopticMetrics(["thing"], [0])
            segments = [{"id": 70000, "category_id": 1}, {"id": (1 << 24) - 1, "category_id": 1}]
            metric.update(decoded, segments, labels, segments)
            self.assertEqual(metric.compute()["All"]["PQ"], 100.0)
            with self.assertRaisesRegex(ValueError, "2\\^24"):
                save_panoptic(np.array([[1 << 24]]), path)

    def test_isaid_conversion_preserves_rgb_ids_and_rejects_rgba(self):
        from tools.prepare_annotations import convert_isaid, save_panoptic

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "annotations"
            panoptic_dir = source / "iSAID_val_panoptic"
            panoptic_dir.mkdir(parents=True)
            output = root / "converted"
            (output / "panoptic").mkdir(parents=True)
            labels = np.array([[70000, 16000000]], dtype=np.int64)
            save_panoptic(labels, panoptic_dir / "sample.png")
            (source / "iSAID_val_instance.json").write_text(json.dumps({
                "images": [{"id": 1, "file_name": "sample.png"}], "annotations": [], "categories": [],
            }))
            (source / "iSAID_val_panoptic.json").write_text(json.dumps({
                "annotations": [{"file_name": "sample.png", "segments_info": [
                    {"id": 70000, "category_id": 0}, {"id": 16000000, "category_id": 1},
                ]}],
            }))
            metadata = convert_isaid(root, output, "val", None, None, None)
            self.assertEqual(metadata["sample"], [
                {"id": 70000, "category_id": 1, "iscrowd": 0},
                {"id": 16000000, "category_id": 2, "iscrowd": 0},
            ])
            np.testing.assert_array_equal(read_panoptic(output / "panoptic", "sample"), labels)
            with Image.open(output / "panoptic/sample.png") as image:
                self.assertEqual(image.mode, "RGB")
            Image.new("RGBA", (2, 1)).save(panoptic_dir / "sample.png")
            with self.assertRaisesRegex(ValueError, "exactly three RGB channels"):
                convert_isaid(root, output, "val", None, None, None)

    def test_cityscapes_conversion_writes_rgb_panoptic_maps(self):
        from tools.prepare_annotations import write_regions

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = root / "gtFine/val/testcity"
            labels.mkdir(parents=True)
            raw = np.array([[7, 24001], [7, 24001]], dtype=np.uint16)
            Image.fromarray(raw).save(labels / "tiny_gtFine_instanceIds.png")
            record = ImageRecord("tiny_leftImg8bit", root / "leftImg8bit/val/testcity/tiny_leftImg8bit.png", root / "unused.png")
            output = root / "converted"
            (output / "panoptic").mkdir(parents=True)
            from tag.runtime import load_config
            config = load_config("cityscapes")
            metadata = write_regions([record], root, output, "cityscapes", "val", config)
            self.assertEqual(metadata[record.key], [
                {"id": 1, "category_id": 1, "iscrowd": 0},
                {"id": 2, "category_id": 12, "iscrowd": 0},
            ])
            with Image.open(output / "panoptic" / f"{record.key}.png") as image:
                self.assertEqual(image.mode, "RGB")
            np.testing.assert_array_equal(read_panoptic(output / "panoptic", record.key), [[1, 2], [1, 2]])

    def test_perfect_and_empty_instance_predictions(self):
        from pycocotools import mask as mask_utils

        mask = np.ones((4, 4), dtype=np.uint8)
        rle = mask_utils.encode(np.asfortranarray(mask))
        rle["counts"] = rle["counts"].decode("ascii")
        annotation = {
            "info": {}, "images": [{"id": 1, "file_name": "sample.png", "width": 4, "height": 4}],
            "categories": [{"id": 18, "name": "thing"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 18, "segmentation": rle,
                             "area": 16, "bbox": [0, 0, 4, 4], "iscrowd": 0}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "instances.json"
            path.write_text(json.dumps(annotation))
            prediction = [{"image_id": 1, "category_id": 1, "segmentation": rle, "score": 1.0}]
            result = evaluate_instances(prediction, path, ["stuff", "thing"], "ade20k", 100)
            self.assertAlmostEqual(result["AP"], 100.0)
            result = evaluate_instances([], path, ["stuff", "thing"], "ade20k", 100)
            self.assertEqual(result["AP"], 0.0)

    def test_ade_conversion_keeps_ignore_pixels_void(self):
        from tools.prepare_annotations import convert_ade

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "semantic.png"
            Image.fromarray(np.array([[1, 0, 255, 2]], dtype=np.uint8)).save(path)
            (root / "annotations_instance/validation").mkdir(parents=True)
            Image.fromarray(np.zeros((1, 4, 3), dtype=np.uint8)).save(root / "annotations_instance/validation/sample.png")
            record = ImageRecord("sample", path, path)
            shape, regions = convert_ade(record, root, "val", set(), {}, 2)
            self.assertEqual(shape, (1, 4))
            self.assertEqual([category for _, category, _ in regions], [0, 1])
            self.assertFalse(any(mask[0, 1] or mask[0, 2] for mask, _, _ in regions))


if __name__ == "__main__":
    unittest.main()
