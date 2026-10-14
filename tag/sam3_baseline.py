"""Inference-only wrapper around the unmodified official SAM3 image model.

The official raw grounding API keeps query logits, native presence logits, and
semantic masks separate until the task-specific baseline readout.
"""

from pathlib import Path
from importlib.metadata import PackageNotFoundError, version

import torch
from torchvision.transforms import functional as TF


class FrozenSAM3:
    def __init__(self, backbone_path, class_names, device="cuda", prompt_batch_size=8):
        backbone_path = Path(backbone_path)
        if not backbone_path.is_file():
            raise FileNotFoundError(f"Official SAM3 checkpoint not found: {backbone_path}")
        self.device = torch.device(device)
        self.prompts = list(class_names)
        self.prompt_batch_size = prompt_batch_size
        if not self.prompts or prompt_batch_size < 1:
            raise ValueError("Provide class names and a positive prompt batch size.")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Use a CUDA-enabled PyTorch environment.")
        self.use_bfloat16 = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        try:
            from sam3.model_builder import build_sam3_image_model
        except ImportError as error:
            raise ImportError("Install the official SAM3 package as described in README.md.") from error

        # A local checkpoint is mandatory: do not silently evaluate random weights.
        self.model = build_sam3_image_model(
            checkpoint_path=str(backbone_path),
            load_from_HF=False,
            device="cpu",
            eval_mode=True,
            enable_segmentation=True,
            enable_inst_interactivity=False,
        ).to(self.device)
        self.model.requires_grad_(False).eval()
        try:
            sam3_version = version("sam3")
        except PackageNotFoundError:
            sam3_version = "unknown"
        self.environment = {
            "torch": torch.__version__, "sam3": sam3_version,
            "device": str(self.device), "bfloat16": self.use_bfloat16,
        }
        if getattr(self.model, "supervise_joint_box_scores", False):
            raise RuntimeError("Expected separate query/presence logits from the official image model.")
        with torch.inference_mode(), self._autocast():
            self.text = self.model.backbone.forward_text(self.prompts, device=self.device)

    def _autocast(self):
        return torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.use_bfloat16)

    @torch.inference_mode()
    def predict_queries(self, image, prompt_batch_size=None):
        from sam3.model.data_misc import FindStage

        batch_size = prompt_batch_size if prompt_batch_size is not None else self.prompt_batch_size
        if batch_size < 1:
            raise ValueError("Prompt batch size must be positive.")
        image = TF.resize(image.convert("RGB"), [1008, 1008])
        tensor = TF.normalize(TF.to_tensor(image), [0.5] * 3, [0.5] * 3)
        with self._autocast():
            visual = self.model.backbone.forward_image(tensor[None].to(self.device))
        for start in range(0, len(self.prompts), batch_size):
            end = min(start + batch_size, len(self.prompts))
            count = end - start
            context = dict(visual)
            context.update({
                "language_features": self.text["language_features"][:, start:end],
                "language_mask": self.text["language_mask"][start:end],
                "language_embeds": self.text["language_embeds"][:, start:end],
            })
            inputs = FindStage(
                img_ids=torch.zeros(count, dtype=torch.long, device=self.device),
                text_ids=torch.arange(count, device=self.device),
                input_boxes=None,
                input_boxes_mask=None,
                input_boxes_label=None,
                input_points=None,
                input_points_mask=None,
            )
            with self._autocast():
                output = self.model.forward_grounding(
                    backbone_out=context,
                    find_input=inputs,
                    find_target=None,
                    geometric_prompt=self.model._get_dummy_prompt(count),
                )
            for index in range(count):
                yield {
                    "logits": output["pred_logits"][index].squeeze(-1).float(),
                    "masks": output["pred_masks"][index].float(),
                    "presence": output["presence_logit_dec"][index].squeeze().float(),
                    "semantic": output["semantic_seg"][index].squeeze(0).float(),
                }
            del output, context
