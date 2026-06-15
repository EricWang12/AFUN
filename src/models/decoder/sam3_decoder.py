from __future__ import annotations

import contextlib
import os
import sys
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
from torch import nn
from omegaconf import ListConfig, DictConfig
from collections.abc import Mapping
from dataclasses import fields as dc_fields, is_dataclass
from sam3.model.model_misc import SAM3Output  # pyright: ignore[reportMissingImports]

from .base import DecoderBase
from hydra.utils import instantiate
from sam3.model_builder import build_sam3_image_model # pyright: ignore[reportMissingImports]
from sam3.visualization_utils import plot_results  # pyright: ignore[reportMissingImports]
import wandb  # type: ignore
from PIL import Image

from src.utils.motion_vis import visualize_trajectory_on_mask
from src.utils.misc import *


def _to_device(x: Any, dev: str):
    if torch.is_tensor(x):
        return x.to(dev)
    if isinstance(x, (list, tuple)):
        return type(x)(_to_device(v, dev) for v in x)
    if is_dataclass(x):
        for f in dc_fields(x):
            v = getattr(x, f.name)
            if v is None:
                continue
            setattr(x, f.name, _to_device(v, dev))
        return x
    return x




class _QwenToSam3Projector(nn.Module):
    """Match `qwen_to_sam_metaquery.py` projector (2-layer MLP)."""

    def __init__(self, in_dim: int, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Sam3Decoder(DecoderBase):
    """
    Decoder wrapper that uses SAM3 for (image) grounding/segmentation.

    This follows the `DecoderBase.forward(point_cloud, feature, images, ...)` convention used in this repo:
    - `point_cloud` is ignored (kept only for API compatibility)
    - `images` is required: (B, 3, H, W), ideally already resized+normalized like SAM3 expects
    - `feature` is the LLM metaquery prompt embeddings: (B, Q, H_llm)
    - `metaquery_attention_mask` (optional) should be (B, Q) with 1 for valid tokens, 0 for padding

    Internally we project LLM embeddings -> 256 dim and feed SAM3 via `forward_grounding`, using the same
    "language_features/language_mask" convention as SAM3's own text encoder outputs.
    """

    def __init__(
        self,
        model_cfg: Optional[Dict[str, Any]] = None,
        ckpt_path: Optional[str] = None,
        loss:  Optional[Dict[str, Any]] = None,
        motion_processor: Optional[Any] = None,
        depth_encoder: Optional[Any] = None,
        training: bool = True,
        freeze: bool = False,
        **kwargs,
    ):
        super().__init__(freeze)
        self.freeze = freeze
        self.training = training
        self.predict_motion = kwargs.get("predict_motion", False)
        # Toggled by EndToEndModel when the experiment yaml sets `freeze_segmentation: true`.
        # When True, forward() skips SAM3 loss and runs SAM3 (incl. projector) under no_grad.
        self.freeze_segmentation: bool = False
        
        model_cfg = model_cfg or {}
        self.model_cfg = model_cfg

        _default_bpe = os.path.join(os.path.dirname(__file__), "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")
        self.sam3_bpe_path: str = model_cfg.get("sam3_bpe_path", _default_bpe)
        # ---- projector config (LLM hidden -> 256) ----
        self.projector_in_dim: Optional[int] = model_cfg.get("projector_in_dim", None)
        self.projector_out_dim: int = int(model_cfg.get("projector_out_dim", 256))

        # Optional RMSNorm in the original inference script; we support it if present in ckpt.
        self.rms_norm: Optional[nn.Module] = None

        # ---- SAM3 config ----
        self.sam3_device: str = str(model_cfg.get("device", "cuda"))
        self.sam3_checkpoint_path: Optional[str] = model_cfg.get("sam3_checkpoint_path", None)
        self.sam3_load_from_hf: bool = bool(model_cfg.get("sam3_load_from_hf", True))
        self.sam3_enable_segmentation: bool = bool(model_cfg.get("enable_segmentation", True))
        self.sam3_enable_inst_interactivity: bool = bool(model_cfg.get("enable_inst_interactivity", False))
        self.sam3_compile: bool = bool(model_cfg.get("compile", False))

        # Postprocess config (convert pred_* -> masks/boxes/scores)
        self.confidence_threshold: float = float(model_cfg.get("confidence_threshold", 0.5))
        # If True, return the full SAM3 output dict (optionally with postprocessed masks/boxes/scores).
        # If False (default), return a (B,1,H,W) logits tensor (best mask per image), to match other decoders.
        self.return_dict: bool = bool(model_cfg.get("return_dict", False))


        # self.loss = {
        #     key: el  # wrap_base_loss(el)
        #     for (key, el) in instantiate(loss, _convert_="all").items()
        # }
        # self.loss = nn.ModuleDict(self.loss)
        loss_obj = loss
        if isinstance(loss, (DictConfig, ListConfig, dict, list)):
            loss_obj = instantiate(loss, _convert_="all")

        if loss_obj is None:
            self.loss = None                      # training-only; never used at inference
        elif isinstance(loss_obj, nn.Module):
            self.loss = nn.ModuleDict({"main": loss_obj})
        elif isinstance(loss_obj, Mapping):
            self.loss = nn.ModuleDict({k: v for k, v in loss_obj.items()})
        else:
            raise TypeError(f"Unsupported loss type: {type(loss_obj)}")
    
        # Build modules
        self.sam3_model: Optional[nn.Module] = None
        self.projector = _QwenToSam3Projector(self.projector_in_dim, self.projector_out_dim).to(self.sam3_device)
        self.motion_processor: Optional[nn.Module] = None
        if self.predict_motion:
            # Hydra instantiate(recursive=True) will often instantiate nested `_target_` configs first,
            # so `motion_processor` may already be an nn.Module (not a DictConfig). Support both.
            if motion_processor is None:
                self.motion_processor = None
            elif isinstance(motion_processor, nn.Module):
                self.motion_processor = motion_processor
            elif isinstance(motion_processor, (DictConfig, dict)):
                self.motion_processor = instantiate(motion_processor)
            else:
                raise TypeError(
                    "Unsupported `motion_processor` type. Expected nn.Module or (DictConfig/dict), "
                    f"got: {type(motion_processor)}"
                )

        # Depth geo encoder (frozen DFormerv2 + trainable projection)
        self.depth_encoder: Optional[nn.Module] = None
        if self.predict_motion and depth_encoder is not None:
            if isinstance(depth_encoder, nn.Module):
                self.depth_encoder = depth_encoder
            elif isinstance(depth_encoder, (DictConfig, dict)):
                self.depth_encoder = instantiate(depth_encoder)
            else:
                raise TypeError(
                    "Unsupported `depth_encoder` type. Expected nn.Module or (DictConfig/dict), "
                    f"got: {type(depth_encoder)}"
                )

        # Build the SAM3 backbone. The trained params (projector, decoder head,
        # motion processor) are filled afterward when the full model checkpoint
        # loads via EndToEndModel (strict=False) — so the projector starts at
        # random init here and gets overwritten.
        self.load_model(ckpt_path=ckpt_path, device=self.sam3_device)
        if model_cfg.get("freeze_sam", False):
            for param in self.sam3_model.parameters():
                param.requires_grad = False
        if freeze:
            self.apply_freeze()

    def freeze_segmentation_components(self) -> None:
        """Freeze every parameter that affects segmentation outputs.

        Set by EndToEndModel when the experiment yaml has `freeze_segmentation: true`.
        Sets requires_grad=False on the SAM3 model, the Qwen->SAM3 projector, and any
        learnable params inside the loss module. Also flips the `freeze_segmentation`
        flag so forward() can short-circuit the SAM3 loss path and wrap the SAM3
        forward in torch.no_grad().
        """
        self.freeze_segmentation = True
        if self.sam3_model is not None:
            for p in self.sam3_model.parameters():
                p.requires_grad = False
        for p in self.projector.parameters():
            p.requires_grad = False
        if hasattr(self, "loss") and self.loss is not None:
            for p in self.loss.parameters():
                p.requires_grad = False

    def _ensure_sam3_imports(self):
        """
        SAM3 lives under `src/models/decoder/sam3/sam3/` in this repo.
        Add `.../decoder/sam3/` to sys.path so `import sam3` works without external install.
        """
        sam3_root = os.path.join(os.path.dirname(__file__), "sam3")
        if sam3_root not in sys.path:
            sys.path.append(sam3_root)

        # Late imports to avoid import-time side effects when SAM3 isn't used.
        from sam3.model_builder import build_sam3_image_model  # noqa: F401  # pyright: ignore[reportMissingImports]
        from sam3.model.data_misc import FindStage  # noqa: F401  # pyright: ignore[reportMissingImports]
        from sam3.model import box_ops  # noqa: F401  # pyright: ignore[reportMissingImports]
        from sam3.model.data_misc import interpolate  # noqa: F401  # pyright: ignore[reportMissingImports]

    def load_model(self, ckpt_path: str = None, device: str = "cuda", **kwargs):
        self._ensure_sam3_imports()


        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.sam3_device = device

        # ---- build SAM3 ----
        self.sam3_model = build_sam3_image_model(
            bpe_path=self.sam3_bpe_path,
            device=device,
            checkpoint_path=self.sam3_checkpoint_path,
            load_from_HF=self.sam3_load_from_hf,
            enable_segmentation=self.sam3_enable_segmentation,
            enable_inst_interactivity=self.sam3_enable_inst_interactivity,
            compile=self.sam3_compile,
            eval_mode=not self.training,
        )

    def _build_find_stage(self, batch_size: int, device: torch.device):
        from sam3.model.data_misc import FindStage  # pyright: ignore[reportMissingImports]

        img_ids = torch.arange(batch_size, device=device, dtype=torch.long)
        text_ids = torch.arange(batch_size, device=device, dtype=torch.long)
        return FindStage(
            img_ids=img_ids,
            text_ids=text_ids,
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

    def _project_language(
        self,
        metaquery_prompt_embeds: torch.Tensor,  # (B,Q,H_llm)
        metaquery_attention_mask: Optional[torch.Tensor],  # (B,Q), 1 valid, 0 pad
    ) -> Dict[str, torch.Tensor]:
        if self.projector is None:
            raise RuntimeError("Projector is not initialized. Did `load_model` run?")

        x = metaquery_prompt_embeds
        # if self.rms_norm is not None:
        #     x = self.rms_norm(x)

        if metaquery_attention_mask is None:
            metaquery_attention_mask = torch.ones(x.shape[0], x.shape[1], device=x.device, dtype=torch.long)

        with torch.amp.autocast(
            device_type=("cuda" if str(x.device).startswith("cuda") else "cpu"),
            enabled=str(x.device).startswith("cuda"),
            dtype=torch.bfloat16,
        ):
            proj = self.projector(x)  # (B,Q,256)

        # SAM3 expects:
        # - language_features: (L, num_prompts, C)
        # - language_mask:     (num_prompts, L) where True means masked out
        language_features = proj.permute(1, 0, 2).contiguous()  # (Q,B,C)
        language_mask = (metaquery_attention_mask == 0).to(torch.bool)  # (B,Q)
        return {"language_features": language_features, "language_mask": language_mask}

    def _postprocess(
        self,
        outputs: Dict[str, torch.Tensor],
        orig_sizes: List[Tuple[int, int]],
        has_motion: bool = False,
    ) -> Dict[str, Any]:
        """
        Convert SAM3 raw outputs (`pred_*`) into per-image lists:
        - masks_logits: List[Tensor(K_i, H, W)]
        - masks:       List[Tensor(K_i, H, W)] bool
        - boxes:       List[Tensor(K_i, 4)] in xyxy pixel coords
        - scores:      List[Tensor(K_i)]
        """
        from sam3.model import box_ops  # pyright: ignore[reportMissingImports]

        pred_boxes = outputs["pred_boxes"]          # (B, N, 4) cxcywh in [0,1]
        pred_logits = outputs["pred_logits"]        # (B, N, 1) or (B, N)
        pred_masks = outputs.get("pred_masks", None)  # (B, N, h, w) logits
        presence = outputs.get("presence_logit_dec", None)  # (B, N) or (B, N, 1)

        if pred_logits.ndim == 3 and pred_logits.shape[-1] == 1:
            pred_logits_ = pred_logits[..., 0]
        else:
            pred_logits_ = pred_logits

        if presence is not None:
            if presence.ndim == 3 and presence.shape[-1] == 1:
                presence_ = presence[..., 0]
            else:
                presence_ = presence
            probs = pred_logits_.sigmoid() * presence_.sigmoid()
        else:
            probs = pred_logits_.sigmoid()

        boxes_list: List[torch.Tensor] = []
        scores_list: List[torch.Tensor] = []
        masks_logits_list: List[torch.Tensor] = []
        masks_list: List[torch.Tensor] = []
        motion_dirs_list: List[torch.Tensor] = []
        B = pred_boxes.shape[0]
        for b in range(B):
            keep = probs[b] > self.confidence_threshold  # (N,)
            scores = probs[b][keep]
            boxes_cxcywh = pred_boxes[b][keep]
            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes_cxcywh)

            h, w = orig_sizes[b]
            scale = torch.tensor([w, h, w, h], device=boxes_xyxy.device, dtype=boxes_xyxy.dtype)
            boxes_xyxy = boxes_xyxy * scale[None, :]

            if pred_masks is None:
                masks_logits = torch.empty((boxes_xyxy.shape[0], h, w), device=boxes_xyxy.device)
                masks = torch.empty((boxes_xyxy.shape[0], h, w), device=boxes_xyxy.device, dtype=torch.bool)
            else:
                m = pred_masks[b][keep]  # (K, h0, w0)
                m = torch.nn.functional.interpolate(
                    m.unsqueeze(1),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                masks_logits = m.sigmoid()
                masks = masks_logits > 0.5

            boxes_list.append(boxes_xyxy)
            scores_list.append(scores)
            masks_logits_list.append(masks_logits)
            masks_list.append(masks)
            if self.predict_motion and has_motion and "motion_directions" in outputs:
                motion_dirs = outputs["motion_directions"][b][keep]
                motion_dirs_list.append(motion_dirs)
        result = {
            "boxes": boxes_list,
            "scores": scores_list,
            "masks_logits": masks_logits_list,
            "masks": masks_list,
        }
        if self.predict_motion and has_motion and motion_dirs_list:
            result["motion_directions"] = motion_dirs_list
            # # tensor to list of tensors
            # motion_dirs = outputs["motion_directions"][keep]  # shape: (K, 3)
            
            # if isinstance(motion_dirs, torch.Tensor):
            #     result["motion_directions"] = [motion_dirs[i] for i in range(motion_dirs.shape[0])]
            # else:
            #     result["motion_directions"] = motion_dirs
                
        return result

    def forward(
        self,
        images: Optional[torch.Tensor] ,
        point_cloud: torch.Tensor = None,
        feature: Optional[torch.Tensor] = None,
        motion_feature: Optional[torch.Tensor] = None,
        normalize: bool = False,
        find_target: Optional[Dict[str, Any]] = None,   
        return_postprocessed: bool = False,
        **kwargs,
    ) -> torch.Tensor | Dict[str, Any]:
        if self.sam3_model is None:
            raise RuntimeError("SAM3 model is not initialized. Did `load_model` run?")
        if images is None:
            raise ValueError("Sam3Decoder requires `images` (B,3,H,W).")
        if feature is None:
            raise ValueError("Sam3Decoder requires `feature` (metaquery_prompt_embeds, shape BxQxH_llm).")
        if point_cloud is not None:
            print("Sam3Decoder does not support `point_cloud` input. Use `images` only. The point cloud is ignored for SAM3.")
            point_cloud = None
        device = torch.device(self.sam3_device if (self.sam3_device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
        has_motion = kwargs.pop("has_motion", False)

        images = images.to(device)
        feature = feature.to(device)
        if self.predict_motion and motion_feature is not None:
            motion_feature = motion_feature.to(device)

        # Caller can pass the attention mask in several common names.
        metaquery_attention_mask =  kwargs.get("metaquery_mask", None)

        if metaquery_attention_mask is not None:
            metaquery_attention_mask = metaquery_attention_mask.to(device)

        # When freeze_segmentation is on, skip autograd for the entire SAM3 path
        # (projector + backbone + grounding). The Hungarian matcher inside
        # forward_grounding still runs (it's non-differentiable anyway), so
        # outputs["indices"] is still populated for the motion loss.
        sam_ctx = torch.no_grad() if self.freeze_segmentation else contextlib.nullcontext()
        with sam_ctx:
            text_outputs = self._project_language(feature, metaquery_attention_mask)

            # Build per-batch FindStage so each image aligns to its corresponding prompt.
            B = images.shape[0]
            find_stage = self._build_find_stage(B, device=images.device)
            geometric_prompt = self.sam3_model._get_dummy_prompt(num_prompts=B)

            backbone_out: Dict[str, Any] = {"img_batch_all_stages": images}
            backbone_out.update(self.sam3_model.backbone.forward_image(images))
            backbone_out.update(text_outputs)

            if find_target is not None:
                find_target = _to_device(find_target, device)

            outputs = self.sam3_model.forward_grounding(
                backbone_out=backbone_out,
                find_input=find_stage,
                find_target=find_target,
                geometric_prompt=geometric_prompt,
            )

        if find_target is not None:
            if self.freeze_segmentation:
                # SAM3 is fully frozen + run under no_grad; don't compute its loss.
                # core_loss is filled in by the motion path below; default to 0 so
                # batches without motion supervision are a no-op rather than crashing.
                losses: Dict[str, torch.Tensor] = {
                    "sam3_loss": torch.zeros((), device=device),
                    "core_loss": torch.zeros((), device=device, requires_grad=False),
                }
            else:
                find_stages = SAM3Output(output=[[outputs]])
                targets_dict = self.sam3_model.back_convert(find_target)
                losses = self.loss["main"](find_stages, [targets_dict])

            if self.predict_motion and has_motion and motion_feature is not None:
                # Encode depth if encoder is available
                geo_tokens, geo_shape = None, None
                if self.depth_encoder is not None and kwargs.get("depth") is not None:
                    depth_input = kwargs["depth"].to(device)
                    intrinsic_input = kwargs.get("head_intrinsic", None)
                    if intrinsic_input is not None:
                        intrinsic_input = intrinsic_input.to(device)
                    geo_tokens, geo_shape = self.depth_encoder(
                        images, depth_input, intrinsic=intrinsic_input
                    )

                motion_pred = self.motion_processor(
                    outputs["queries"], motion_feature,
                    geo_tokens=geo_tokens,
                    mask_logits=outputs.get("pred_masks"),
                    geo_spatial_shape=geo_shape,
                )
                outputs["motion_directions"] = motion_pred.pred
                motion_loss = self.motion_processor.compute_loss(motion_pred.pred, find_target.motion_directions, indices=outputs['indices'])

                if self.freeze_segmentation:
                    # No SAM3 contribution to backprop; core_loss == motion_loss.
                    losses["motion_loss"] = motion_loss["loss"].detach()
                    losses["core_loss"] = motion_loss["loss"]
                else:
                    sam_loss_weight = float(self.model_cfg.get("sam_loss_weight", 1.0))
                    sam_weighted = sam_loss_weight * losses["core_loss"]
                    # Log both terms as their weighted contribution to core_loss,
                    # so sam3_loss + motion_loss ≈ core_loss and reweighting is visible.
                    losses["sam3_loss"] = sam_weighted.detach()
                    losses["motion_loss"] = motion_loss["loss"].detach()
                    losses["core_loss"] = sam_weighted + motion_loss["loss"]
            
            metric_keys = self.model_cfg.get("metric_keys", ["miou_semantic_seg"])
            
            # Only keep keys in metric_keys and "core_loss" in losses and return
            loss_and_metrics = {k: v for k, v in losses.items() if k in metric_keys or k == "core_loss"}

            return outputs, loss_and_metrics
        
        
        # Default behavior: return a logits tensor like other decoders do.
        # We pick the best mask per image (argmax over predicted class logits, optionally gated by presence).
        if not self.return_dict:
            pred_masks = outputs.get("pred_masks", None)
            if pred_masks is None:
                raise KeyError("SAM3 output missing `pred_masks`; cannot return logits tensor.")

            pred_logits = outputs["pred_logits"]
            if pred_logits.ndim == 3 and pred_logits.shape[-1] == 1:
                pred_logits_ = pred_logits[..., 0]
            else:
                pred_logits_ = pred_logits

            presence = outputs.get("presence_logit_dec", None)
            if presence is not None:
                if presence.ndim == 3 and presence.shape[-1] == 1:
                    presence_ = presence[..., 0]
                else:
                    presence_ = presence
                scores = pred_logits_.sigmoid() * presence_.sigmoid()
            else:
                scores = pred_logits_.sigmoid()

            best_idx = scores.argmax(dim=1)  # (B,)
            H, W = int(images.shape[-2]), int(images.shape[-1])
            best_masks = []
            for b in range(B):
                m = pred_masks[b, best_idx[b]]  # (h0,w0) logits
                m = torch.nn.functional.interpolate(
                    m[None, None, ...],
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
                best_masks.append(m)
            return torch.stack(best_masks, dim=0).unsqueeze(1)  # (B,1,H,W) logits

        # Dict behavior: raw SAM3 outputs (+ optional postprocessed masks/boxes/scores).
        if not return_postprocessed:
            return outputs, {}

        # Default "original" size = tensor size; caller can override.
        orig_sizes = kwargs.get("original_sizes", None)
        if orig_sizes is None:
            orig_sizes = [(int(images.shape[-2]), int(images.shape[-1])) for _ in range(B)]

        post = self._postprocess(outputs, orig_sizes=orig_sizes)
        return {**outputs, **post}, {}

    def normalize_pc(self, pc: np.ndarray | torch.Tensor):
        # SAM3 is image-based; keep point cloud unchanged for API compatibility.
        return pc

    def make_wandb_vis(self, batch: Dict[str, Any], model_outputs: Dict[str, Any], *, step: int, idx: int = 0, draw_trajectory: bool = False, output_path: str = None):
        """
        Minimal SAM3 visualization for WandB:
        - postprocess predictions via `self._postprocess(...)`
        - plot with SAM3 `plot_results(...)`
        - returns dict ready to merge into `wandb.log(...)`
        """
        if "image" not in batch or batch["image"] is None:
            return None
        if "find_target" not in batch or batch["find_target"] is None:
            return None
        if "outputs" not in model_outputs or model_outputs["outputs"] is None:
            return None


        img_bchw = batch["image"]
        if not torch.is_tensor(img_bchw) or img_bchw.ndim != 4:
            return None
        B, _, H, W = map(int, img_bchw.shape)
        # Default to predict_motion: combined data modules set has_motion per-sample,
        # but single-source datasets (e.g. robot) don't — so fall back to the model config.
        has_motion = batch.get("has_motion", bool(self.predict_motion))

        idx = idx
        x = img_bchw[idx].detach().float()
        if float(x.min()) < 0.0:
            x = x * 0.5 + 0.5  # [-1,1] -> [0,1]
        x = x.clamp(0.0, 1.0)
        img_uint8 = (x * 255.0).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        img_pil = Image.fromarray(img_uint8)

        q = batch.get("query", None)
        q0 = q[idx] if isinstance(q, list) and len(q) > idx else q
        caption = f"step={int(step)} | query={q0}"

        post = self._postprocess(model_outputs["outputs"], orig_sizes=[(H, W)] * B, has_motion=has_motion)
        pred_results = {
            "boxes": post["boxes"][idx].detach(),
            "scores": post["scores"][idx].detach(),
            "masks": post["masks"][idx].detach().to(torch.float32).unsqueeze(1),  # (K,1,H,W)
            "motion_directions": post["motion_directions"][idx].detach() if (self.predict_motion and has_motion and "motion_directions" in post) else None,
        }

        while post["boxes"][idx].shape[0] == 0 and idx < B - 1:
            idx += 1
            pred_results = {
                "boxes": post["boxes"][idx].detach(),
                "scores": post["scores"][idx].detach(),
                "masks": post["masks"][idx].detach().to(torch.float32).unsqueeze(1),  # (K,1,H,W)
                "motion_directions": post["motion_directions"][idx].detach() if (self.predict_motion and has_motion and "motion_directions" in post) else None,
            }
        
        if post["boxes"][idx].shape[0] == 0:
            return {}
        
        vis_images = {}

        pred_fig = plot_results(img_pil, pred_results)
        
        ft = batch["find_target"]
        num_boxes = getattr(ft, "num_boxes", None)
        boxes_packed = getattr(ft, "boxes", None)
        segs_packed = getattr(ft, "segments", None)
        if not (torch.is_tensor(num_boxes) and torch.is_tensor(boxes_packed) and torch.is_tensor(segs_packed)):
            vis_images.update({"vis/sam3_pred": wandb.Image(pred_fig, caption=caption)})
            return vis_images

        start = int(num_boxes[:idx].sum().item())
        K = int(num_boxes[idx].item())
        if K <= 0:
            vis_images.update({"vis/sam3_pred": wandb.Image(pred_fig, caption=caption)})
            return vis_images

        gt_boxes_cxcywh = boxes_packed.view(-1, 4)[start : start + K]  # (K,4) norm cxcywh
        cx, cy, bw, bh = gt_boxes_cxcywh.unbind(dim=-1)
        x0 = (cx - 0.5 * bw) * float(W)
        y0 = (cy - 0.5 * bh) * float(H)
        x1 = (cx + 0.5 * bw) * float(W)
        y1 = (cy + 0.5 * bh) * float(H)
        gt_boxes_xyxy = torch.stack([x0, y0, x1, y1], dim=-1)

        gt_results = {
            "boxes": gt_boxes_xyxy,
            "scores": torch.ones((K,), device=gt_boxes_xyxy.device, dtype=torch.float32),
            "masks": segs_packed[start : start + K].to(torch.float32).unsqueeze(1),  # (K,1,H,W)
        }
        if self.predict_motion and has_motion and self.motion_processor is not None and hasattr(self.motion_processor, "make_wandb_vis"):
            vis_images.update(self.motion_processor.make_wandb_vis(batch, pred_results, gt_results, step=step, idx=idx, output_path=output_path))

        # if self.predict_motion and draw_trajectory and batch.get("traj_data", None) is not None and batch.get("head_intrinsic", None) is not None:
        #     traj_data = batch["traj_data"][idx]
        #     head_intrinsic = batch["head_intrinsic"][idx]
            
        #     if  post["masks"][idx].shape[0] != 0:
        #         vis_image = visualize_trajectory_on_mask(img_pil, 
        #                                     post["masks"][idx].detach().to(torch.float32).squeeze(1), 
        #                                     post["motion_directions"][idx].detach(), 
        #                                     None, 
        #                                     head_intrinsic,
        #                                     arrow_length=80,
        #                                     figsize=(10, 10))
            
        #         gt_image = visualize_trajectory_on_mask(img_pil, 
        #                                     batch['find_target'].segments[0].detach(), 
        #                                     batch['find_target'].motion_directions[idx].detach(), 
        #                                     traj_data, 
        #                                     head_intrinsic,
        #                                     arrow_length=80,
        #                                     figsize=(10, 10))
                
        #         # vis_images["vis/motion_pred"] = wandb.Image(vis_image, caption=caption)
        #         # vis_images["vis/motion_gt"] = wandb.Image(gt_image, caption=caption)
        #         vis_images["vis/motion_pred_vs_gt"] = wandb.Image(concat_side_by_side(vis_image, gt_image), caption=caption)
        #         if output_path is not None:
        #             vis_images["vis/sam3_pred_vs_gt"].save(output_path + "/sam3_pred_vs_gt.png")


        gt_fig = plot_results(img_pil, gt_results)

        vis_images.update({
            # "vis/sam3_pred": wandb.Image(pred_fig, caption=caption),
            # "vis/sam3_gt": wandb.Image(gt_fig, caption=caption),
            "vis/sam3_pred_vs_gt": wandb.Image(concat_side_by_side(pred_fig, gt_fig), caption=caption),
        })
        
        # if hasattr(self.motion_processor, "make_wandb_vis"):
        #     vis_images.update(self.motion_processor.make_wandb_vis(batch, model_outputs, step=step, idx=idx, output_path=output_path))
            
        return vis_images
    
    def _debug_save_vis(
        self,
        images: torch.Tensor,                  # (B,3,H,W) normalized [-1,1] or [0,1]
        outputs: Dict[str, Any],               # SAM3 forward_grounding output
        find_target: Optional[Any] = None,     # your packed targets (optional)
        **kwargs,
    ) -> None:
        debug_dir = kwargs.get("debug_vis_dir", None)  # e.g. "/tmp/sam3_debug"
        if not debug_dir:
            return


        step = int(kwargs.get("global_step", kwargs.get("step", 0)))
        every = int(kwargs.get("debug_vis_every", 1))
        if every > 1 and (step % every != 0):
            return

        os.makedirs(debug_dir, exist_ok=True)

        idx = int(kwargs.get("debug_vis_idx", 0))
        B, _, H, W = images.shape
        idx = max(0, min(idx, B - 1))

        # image -> uint8 PIL
        x = images[idx].detach().float()
        if float(x.min()) < 0.0:
            x = x * 0.5 + 0.5  # [-1,1] -> [0,1]
        x = x.clamp(0.0, 1.0)
        img = Image.fromarray((x * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy())

        # pred fig
        post = self._postprocess(outputs, orig_sizes=[(H, W)] * B)
        pred_results = {
            "boxes": post["boxes"][idx].detach().cpu(),
            "scores": post["scores"][idx].detach().cpu(),
            "masks": post["masks"][idx].detach().to(torch.float32).unsqueeze(1).cpu(),  # (K,1,H,W)
        }
        pred_fig = plot_results(img, pred_results)

        out_img = pred_fig
        start = K = 0  # for optional motion below

        # gt fig (if available) + concat
        ft = find_target
        if ft is not None and hasattr(ft, "num_boxes") and hasattr(ft, "boxes") and hasattr(ft, "segments"):
            num_boxes, boxes_packed, segs_packed = ft.num_boxes, ft.boxes, ft.segments
            if torch.is_tensor(num_boxes) and torch.is_tensor(boxes_packed) and torch.is_tensor(segs_packed):
                start = int(num_boxes[:idx].sum().item())
                K = int(num_boxes[idx].item())
                if K > 0:
                    gt_boxes_cxcywh = boxes_packed.view(-1, 4)[start : start + K]  # (K,4) norm cxcywh
                    cx, cy, bw, bh = gt_boxes_cxcywh.unbind(dim=-1)
                    x0 = (cx - 0.5 * bw) * float(W)
                    y0 = (cy - 0.5 * bh) * float(H)
                    x1 = (cx + 0.5 * bw) * float(W)
                    y1 = (cy + 0.5 * bh) * float(H)
                    gt_boxes_xyxy = torch.stack([x0, y0, x1, y1], dim=-1).cpu()

                    gt_results = {
                        "boxes": gt_boxes_xyxy,
                        "scores": torch.ones((K,), dtype=torch.float32),
                        "masks": segs_packed[start : start + K].to(torch.float32).unsqueeze(1).cpu(),
                    }
                    gt_fig = plot_results(img, gt_results)
                    out_img = concat_side_by_side(pred_fig, gt_fig)

        Image.fromarray(out_img).save(os.path.join(debug_dir, f"sam3_pred_vs_gt_step{step}_b{idx}.png"))

        # optional motion pred vs gt (only if caller passes traj_data/head_intrinsic)
        if (
            self.predict_motion
            and ("motion_directions" in outputs)
            and (ft is not None)
            and hasattr(ft, "motion_directions")
        ):
            traj_data = kwargs.get("traj_data", None)
            head_intrinsic = kwargs.get("head_intrinsic", None)
            if traj_data is not None and head_intrinsic is not None and post["masks"][idx].numel() > 0 and K > 0:
                mp = visualize_trajectory_on_mask(
                    img,
                    post["masks"][idx].detach().to(torch.float32).cpu(),
                    outputs["motion_directions"][idx].detach().cpu(),
                    None,
                    head_intrinsic[idx],
                    arrow_length=80,
                    figsize=(10, 10),
                )
                mg = visualize_trajectory_on_mask(
                    img,
                    ft.segments[start : start + K].detach().to(torch.float32).cpu(),
                    ft.motion_directions[idx].detach().cpu(),
                    traj_data[idx],
                    head_intrinsic[idx],
                    arrow_length=80,
                    figsize=(10, 10),
                )
                Image.fromarray(concat_side_by_side(mp, mg)).save(
                    os.path.join(debug_dir, f"motion_pred_vs_gt_step{step}_b{idx}.png")
                )

