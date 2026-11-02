"""AFUN language-conditioned functional segmentation and motion model."""
from __future__ import annotations

import torch
from torch import nn


class EndToEndModel(nn.Module):
    def __init__(self, encoder, llm, decoder, rms_norm=False, freeze_segmentation=False, **kwargs):
        super().__init__()
        if encoder is not None:
            raise ValueError("AFUN uses the decoder's RGB-D encoder; encoder must be None.")
        self.llm = llm
        self.decoder = decoder
        if rms_norm:
            raise ValueError("Configure RMS normalization through llm.llm_cfg.rms_norm.")
        if freeze_segmentation:
            self.decoder.freeze_segmentation_components()
            self.llm.freeze_segmentation_metaqueries()

    def encode_prompt(self, prompt, image, device):
        with torch.no_grad():
            tokens = self.llm.tokenize(prompt, image=image)
        keys = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids")
        inputs = {key: tokens[key].to(device) for key in keys if key in tokens}
        return self.llm(**inputs, point_cloud_embeddings=None)

    def forward(self, prompt, image, image_raw=None, find_target=None,
                depth=None, head_intrinsic=None, has_motion=False, **kwargs):
        if image_raw is None:
            image_raw = (image + 1) / 2
        features = self.encode_prompt(prompt, image_raw, image.device)
        embeddings = features["metaquery_prompt_embeds"]
        masks = features["metaquery_attention_mask"]
        motion = None
        if isinstance(embeddings, dict):
            if has_motion:
                motion = embeddings["motion"]
            embeddings, masks = embeddings["semantic"], masks["semantic"]
        outputs, losses = self.decoder(
            image, feature=embeddings, metaquery_mask=masks, motion_feature=motion,
            find_target=find_target, depth=depth if has_motion else None,
            head_intrinsic=head_intrinsic if has_motion else None, has_motion=has_motion,
        )
        return {"outputs": outputs, "losses": losses.get("core_loss"),
                "metrics": {k: v for k, v in losses.items() if k != "core_loss"}}
