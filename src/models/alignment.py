"""Visual Genome alignment using frozen SAM3 teacher and student branches."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .end_to_end import EndToEndModel


def alignment_loss(student, teacher):
    keys = ("encoder_hidden_states", "decoder_hidden_states", "seg_head_outputs")
    losses = []
    for key in keys:
        if key not in student or key not in teacher:
            raise KeyError(f"SAM3 is missing {key}; install the SAM3 fork included with AFUN.")
        if student[key].shape != teacher[key].shape:
            raise ValueError(f"Alignment shape mismatch for {key}.")
        losses.append(F.mse_loss(student[key].float(), teacher[key].detach().float()))
    return torch.stack(losses).mean()


class AlignmentModel(EndToEndModel):
    def forward(self, prompt, image, image_raw=None, **kwargs):
        features = self.encode_prompt(prompt, image_raw, image.device)
        embeddings = features["metaquery_prompt_embeds"]
        masks = features["metaquery_attention_mask"]
        if isinstance(embeddings, dict):
            embeddings, masks = embeddings["semantic"], masks["semantic"]
        sam = self.decoder.sam3_model
        # Both branches use the same frozen image features and evaluation behavior.
        sam.eval()
        stage = self.decoder._build_find_stage(image.shape[0], image.device)
        geometric_prompt = sam._get_dummy_prompt(num_prompts=image.shape[0])
        with torch.no_grad():
            backbone = {"img_batch_all_stages": image, **sam.backbone.forward_image(image)}
            teacher = sam.forward_grounding(
                backbone_out={**backbone, **sam.backbone.forward_text(prompt, device=image.device)},
                find_input=stage, find_target=None, geometric_prompt=geometric_prompt,
            )["alignment_elements"]
        student = sam.forward_grounding(
            backbone_out={**backbone, **self.decoder._project_language(embeddings, masks)},
            find_input=stage, find_target=None, geometric_prompt=geometric_prompt,
        )["alignment_elements"]
        return {"losses": alignment_loss(student, teacher), "metrics": {}}
