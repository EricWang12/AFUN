# src/models/end_to_end.py
from __future__ import annotations
from typing import Dict, TYPE_CHECKING
import torch
from torch import nn

from .encoder.base import EncoderBase
from .llm.base import LLMBase
if TYPE_CHECKING:  # type hints only; does NOT run at runtime
    from .decoder import Sam3Decoder

from .decoder.base import DecoderBase
from .projection.projection_mlp import ProjectionMLP

try:  # optional
    from .losses.binary_segmentation import BinarySegmentationLoss
except Exception:  # pragma: no cover
    BinarySegmentationLoss = None
                
class EndToEndModel(nn.Module):
    def __init__(self, encoder: EncoderBase|int|None, llm: LLMBase, decoder: DecoderBase, pc_projection: bool = False, rms_norm: bool = False, training: bool = True, freeze_segmentation: bool = False):
        super().__init__()
        
        if encoder is not None:
            if isinstance(encoder, int):
                self.encoder = nn.Identity()
                encoder_output_dim = encoder
            elif isinstance(encoder, nn.Module):
                self.encoder = encoder
                encoder_output_dim = encoder.output_dim
            else:
                raise ValueError(f"Invalid encoder type: {type(encoder)}")
            self.projection = ProjectionMLP(in_dim=encoder_output_dim, out_dim=llm.llm_hidden_size)
        else:
            self.encoder = nn.Identity()
            encoder_output_dim = 0
            self.projection = nn.Identity()
        # self.tokenizer = tokenizer
        self.llm = llm
        self.decoder = decoder

        self.sigmoid = nn.Sigmoid()
        if rms_norm and llm.rms_norm_layer is None:
            self.rms_norm = True
            self.rms_norm_layer = nn.RMSNorm((llm.num_metaqueries, llm.llm_hidden_size),eps=1e-5)
        else:
            self.rms_norm = False
        self.encoder_offloaded = False
        self.training = training

        if freeze_segmentation:
            # Freeze every parameter that affects segmentation outputs so motion
            # can train without changing segmentation behavior. Coordinates two
            # places: the decoder freezes sam3_model + projector + loss params,
            # and the LLM swaps its embed_tokens gradient hook so only the motion
            # metaquery rows receive gradients.
            if hasattr(self.decoder, "freeze_segmentation_components"):
                self.decoder.freeze_segmentation_components()
            else:
                raise AttributeError(
                    f"freeze_segmentation=True but decoder {type(self.decoder).__name__} "
                    "has no freeze_segmentation_components()."
                )
            if hasattr(self.llm, "freeze_segmentation_metaqueries"):
                self.llm.freeze_segmentation_metaqueries()
            else:
                raise AttributeError(
                    f"freeze_segmentation=True but llm {type(self.llm).__name__} "
                    "has no freeze_segmentation_metaqueries()."
                )

    def forward_with_pc_feats(self, points: torch.Tensor) -> Dict[str, torch.Tensor]:
        pass
    
    def offload_encoder(self, device = "cpu"):
        self.encoder.to(device)
        self.encoder_offloaded = True
    
    def forward(self,  prompt: str, pc: torch.Tensor | None = None, pc_feats: torch.Tensor | None = None, image: torch.Tensor | None = None, image_raw: torch.Tensor | None = None, object_points: torch.Tensor | None = None, **kwargs) -> Dict[str, torch.Tensor]:
        if pc is not None and object_points is None:
            object_points = pc.clone()
        if image is not None and pc is None and object_points is None:
            object_points = image
        
        if pc_feats is None and pc is not None:
            if not self.encoder_offloaded:
                with torch.no_grad():
                    pc_feats = self.encoder(pc) # List of feat in len(B)
                    
                    # pc_feats.view(pc.shape[0], -1, pc_feats.shape[-1])
                    
            else:
                raise ValueError("Encoder is offloaded. pc_feats must be provided for inference.")
            
        # pc_embeddings = self.projection(pc_feats)
        
        # pc_embeddings = pc_embeddings.view(pc.shape[0], -1, pc_embeddings.shape[-1])
        # # tok_out = self.tokenizer(pc_embeddings)
        with torch.no_grad():
            inputs_metaquery = self.llm.tokenize(prompt, image=image_raw if image_raw is not None else image)
            if pc_feats is not None:
                pc_embeddings = [self.projection(pc_feats[i]) for i in range(pc.shape[0])]
            else:
                pc_embeddings = None
            
            device = pc_embeddings[0].device if pc_embeddings is not None else image.device
        # pc_embeddings = pc_embeddings.view(pc.shape[0], -1, pc_embeddings.shape[-1])
        # tok_out = self.tokenizer(pc_embeddings)
        
        # device = pc_embeddings.device
        # Extract keys that should NOT go to the LLM
        depth = kwargs.pop("depth", None)
        head_intrinsic = kwargs.pop("head_intrinsic", None)
        find_target = kwargs.pop("find_target", None)
        has_motion = kwargs.pop("has_motion", False)

        outputs = self.llm(
            input_ids=inputs_metaquery["input_ids"].to(device),
            attention_mask=inputs_metaquery["attention_mask"].to(device),
            pixel_values=inputs_metaquery["pixel_values"].to(device) if "pixel_values" in inputs_metaquery else None,
            image_grid_thw=inputs_metaquery["image_grid_thw"].to(device) if "image_grid_thw" in inputs_metaquery else None,
            mm_token_type_ids=inputs_metaquery["mm_token_type_ids"].to(device) if "mm_token_type_ids" in inputs_metaquery else None,
            point_cloud_embeddings=pc_embeddings,
            **kwargs
        )
        
        if type(outputs["metaquery_prompt_embeds"]) == dict:
            outputs["metaquery_semantic_prompt_embeds"] = outputs["metaquery_prompt_embeds"]["semantic"]
            outputs["metaquery_semantic_attention_mask"] = outputs["metaquery_attention_mask"]["semantic"]
            
            if has_motion:
                outputs["metaquery_motion_prompt_embeds"] = outputs["metaquery_prompt_embeds"]["motion"]
                outputs["metaquery_motion_attention_mask"] = outputs["metaquery_attention_mask"]["motion"]
            # outputs["metaquery_prompt_embeds"] = torch.cat([outputs["metaquery_motion_prompt_embeds"], outputs["metaquery_semantic_prompt_embeds"]], dim=1)
            # outputs["metaquery_attention_mask"] = torch.cat([outputs["metaquery_motion_attention_mask"], outputs["metaquery_semantic_attention_mask"]], dim=1)
            
        elif self.rms_norm and self.rms_norm_layer is not None and self.llm.rms_norm_layer is None:
            outputs["metaquery_prompt_embeds"] = self.rms_norm_layer(outputs["metaquery_prompt_embeds"])
        # outputs = {}
        # N = object_points.shape[0]
        # outputs["metaquery_semantic_prompt_embeds"] = torch.randn((N, 32, 4096), device=device, dtype=torch.bfloat16)
        # outputs["metaquery_semantic_attention_mask"] = torch.ones((N, 32), device=device, dtype=torch.long)
        # outputs["metaquery_motion_prompt_embeds"] = torch.randn((N, 32, 4096), device=device, dtype=torch.bfloat16)
        # outputs["metaquery_motion_attention_mask"] = torch.ones((N, 32), device=device, dtype=torch.long)

                
        # rand_tensor = torch.rand_like(outputs["metaquery_prompt_embeds"])

        # unwrap torch.compile if present
        _dec = getattr(self.decoder, "_orig_mod", self.decoder)
        dec_cls = _dec.__class__
        dec_name = dec_cls.__name__
        dec_fqcn = f"{dec_cls.__module__}.{dec_cls.__qualname__}"

        if dec_name == "Sam3Decoder":
            motion_feature = outputs.get("metaquery_motion_prompt_embeds", None) if has_motion else None
            outputs, loss_and_metrics = self.decoder(
                object_points,
                feature=outputs["metaquery_semantic_prompt_embeds"],
                motion_feature=motion_feature,
                find_target=find_target,
                depth=depth if has_motion else None,
                head_intrinsic=head_intrinsic if has_motion else None,
                has_motion=has_motion,
            )
            metrics = {k: v for k, v in loss_and_metrics.items() if k != "core_loss"}
            result = {"outputs": outputs, "losses": loss_and_metrics.get("core_loss", None), "metrics": metrics, **inputs_metaquery}
            if motion_feature is not None:
                result["metaquery_motion_prompt_embeds"] = motion_feature
            return result
        else:
            raise TypeError(f"Unsupported decoder type: {dec_fqcn}")
