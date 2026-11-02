"""Load AFUN training weights and legacy alignment checkpoints."""
from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def load_weights(model, path, *, allow_new_motion=False):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if "model" in checkpoint:
        state = dict(checkpoint["model"])
    elif "projector_state_dict" in checkpoint:
        state = {f"decoder.projector.{k}": v for k, v in checkpoint["projector_state_dict"].items()}
        state.update({f"llm.rms_norm_layer.{k}": v for k, v in (checkpoint.get("rms_norm_state_dict") or {}).items()})
        state.update({f"llm.llm_model.{k}": v for k, v in (checkpoint.get("qwen_llm_trainable_state_dict") or {}).items()})
    else:
        raise ValueError(f"{path} is not an AFUN or alignment checkpoint.")

    expected = model.state_dict()
    embedding_names = [k for k in expected if k.endswith("embed_tokens.weight")]
    for key in list(state):
        if key.endswith("embed_tokens.weight") and key not in expected and len(embedding_names) == 1:
            state[embedding_names[0]] = state.pop(key)
    # Legacy Stage 1 used 32 queries. Preserve its rows when adding the motion queries.
    for key in list(state):
        if key not in expected or state[key].shape == expected[key].shape:
            continue
        old, new = state[key], expected[key]
        if allow_new_motion and old.ndim == new.ndim == 2 and old.shape[1] == new.shape[1] and old.shape[0] < new.shape[0]:
            if key.endswith("embed_tokens.weight"):
                expanded = new.detach().cpu().clone()
                expanded[:old.shape[0]] = old
                state[key] = expanded
            elif key == "llm.rms_norm_layer.weight" and new.shape[0] % old.shape[0] == 0:
                state[key] = old.repeat(new.shape[0] // old.shape[0], 1)
    optional = ("decoder.motion_processor.", "decoder.depth_encoder.") if allow_new_motion else ()
    required = {
        name for name, parameter in model.named_parameters()
        if name.startswith(("decoder.projector.", "llm.rms_norm_layer.", "decoder.motion_processor."))
        or name.endswith("embed_tokens.weight")
        or (name.startswith("decoder.depth_encoder.") and parameter.requires_grad)
    }
    missing = sorted(k for k in required if k not in state and not k.startswith(optional))
    if missing:
        raise ValueError(f"Checkpoint is missing required weights: {', '.join(missing[:8])}")
    incompatible = model.load_state_dict(state, strict=False)
    ignored = ("decoder.loss.",)
    if not getattr(getattr(model, "decoder", None), "predict_motion", False):
        ignored += ("decoder.motion_processor.", "decoder.depth_encoder.")
    unexpected = [k for k in incompatible.unexpected_keys if not k.startswith(ignored)]
    if unexpected:
        raise ValueError(f"Unrecognized checkpoint weights: {', '.join(unexpected[:8])}")
    logger.info("Loaded %d tensors from %s", len(state), path)
    return checkpoint
