"""Construct the same model for training and evaluation."""
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict


def build_model(cfg, device="cuda", *, alignment=False, evaluation=False):
    from .alignment import AlignmentModel
    from .end_to_end import EndToEndModel

    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    with open_dict(cfg):
        cfg.llm.device = str(device)
        cfg.decoder.model_cfg.device = str(device)
        if evaluation:
            cfg.decoder.training = False
            cfg.decoder.loss = None
    llm = instantiate(cfg.llm)
    if str(device).startswith("cuda"):
        import torch
        llm.to(torch.bfloat16)
        if alignment and llm.rms_norm_layer is not None:
            # As in AFUN's Stage 1, the MetaQuery RMSNorm trains in fp32. Stages 2-3 kept it in
            # bf16, where Adam updates near 1.0 round away, so it stays fixed there.
            llm.rms_norm_layer.float()
    cfg.decoder.model_cfg.projector_in_dim = llm.llm_hidden_size
    if cfg.decoder.get("motion_processor") is not None:
        cfg.decoder.motion_processor.model_cfg.kv_dim = llm.llm_hidden_size
    decoder = instantiate(cfg.decoder, _recursive_=False)
    model_class = AlignmentModel if alignment else EndToEndModel
    return model_class(encoder=None, llm=llm, decoder=decoder, rms_norm=cfg.get("rms_norm", False))
