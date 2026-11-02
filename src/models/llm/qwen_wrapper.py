"""Qwen visual-language features with trainable MetaQuery tokens."""
from __future__ import annotations
from typing import List
import torch
from torch import nn
import torch.nn.functional as F

from PIL import Image
from transformers import PreTrainedModel, PretrainedConfig
from .base import LLMBase




from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModelForImageTextToText



class QwenWrapperConfig(PretrainedConfig):
    model_type = "qwen_wrapper"
    system_prompt: str = None
    max_input_text_tokens: int = 256
    rms_norm: bool = True
    skip_lm_head: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.system_prompt = kwargs.get("system_prompt", None)
        self.max_input_text_tokens = kwargs.get("max_input_text_tokens", 256)
        self.skip_lm_head = kwargs.get("skip_lm_head", False)

class QwenWrapper(PreTrainedModel, LLMBase):    
    
    config_class = QwenWrapperConfig
    """Encode image instructions and extract learned MetaQuery features."""
    num_metaqueries: int = 0
    def __init__(self, model_name="Qwen/Qwen-7B",
                 llm_cfg: dict | None = None,
                 ft_cfg: dict | None = None,
                 freeze: bool = False,
                 device: str = "cuda",
                 ):
        self.llm_cfg = QwenWrapperConfig(**(llm_cfg or {}))
        PreTrainedModel.__init__(self, self.llm_cfg)   # registers config etc.
        LLMBase.__init__(self, freeze=freeze)

        self.model_name = model_name
        self.freeze = freeze
        # HF weights use the default cache (~/.cache/huggingface, or $HF_HOME if set).
        self._cache_dir = None


        from omegaconf import OmegaConf
        self.metaquery_cfg = OmegaConf.create(ft_cfg.get("metaquery_cfg")) if ft_cfg and ft_cfg.get("metaquery_cfg") else None
        if self.metaquery_cfg and "metaquery_types" in self.metaquery_cfg:
            self.metaquery_cfg["num_metaqueries"] *= len(self.metaquery_cfg["metaquery_types"])


        self.num_point_tokens = ft_cfg.get("num_point_tokens", 0) if ft_cfg and hasattr(ft_cfg, 'num_point_tokens') else 0



        self.load_model(device=device)
        
    def _get_embed_tokens(self):
        """Find embed_tokens across different Qwen model structures."""
        # Qwen3-VL: model.model.embed_tokens
        if hasattr(self.llm_model.model, "embed_tokens"):
            return self.llm_model.model.embed_tokens
        # Qwen3.5: model.model.language_model.embed_tokens
        if hasattr(self.llm_model.model, "language_model") and hasattr(self.llm_model.model.language_model, "embed_tokens"):
            return self.llm_model.model.language_model.embed_tokens
        # Qwen2.5-VL: model.language_model.embed_tokens
        if hasattr(self.llm_model, "language_model") and hasattr(self.llm_model.language_model, "embed_tokens"):
            return self.llm_model.language_model.embed_tokens
        raise AttributeError(f"Cannot find embed_tokens in model {type(self.llm_model).__name__}")

    def _interleave_metaquery_types(self, metaquery_prompt_embeds, metaquery_attention_mask):
        if self.metaquery_cfg is not None and "metaquery_types" in self.metaquery_cfg:
            metaquery_types = self.metaquery_cfg["metaquery_types"]
            num_types = len(metaquery_types)
            num_metaqueries_per_type = self.num_metaqueries // num_types if num_types > 0 else  self.num_metaqueries 
            embeds = {}
            masks = {}
            for idx, t in enumerate(metaquery_types):
                start = idx * num_metaqueries_per_type
                end = start + num_metaqueries_per_type
                embeds[t] = metaquery_prompt_embeds[:, start:end, :]    # (B, num_metaqueries_per_type, H)
                masks[t] = metaquery_attention_mask[:, start:end]       # (B, num_metaqueries_per_type)
            return embeds, masks
        else:
            return metaquery_prompt_embeds, metaquery_attention_mask
        

    def process_mllm_metaquery(self):
        if not self.metaquery_cfg or "num_metaqueries" not in self.metaquery_cfg or self.metaquery_cfg["num_metaqueries"] <= 0:
            return
        
        self.num_metaqueries = self.metaquery_cfg["num_metaqueries"]
        num_embeddings = self.llm_model.get_input_embeddings().num_embeddings
        self.num_embeddings = num_embeddings
        
        
        num_mq_extra_tokens = self.num_metaqueries + 2 if self.num_metaqueries > 0 else 0 # begin_of_img, end_of_img
        num_pc_extra_tokens = 3  # Reserved IDs retained for checkpoint compatibility.
        
        num_extra_tokens = num_mq_extra_tokens + num_pc_extra_tokens

        if self.num_metaqueries > 0:
            self.llm_model.resize_token_embeddings(
                num_embeddings + num_extra_tokens, mean_resizing=False
            )


        def freeze_hook(grad: torch.Tensor) -> torch.Tensor:
            # build a *new* mask tensor the same size as the grad
            mask = torch.zeros_like(grad)
            mask[-num_mq_extra_tokens:, :] = 1          # keep only these rows
            return grad * mask                          # never modify grad in-place
        

        embed_tokens = self._get_embed_tokens()
        # Store handle so freeze_segmentation_metaqueries() can swap the hook later.
        self._embed_freeze_handle = embed_tokens.weight.register_hook(freeze_hook)
            
        try:
            self.llm_hidden_size = self.llm_model.config.hidden_size
        except AttributeError:
            self.llm_hidden_size = self.llm_model.config.text_config.hidden_size
        
        self.processor.num_metaqueries = self.num_metaqueries
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    f"<pad_token_{i}>"
                    for i in range(num_embeddings - len(tokenizer))
                ]
            }
        )
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|begin_of_point|>", "<|pad_point|>", "<|end_of_point|>"]})
        self.point_start_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_point|>")
        self.point_pad_token_id = tokenizer.convert_tokens_to_ids("<|pad_point|>")
        self.point_end_token_id = tokenizer.convert_tokens_to_ids("<|end_of_point|>")
        
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": ["<begin_of_mq>", "<end_of_mq>"]
                + [f"<mq{i}>" for i in range(self.num_metaqueries)]
            }
        )
        self.boi_token_id = tokenizer.convert_tokens_to_ids("<begin_of_mq>")
        self.eoi_token_id = tokenizer.convert_tokens_to_ids("<end_of_mq>")


    def _apply_metaquery_rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize jointly, or per query type when segmentation is frozen."""
        if not getattr(self, "_split_rms_per_type", False):
            return self.rms_norm_layer(x)

        types = (
            list(self.metaquery_cfg.get("metaquery_types", []))
            if self.metaquery_cfg is not None else []
        )
        if len(types) <= 1:
            return self.rms_norm_layer(x)

        num_per_type = self.num_metaqueries // len(types)
        H = x.shape[-1]
        weight = self.rms_norm_layer.weight  # (num_metaqueries, H)
        eps = self.rms_norm_layer.eps
        normed = []
        for i in range(len(types)):
            s = i * num_per_type
            e = s + num_per_type
            normed.append(F.rms_norm(x[:, s:e, :], (num_per_type, H), weight[s:e, :], eps))
        return torch.cat(normed, dim=1)

    def freeze_segmentation_metaqueries(self) -> None:
        """Train only motion queries and isolate their RMS normalization."""
        if self.num_metaqueries <= 0:
            return

        metaquery_types = (
            list(self.metaquery_cfg.get("metaquery_types", []))
            if self.metaquery_cfg is not None else []
        )
        if "motion" not in metaquery_types:
            raise ValueError(
                "freeze_segmentation requires 'motion' in metaquery_cfg.metaquery_types; "
                f"got {metaquery_types}"
            )

        num_per_type = self.num_metaqueries // len(metaquery_types)
        motion_idx = metaquery_types.index("motion")
        # Rows for type at position i live at offsets [i*num_per_type, (i+1)*num_per_type)
        # within the trailing block of self.num_metaqueries rows in embed_tokens.weight.
        start = -(self.num_metaqueries - motion_idx * num_per_type)
        end = -(self.num_metaqueries - (motion_idx + 1) * num_per_type)

        def freeze_hook(grad: torch.Tensor) -> torch.Tensor:
            mask = torch.zeros_like(grad)
            if end == 0:
                mask[start:, :] = 1
            else:
                mask[start:end, :] = 1
            return grad * mask

        # Swap the existing hook for the tighter one.
        if getattr(self, "_embed_freeze_handle", None) is not None:
            self._embed_freeze_handle.remove()
        embed_tokens = self._get_embed_tokens()
        self._embed_freeze_handle = embed_tokens.weight.register_hook(freeze_hook)

        # rms_norm_layer.weight has shape (num_metaqueries, H). With per-type
        # application (set below), each type uses ONLY its slice of weight, so
        # only the semantic rows would affect segmentation. Mask grads on the
        # non-motion rows so motion can still tune its half — same pattern as
        # the embed_tokens hook above.
        if self.rms_norm_layer is not None:
            rms_motion_start = motion_idx * num_per_type
            rms_motion_end = rms_motion_start + num_per_type

            def rms_freeze_hook(grad: torch.Tensor) -> torch.Tensor:
                mask = torch.zeros_like(grad)
                mask[rms_motion_start:rms_motion_end, :] = 1
                return grad * mask

            if getattr(self, "_rms_freeze_handle", None) is not None:
                self._rms_freeze_handle.remove()
            self._rms_freeze_handle = self.rms_norm_layer.weight.register_hook(rms_freeze_hook)

        self._split_rms_per_type = True

    def tokenize(self, caption: str|List[str], pc_embeddings: torch.Tensor = None, image: List[List[Image.Image]] | List[List[torch.Tensor]] = None, text_response: List[str] = None, add_generation_prompt: bool = True) -> torch.Tensor:
        if isinstance(caption, str):
            caption = [caption]

        # if disable image, set image to None
        if hasattr(self.llm_cfg, "enable_image") and not self.llm_cfg.enable_image:
            image = None
    

        if  self.num_metaqueries <= 0:
            suffix = ""
        else:
            suffix = (
                "\n<begin_of_mq>"
                + "".join([f"<mq{i}>" for i in range(self.num_metaqueries)])
                + "<end_of_mq><|im_end|>"
            )
        if self.num_point_tokens > 0:
            pc_prefix = "<|begin_of_point|>" + "<|pad_point|>" * self.num_point_tokens + "<|end_of_point|>"

        prefix = (
            [
                {
                    "role": "system",
                    "content": (
                             [{"type": "text", "text": self.processor.system_prompt}]
                    ),
                },
            ]
            if self.processor.system_prompt is not None
            else []
        )

        caption = [
            self.processor.decode(
                self.processor(text=cap, return_tensors="pt", padding=False).input_ids[
                    0, : self.processor.max_input_text_tokens
                ]
            )
            for cap in caption
        ]
        if image is not None:
            # If image is not a list, wrap it in a list
            if isinstance(image, torch.Tensor):
                image = list(image.unbind(dim=0)) 
            elif not isinstance(image, list):
                image = [image]
                
            # If each batch item is not a list, wrap it in a single-element list (or empty list if None)
            for i, img in enumerate(image):
                if img is not None and not isinstance(img, list):
                    image[i] = [img]

            # Resize each image in each batch if resize_fn is not None
            if hasattr(self.processor, "resize_fn") and self.processor.resize_fn is not None:
                image = [
                    [self.processor.resize_fn(sub_img) for sub_img in imgs] if imgs else None
                    for imgs in image
                ]

            conversations = [
                prefix
                + [
                    {
                        "role": "user",
                        "content": (
                            [{"type": "image", "image": img} for img in imgs]
                            + [{"type": "text", "text": cap}]
                            if imgs
                            else [{"type": "text", "text": cap}]
                        ),
                    },
                ]
                for cap, imgs in zip(caption, image)
            ]
            kwargs = {"images": [imgs for imgs in image if imgs]}
        else:
            conversations = [
                prefix
                + [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": cap}],
                    },
                ]
                for cap in caption
            ]
            kwargs = dict()


        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Kwargs passed to.*processor_kwargs")
            prompts = [
                self.processor.apply_chat_template(conv, add_generation_prompt=add_generation_prompt)
                for conv in conversations
            ]
            if text_response is not None:
                prompts = [p + t.strip() for p, t in zip(prompts, text_response)]
            if self.num_point_tokens > 0:
                prompts = [p + pc_prefix for p in prompts]

            if self.num_metaqueries > 0:
                prompts = [p + suffix for p in prompts]

            text_inputs = self.processor(
                text=prompts,
                return_tensors="pt",
                padding=True,
                **kwargs,
            ).to(self.llm_model.device)
        
        if   "pixel_values" in text_inputs:
            text_inputs["pixel_values"] = text_inputs["pixel_values"].unsqueeze(0)
        return text_inputs


        
    def load_model(self, ckpt_path: str = None, device: str = "cuda", **kwargs):
        """
        Load the Qwen model and tokenizer.
        
        Args:
            ckpt_path: Path to checkpoint (if None, uses model_name)
            device: Device to load model on
            **kwargs: Additional arguments for model loading
        """

        # Use checkpoint path if provided, otherwise use model_name
        model_path = ckpt_path if ckpt_path is not None else self.model_name

                
        # Load model
        
        if "Qwen3" in self.model_name:
            self.llm_model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                dtype="auto",
                device_map=device,
                cache_dir=self._cache_dir
            )
            cfg = self.llm_model.config
            self.llm_hidden_size = cfg.text_config.hidden_size if hasattr(cfg, 'text_config') else cfg.hidden_size
        else:
            self.llm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map=device,
                attn_implementation="sdpa",
                cache_dir=self._cache_dir
            )
            self.llm_hidden_size = self.llm_model.config.hidden_size
        # Load tokenizer/processor
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            cache_dir=self._cache_dir
        )
        
        self.process_mllm_metaquery()
                
        # Resolve actual device for .to() calls ("auto" is only valid for device_map)
        _to_device = "cuda" if device == "auto" else device
        if hasattr(self.llm_cfg, 'rms_norm') and self.llm_cfg.rms_norm:
            self.rms_norm_layer = nn.RMSNorm((self.num_metaqueries, self.llm_hidden_size), eps=1e-5).to(_to_device)
        else:
            self.rms_norm_layer = None

        self.processor.max_input_text_tokens = self.llm_cfg.max_input_text_tokens

        self.processor.system_prompt = self.llm_cfg.system_prompt

        self.pad_token_id = getattr(
            self.processor, "tokenizer", self.processor
        ).pad_token_id

        # Move to device if not using device_map
        if device not in ("auto",) and (device != "cuda" or self.llm_model.device.type != device):
            self.llm_model = self.llm_model.to(device)


        # --- skip_lm_head: replace lm_head with Identity at init ---
        # Uses a norm hook in forward() to capture the last hidden state
        # instead of output_hidden_states=True (which stores all 32 layers).
        # Saves ~2 GB (lm_head weights) + ~10-20 GB (logits tensor).
        self._final_norm = None
        if getattr(self.llm_cfg, 'skip_lm_head', False) and self.num_metaqueries > 0:
            for attr_path in [
                "model.language_model.model.norm",   # Qwen3.5
                "model.language_model.norm",          # Qwen2.5-VL
                "model.model.norm",                   # Qwen3-VL
                "model.norm",                         # Qwen3 text-only (CausalLM)
            ]:
                obj = self.llm_model
                try:
                    for part in attr_path.split("."):
                        obj = getattr(obj, part)
                    self._final_norm = obj
                    break
                except AttributeError:
                    continue

            if self._final_norm is not None and hasattr(self.llm_model, 'lm_head'):
                self.llm_model.lm_head = nn.Identity()
                
    def forward(self, input_ids, attention_mask,  pixel_values, image_sizes=None, point_cloud_embeddings = None, image_grid_thw = None, mm_token_type_ids = None, past_key_values=None, **kwargs):

        kwargs.pop("input_ids",        None)
        kwargs.pop("attention_mask",   None)
        kwargs.pop("pixel_values",     None)
        kwargs.pop("image_sizes",      None)

        if point_cloud_embeddings is not None:
            raise ValueError("AFUN accepts RGB images; point-cloud prompt tokens are unsupported.")

        # If skip_lm_head was set at init, _final_norm is the cached norm layer
        # and lm_head was replaced with Identity. Register a hook each forward
        # to capture the last hidden state without output_hidden_states=True.
        _skip_lm_head = self._final_norm is not None
        _captured_hidden = [None]
        hook_handle = None

        if _skip_lm_head:
            def _capture_last_hidden(module, input, output):
                _captured_hidden[0] = output[0] if isinstance(output, tuple) else output
            hook_handle = self._final_norm.register_forward_hook(_capture_last_hidden)

        # Build kwargs for the underlying model call
        kwargs.pop("skip_logits", None)
        llm_kwargs = dict(
            input_ids=input_ids, attention_mask=attention_mask,
            pixel_values=pixel_values, image_grid_thw=image_grid_thw,
            output_hidden_states=(not _skip_lm_head),
            use_cache=False,
        )
        if mm_token_type_ids is not None:
            llm_kwargs["mm_token_type_ids"] = mm_token_type_ids
        llm_kwargs.update(kwargs)
        try:
            outputs = self.llm_model(**llm_kwargs)
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        batch_size, seq_len = input_ids.shape
        indices = torch.arange(seq_len, device=input_ids.device)[None, :].expand(batch_size, -1)

        if self.num_metaqueries > 0:
            # Get last hidden state: from hook (skip_lm_head) or hidden_states (default)
            if _captured_hidden[0] is not None:
                metaquery_prompt_embeds = _captured_hidden[0]
            else:
                metaquery_prompt_embeds = outputs.hidden_states[-1]
            # Get positions for all sequences in batch at once
            boi_pos = torch.where(input_ids == self.boi_token_id)[1]
            eoi_pos = torch.where(input_ids == self.eoi_token_id)[1]

            # Create mask for selecting tokens between BOI and EOI
            batch_size, seq_len = input_ids.shape
            indices = torch.arange(seq_len, device=input_ids.device)[None, :].expand(
                batch_size, -1
            )
            mask = (indices > boi_pos[:, None]) & (indices < eoi_pos[:, None])

            metaquery_prompt_embeds = metaquery_prompt_embeds[mask].view(
                batch_size, -1, metaquery_prompt_embeds.size(-1)
            )
            metaquery_attention_mask = attention_mask[mask].view(batch_size, -1)
            
            outputs["metaquery_prompt_embeds"] = metaquery_prompt_embeds
            if self.rms_norm_layer is not None:
                outputs["metaquery_prompt_embeds"] = self._apply_metaquery_rms_norm(
                    outputs["metaquery_prompt_embeds"]
                )

            outputs["metaquery_attention_mask"] = metaquery_attention_mask

            outputs["metaquery_prompt_embeds"], outputs["metaquery_attention_mask"] = \
                self._interleave_metaquery_types(outputs["metaquery_prompt_embeds"], outputs["metaquery_attention_mask"])
            
            inverse_mask = (indices < boi_pos[:, None]) | (indices > eoi_pos[:, None])

            # Apply the inverse mask to logits
            outputs.logits = outputs.logits[inverse_mask].view(batch_size, -1, outputs.logits.size(-1))


        # Convert to plain dict so DDP pytree doesn't try to reconstruct
        # the HF ModelOutput with custom keys (metaquery_prompt_embeds etc.)
        return dict(outputs)
        
