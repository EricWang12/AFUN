# src/models/llm/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
import torch
from torch import nn

from ..model_base import ModelBase
from transformers import PretrainedConfig, PreTrainedModel, AutoProcessor


class LLMBase(ModelBase, ABC):
    
    model_name: str
    llm_model: PreTrainedModel
    processor: AutoProcessor
    
    llm_hidden_size: int
    
    def __init__(self, freeze: bool = False):
        super().__init__(freeze)
        self.apply_freeze()

    @abstractmethod
    def load_model(self, ckpt_path: str = None, device: str = "cuda", **kwargs):
        raise NotImplementedError("Subclasses must implement load_model")
    
    @abstractmethod
    def process_mllm_metaquery(self):
        pass
    

    def forward(self, token_ids: torch.Tensor, attn_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.run_llm(token_ids, attn_mask, **kwargs) 