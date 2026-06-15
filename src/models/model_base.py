# models package
import torch.nn as nn

class ModelBase(nn.Module):
    def __init__(self, freeze: bool = False):
        super().__init__()
        self._frozen = freeze

    def apply_freeze(self):
        for p in self.parameters():
            p.requires_grad = not self._frozen

    def set_freeze(self, freeze: bool):
        self._frozen = freeze
        self.apply_freeze()
        
        