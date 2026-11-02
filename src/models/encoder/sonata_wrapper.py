# src/models/encoder/sonata_wrapper.py
from __future__ import annotations
import torch
from torch import nn
import numpy as np

from .base import EncoderBase
from .sonata import sonata
try:
    import flash_attn
except ImportError:
    flash_attn = None

def fourier_encode_vector(vec, num_bands=10, sample_rate=60):
    """Fourier encode a vector.

    Args:
        vec: [N, D] torch.FloatTensor.
        num_bands: int.
        sample_rate: int.

    Returns:
        [N, (2 * num_bands + 1) * D] torch.FloatTensor.
    """
    n, d = vec.shape
    samples = torch.linspace(1, sample_rate / 2, num_bands).to(vec.device) * torch.pi
    sines = torch.sin(samples[None, :, None] * vec[:, None, :])
    cosines = torch.cos(samples[None, :, None] * vec[:, None, :])

    encoding = torch.stack([sines, cosines], dim=2).reshape(n, 2 * num_bands, d)
    encoding = torch.cat([vec[:, None, :], encoding], dim=1)
    return encoding.flatten(1)


    

def format_point(coords, rgb, normal, use_numpy=False):
    """coords,rgb,normal: (B,N,3) (numpy or torch). Optional lengths: per-cloud valid sizes."""
    
    if coords.ndim == 2:
        assert rgb.ndim == 2 and normal.ndim == 2
        assert coords.shape == rgb.shape == normal.shape
        assert coords.shape[0] == rgb.shape[0] == normal.shape[0]
        assert coords.shape[1] == rgb.shape[1] == normal.shape[1] == 3
        return {"coord": coords, "color": rgb, "normal": normal}
    
    assert coords.shape == rgb.shape == normal.shape and coords.dim() == 3 and coords.size(-1) == 3
    B, N, _ = coords.shape
    device = coords.device


    coord  = coords.reshape(-1, 3)
    color  = rgb.reshape(-1, 3)
    nrm    = normal.reshape(-1, 3)

    batch  = torch.repeat_interleave(torch.arange(B, device=device), N)  # (sum(lengths),)
    
    if use_numpy:
        coord = coord.cpu().numpy()
        color = color.cpu().numpy()
        nrm = nrm.cpu().numpy()
        batch = batch.cpu().numpy()

    return {"coord": coord, "color": color, "normal": nrm,  "batch": batch}

def format_point2(transform, point: torch.Tensor):

    coords = point[:, :, :3]    # (B, N, 3)
    rgb = point[:, :, 3:6]      # (B, N, 3)
    normal = point[:, :, 6:9]   # (B, N, 3)
    
    
    point = [ transform(format_point(coords[i:i+1], rgb[i:i+1], normal[i:i+1], use_numpy=False)) for i in range(point.shape[0])]
    point = sonata.data.collate_fn(point)
    return point


class SonataWrapper(EncoderBase):
    def __init__(self, sonata_cfg, ckpt_path: str = "facebook/sonata", output_dim: int = 256, freeze: bool = False):
        super().__init__(freeze)

        self.load_model(ckpt_path)
        
        encs_stages = list(self.sonata.enc.children())
        
        self.upcast_levels = max(0, min(sonata_cfg.upcast_levels, len(encs_stages)-1))
        self.use_numpy = sonata_cfg.use_numpy if hasattr(sonata_cfg, "use_numpy") else False
        if self.use_numpy:
            self.transform = sonata.transform.default()
        else:
            self.transform = sonata.transform.default_torch()

        self.output_dim = sum([list(encs_stages[-i-1].children())[-1].channels for i in range(self.upcast_levels+1)])
        
        self.enable_fourier_encode = sonata_cfg.enable_fourier_encode
        downconvs = len(encs_stages) - 1
        res_reduction = 2**downconvs  # voxel resolution reduction
        self.reduced_grid_size = int(sonata_cfg.num_bins / res_reduction)
        self.input_proj = (
            nn.Linear(self.output_dim + 63, self.output_dim)
            if self.enable_fourier_encode
            else None
        ) 
        
        
        self.apply_freeze()


    def load_model(self, ckpt_path: str = None, device: str = "cuda", **kwargs):
        if flash_attn is not None:
            self.sonata = sonata.load("sonata", repo_id=ckpt_path).to(device, non_blocking=True)
        else:
            custom_config = dict(
                enc_patch_size=[1024 for _ in range(5)],  # reduce patch size if necessary
                enable_flash=False,
            )
            self.sonata = sonata.load(
                "sonata", repo_id=ckpt_path, custom_config=custom_config
            ).to(device, non_blocking=True)

    def normalize_pc(self, pc : np.ndarray | torch.Tensor):
        pass

    def forward(self, point: torch.Tensor):
        
        # point is Nx9: [x, y, z, r, g, b, nx, ny, nz]
        # Decompose into separate tensors
        # point: (B, N, 9)
        point = format_point2(self.transform, point)
        
        for key in point.keys():
            if isinstance(point[key], torch.Tensor):
                point[key] = point[key].cuda(non_blocking=True)
        point = self.sonata(point)
        for _ in range(self.upcast_levels ):
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent

        context = point.feat
            
            
            
        if self.enable_fourier_encode:
            coords = point["grid_coord"]
            coords_normalised = coords / (self.reduced_grid_size - 1)
            encoded_coords = fourier_encode_vector(coords_normalised)

            context = torch.cat([context, encoded_coords], dim=-1)
            context = self.input_proj(context)
            
            
        B = int(point['batch'].amax().item() + 1)
        context_list = [context[point['batch'] == b] for b in range(B)]
            
        return context_list
            
