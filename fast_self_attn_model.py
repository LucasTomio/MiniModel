import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.cpp_extension import load
import gc
import os
import time
import math
import numpy as np
from functools import partial
from collections import Counter
from typing import Dict, List, Optional, Tuple, Callable, Union

from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

eps = torch.finfo(torch.float32).eps

from liger_kernel.transformers.rms_norm import LigerRMSNorm

def norm(x: torch.Tensor):
    return torch.rms_norm(x, (x.size(-1),), eps=eps)

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: torch.Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSoftmaxAttention(nn.Module):
    def __init__(
        self, 
        layer_id: int, 
        layers: int, 
        num_heads: int, 
        vocab_size: int,
        input_dims: int, 
        hidden_dims: Union[int, None] = None, 
    ):
        super().__init__()
        
        self.layer_id = layer_id
        self.head_dim = input_dims // num_heads
        self.num_heads = num_heads
        assert input_dims % self.num_heads == 0

        H = self.num_heads
        N = self.head_dim
        C = input_dims

        with torch.no_grad():
            init_bounds = 0.5 / (C ** 0.5)
    
            self.q_proj = nn.Linear(C, C, bias=False)
            self.k_proj = nn.Linear(C, C, bias=False)
            self.v_proj = nn.Linear(C, C, bias=False)
            self.g_proj = nn.Linear(C, C, bias=False)
            self.o_proj = nn.Linear(C, C, bias=False)

            self.rotary = Rotary(N, 2048)
    
            self.q_proj.weight.data.uniform_(-init_bounds, init_bounds)
            self.k_proj.weight.data.uniform_(-init_bounds, init_bounds)
            self.v_proj.weight.data.uniform_(-init_bounds, init_bounds)
            self.g_proj.weight.data.uniform_(-init_bounds, init_bounds)
            self.o_proj.weight.data.zero_()          

    def forward(self, x):
        B, T, C = x.size()
        H = self.num_heads
        N = C // H

        def forward1(x):
            x = norm(x)
            
            q = self.q_proj(x).view(B, T, H, N)
            k = self.k_proj(x).view(B, T, H, N)
            v = self.v_proj(x).view(B, T, H, N)
            g = self.g_proj(x).sigmoid()

            q, k = norm(q), norm(k)
            q, k = self.rotary(q), self.rotary(k)
            
            return (q, k, v, g)

        (q, k, v, g) = torch.utils.checkpoint.checkpoint(forward1, x, use_reentrant=False)

        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
            x = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True).transpose(1, 2).contiguous().view(B, T, C)

        x = self.o_proj(x * g)
        
        return x
        
class MLP(nn.Module):
    def __init__(
        self, 
        layer_id: int, 
        layers: int, 
        num_heads: int, 
        vocab_size: int, 
        input_dims: int, 
        hidden_dims: Union[int, None] = None, 
    ):
        super().__init__()
        
        self.layer_id = layer_id
        
        C = input_dims
        hidden_dims = hidden_dims or 4 * C

        with torch.no_grad():    
            init_bounds = 0.5 / (C ** 0.5)
            
            self.k_proj = nn.Linear(C, hidden_dims, bias=False)
            self.v_proj = nn.Linear(hidden_dims, C, bias=False)
            
            self.k_proj.weight.data.uniform_(-init_bounds, init_bounds)
            self.v_proj.weight.data.zero_()

    def forward(self, x):
        B, T, C = x.size()

        def forward1(x):
            x = norm(x)
            
            k = torch.relu(self.k_proj(x)).square()
            
            return self.v_proj(k)
        
        output = torch.utils.checkpoint.checkpoint(forward1, x, use_reentrant=False)
        
        return output

class SoftmaxBlock(nn.Module):
    def __init__(
        self, 
        layer_id: int, 
        layers: int, 
        num_heads: int, 
        vocab_size: int, 
        input_dims: int, 
        hidden_dims: Union[int, None] = None, 
    ):
        super().__init__()
        self.layer_id = layer_id

        self.att = CausalSoftmaxAttention(layer_id, layers, num_heads, vocab_size, input_dims, hidden_dims)
        self.ffn = MLP(layer_id, layers, num_heads, vocab_size, input_dims, hidden_dims)
    
    def forward(self, x):
        xx = self.att(x)
        x = x + xx
        
        xx = self.ffn(x)
        x = x + xx
        
        return x

class Transformer(nn.Module):
    def __init__(
        self, 
        layers: int, 
        num_heads: int, 
        vocab_size: int, 
        input_dims: int, 
        hidden_dims: Union[int, None] = None, 
        dtype = None
    ):
        super().__init__()
        
        self.emb = nn.Embedding(vocab_size, input_dims)
        self.emb.weight.data.uniform_(-1e-4, 1e-4)
        
        self.blocks = nn.ModuleList([SoftmaxBlock(i, layers, num_heads, vocab_size, input_dims, hidden_dims) for i in range(layers)])

    def forward(self, idx, tgt=None):
        
        x = norm(self.emb(idx))
        
        for i, block in enumerate(self.blocks):
            x = block(x)

        x = norm(x)

        if self.training:
            loss_fn = LigerFusedLinearCrossEntropyLoss(ignore_index=0, reduction="mean")
            
            return loss_fn(self.emb.weight, x.view(-1, x.shape[-1]), tgt.view(-1))
        else:                
            logits = F.linear(x, self.emb.weight)
            
            return logits