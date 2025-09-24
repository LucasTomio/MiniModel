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
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from functools import partial
from collections import Counter
from typing import Dict, List, Optional, Tuple, Callable, Union

os.environ['TORCH_USE_CUDA_DSA'] = '1'

from dev_optim  import AdaMuon, CosineAnnealingWarmup
from data_utils import load_parquet
from fast_self_attn_model import Transformer as Model

from torchao.float8 import convert_to_float8_training

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

torch.set_default_device('cuda:0')
torch.backends.cuda.matmul.allow_tf32 = True
torch._dynamo.config.capture_scalar_outputs = True

config = {
    'layers': 24,
    'num_heads': 12,
    'vocab_size': 32768,
    'input_dims': 768,
    'hidden_dims': None, 
    'dtype': torch.bfloat16,
}

batch_size = 64
seq_length = 2048
grad_steps = 1

base_lr      = 1e-3
init_lr      = base_lr/10
anneal_lr    = base_lr/10

warmup_steps = 100
anneal_steps = 108_000

model = Model(**config)
model.zero_grad()
model.bfloat16()

model = convert_to_float8_training(model)

param_shapes = set([p.shape for p in model.parameters()])

optimizer = AdaMuon([{'params': [p for p in model.parameters() if (p.shape == shape)]} for shape in param_shapes], 
                   lr=base_lr, weight_decay=1e-1)

scheduler = CosineAnnealingWarmup(optimizer, init_lr, base_lr, anneal_lr, warmup_steps, anneal_steps)

@torch.compile(mode='max-autotune-no-cudagraphs')
def loss_fn(model, X):

    Y = torch.roll(X, -1, dims=1)
    Y[:, -1] *= 0
    
    loss = model(X, Y)

    return loss

# Training Loop

cur_step = 0

def checkpoint(cur_step, loss, save_states, config):
    model, optimizer = save_states
    
    dataset_name = 'TinyCorpus'
    date = time.strftime("%b_%d_%Y", time.gmtime())
    file_name = f'{dataset_name}_{cur_step}_{loss}_{date}'

    checkpoint = {
        'model': model.state_dict(), 
        'optimizer': optimizer.state_dict(), 
    }
    
    torch.save(checkpoint, f'{file_name}_checkpoint.pt')
    torch.save(model.state_dict(), f'{file_name}_weights.pt')

for dataset_n in range(48):

    torch.cuda.empty_cache()
    gc.collect()
    
    # Load Data
    
    base_path = "datasets/TinyCorpus/128/"
    data_name = f"tinycorpus-{dataset_n%480:03d}-of-128.parquet"

    loader, n_rows = load_parquet(base_path, data_name, batch_size=batch_size, columns=['0'])

    print(f'Loaded {data_name}')
    
    pbar = tqdm(total=n_rows*seq_length, unit="tokens")

    torch.cuda.synchronize()
    
    # Init
    model.train()

    torch.cuda.empty_cache()
    
    for mini_batch, data in enumerate(loader):
        
        tok_batch = torch.tensor(data['0'], device='cuda:0')

        loss = loss_fn(model, tok_batch)

        if torch.isnan(loss).item():
            states = (model, optimizer)
            checkpoint(cur_step, 'NaN', states, config)
            assert False, 'NaNs encountered during training'
             
        loss.backward()
        loss = loss.item()

        optimizer.step()
        scheduler.step()

        model.zero_grad(set_to_none=True)
        
        writer.add_scalar("Loss/train", loss, cur_step)
        
        torch.cuda.reset_peak_memory_stats(device='cuda')

        pbar.update(batch_size*seq_length)
        
        cur_step += 1

    loader.cleanup()
    gc.collect()
    del loader

    writer.flush()
    states = (model, optimizer)
    checkpoint(cur_step, loss, states, config)
    
    model.eval()