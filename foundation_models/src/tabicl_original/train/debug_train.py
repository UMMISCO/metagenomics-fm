from __future__ import annotations
import sys
sys.path.append('/data/projects/deepintegromics/analyses/3.tabpfn/tab_icl/tabicl/src')

import os
import timeit
import warnings
import functools
from contextlib import nullcontext

import math
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.multiprocessing import set_start_method
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from tqdm import tqdm
import wandb
import pdb

from tabicl_original import TabICL
from tabicl_original.prior.dataset import PriorDataset
from tabicl_original.prior.genload import LoadPriorDataset
from tabicl_original.train.optim import get_scheduler
from tabicl_original.train.train_config import build_parser

warnings.filterwarnings(
    "ignore", message=".*The PyTorch API of nested tensors is in prototype stage.*", category=UserWarning
)

#%%
# ours: /data/projects/deepintegromics/analyses/3.tabpfn/tab_icl/tabicl/src/tabicl/prior/tabpfn_prior/generated_data
# theirs: /data/projects/deepintegromics/analyses/3.tabpfn/tab_icl/tabicl/src/tabicl/
dataset = LoadPriorDataset(
    data_dir="/data/projects/deepintegromics/analyses/3.tabpfn/tab_icl/tabicl/src/tabicl/prior/tabpfn_prior/generated_data",
    batch_size=16,
    ddp_world_size=1,
    ddp_rank=0,
    start_from=0,
    delete_after_load=False,
    device="cuda",
)

dataloader = DataLoader(
    dataset,
    batch_size=None,  # No additional batching since PriorDataset handles batching internally
    shuffle=False,
    num_workers=1,
    prefetch_factor=4,
    pin_memory=False,
    pin_memory_device="",
)
for batch in dataloader:
    break

"""
Theirs:
{'X': torch.Size([26227712]),
 'd': torch.Size([512]),
 'seq_lens': torch.Size([512]),
 'seq_lens_0': tensor(1024, device='cuda:0'),
 'batch_size': 512,
 'd_repeat': tensor([13, 13, 13,  ..., 88, 88, 88], device='cuda:0'),
 'X post sparse2dense': torch.Size([512, 1024, 100])}
 
 
Ours:
{'micro_batch_idx': 0, 'micro_X': torch.Size([4, 1024, 3]), 'y': torch.Size([4, 420]), 'micro_d shape': torch.Size([4]), 'micro_d': tensor([3, 3, 3, 3], device='cuda:0')}
{'micro_batch_idx': 1, 'micro_X': torch.Size([4, 1024, 3]), 'y': torch.Size([4, 420]), 'micro_d shape': torch.Size([4]), 'micro_d': tensor([3, 3, 3, 3], device='cuda:0')}
{'micro_batch_idx': 2, 'micro_X': torch.Size([4, 1024, 3]), 'y': torch.Size([4, 420]), 'micro_d shape': torch.Size([4]), 'micro_d': tensor([3, 3, 3, 3], device='cuda:0')}
{'micro_batch_idx': 3, 'micro_X': torch.Size([4, 1024, 3]), 'y': torch.Size([4, 420]), 'micro_d shape': torch.Size([4]), 'micro_d': tensor([3, 3, 3, 3], device='cuda:0')}
"""

#%%
#OLD One_hot_encoder class
class OneHotAndLinear(nn.Linear):
    """Combines one-hot encoding and linear projection in a single efficient operation
    to convert categorical indices to embeddings.

    Parameters
    ----------
    num_classes : int
        Number of distinct categories for one-hot encoding

    embed_dim : int
        Output embedding dimension
    """

    def __init__(self, num_classes: int, embed_dim: int):
        super().__init__(num_classes, embed_dim)
        self.num_classes = num_classes
        self.embed_dim = embed_dim

    def forward(self, src: Tensor) -> Tensor:
        """Transform integer indices to dense embeddings.

        Parameters
        ----------
        src : Tensor
            Integer tensor of shape (batch_size, sequence_length) containing category indices

        Returns
        -------
        Tensor
            Embedded representation of shape (batch_size, sequence_length, embed_dim)
        """

        # Convert indices to one-hot vectors and apply linear projection
        one_hot = F.one_hot(src.long(), self.num_classes).to(src.dtype)
        return F.linear(one_hot, self.weight, self.bias)
