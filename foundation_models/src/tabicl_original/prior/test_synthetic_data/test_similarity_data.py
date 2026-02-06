#%%
import numpy as np
from openTSNE import TSNE
import matplotlib.pyplot as plt
import torch
import pandas as pd

#%%

X = torch.load("/data/projects/deepintegromics/analyses/3.tabpfn/tab_icl/tabicl/src/tabicl_original/checkpoints/checkpoints_tabicl_similarity/micro_X.pt")
X_cmp = torch.load("/data/projects/deepintegromics/analyses/3.tabpfn/tab_icl/tabicl/src/tabicl_original/checkpoints/checkpoints_tabicl_similarity/micro_X_cmp.pt")

#%%

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.hist(X[0][0].cpu().numpy())
plt.show()

#%%
def clr_transform(X, axis=-1, eps=1e-8):
    """
    Apply centered log-ratio (CLR) transform to compositional data per sample.

    Parameters
    ----------
    X : Tensor (B, T, H)
        Input tensor (batch, sequence length, features)
    axis : int
        Axis along which features are compositional (typically last axis)
    eps : float
        Small constant to avoid log(0)

    Returns
    -------
    Tensor
        CLR-transformed tensor
    """

    # Make X positive (smooth, stable)
    X_copy = X.clone()
    X_pos = F.softplus(X_copy) + eps

    # Convert to a composition
    X_comp = X_pos / X_pos.sum(dim=axis, keepdim=True)

    # Apply CLR
    X_safe = X_comp + eps
    logX = torch.log(X_safe)

    return logX - logX.mean(dim=axis, keepdim=True)
