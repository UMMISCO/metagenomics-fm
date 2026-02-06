#%%
import numpy as np
from openTSNE import TSNE
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torch import Tensor
from typing import Optional, List
import types
import pandas as pd

import sys
sys.path.append('/data/projects/deepintegromics/analyses/3.tabpfn/tab_icl/tabicl/src')
from tabicl_original.model.inference_config import InferenceConfig
from tabicl_original.sklearn.classifier import TabICLClassifier

clf = TabICLClassifier(n_estimators=1, norm_methods=["none", "power"],preprocess=True, model_path="/data/projects/deepintegromics/analyses/3.tabpfn/tab_icl/tabicl/src/tabicl/checkpoints/checkpoints_tabicl_original_retrained_no_seed/step-2000.ckpt", allow_auto_download=False)
#%%
X = torch.load('/data/projects/deepintegromics/analyses/3.tabpfn/tab_icl/tabicl/src/tabicl/checkpoints/checkpoints_try/our_batch/batch_000000.pt')
Xo = torch.load('/data/projects/deepintegromics/analyses/3.tabpfn/tab_icl/tabicl/src/tabicl/checkpoints/checkpoints_try/original_batch/batch_000000.pt')

#%%
def sparse2dense(

        sparse_tensor: torch.Tensor,
        row_lengths: torch.Tensor,
        max_len: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reconstruct a dense tensor from its sparse representation.
    This function is the inverse of dense2sparse, reconstructing a padded dense
    tensor from a compact 1D representation and the corresponding row lengths.
    Unused entries in the output are filled with zeros.
    Parameters
    ----------
    sparse_tensor : torch.Tensor
        1D tensor containing the valid entries from the original dense tensor
    row_lengths : torch.Tensor
        Number of valid entries for each row in the output tensor
    max_len : Optional[int], default=None
        Maximum length for each row in the output. If None, uses max(row_lengths)
    dtype : torch.dtype, default=torch.float32
        Output data type for the dense representation
    Returns
    -------
    torch.Tensor
        Dense tensor of shape (num_rows, max_len) with zeros padding
    """
    assert sparse_tensor.dim() == 1, "data must be 1D"
    assert row_lengths.sum() == len(sparse_tensor), "data length must match sum of row_lengths"
    num_rows = len(row_lengths)
    max_len = max_len or row_lengths.max().item()
    dense = torch.zeros(num_rows, max_len, dtype=dtype, device=sparse_tensor.device)
    indices = torch.arange(max_len, device=sparse_tensor.device)
    mask = indices.unsqueeze(0) < row_lengths.unsqueeze(1)
    dense[mask] = sparse_tensor.to(dtype)
    return dense
#%%

assignments = X['feature_assignments']
X_original = X['X']
y = X['y']
batch_size_original = X['batch_size']
seq_lens_original = X['seq_lens']
d_original = X['d']
X = sparse2dense(X_original, d_original.repeat_interleave(seq_lens_original[0]), dtype=torch.float32).view(batch_size_original, seq_lens_original[0], -1)

#%%
#nd, ns, nf = X.shape
#X.view(nd*ns, nf).shape
Xs = X[1]
ys = y[1]
assignment = assignments[1]
split_idx = int(Xs.shape[0] * 0.5)
x_train, x_test = Xs[:split_idx], Xs[split_idx:]
y_train, y_test = ys[:split_idx], ys[split_idx:]

clf.fit(x_train, y_train)

#%%
x_test = clf._validate_data(x_test, reset=False, dtype=None, skip_check_array=True)
x_test = clf.X_encoder_.transform(x_test)
data = clf.ensemble_generator_.transform(x_test)

def model_new_inference_forward(
        self,
        X: Tensor,
        y_train: Tensor,
        feature_shuffles: Optional[List[List[int]]] = None,
        embed_with_test: bool = False,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        inference_config: InferenceConfig = None,
) -> Tensor:

    train_size = y_train.shape[1]
    assert train_size <= X.shape[1], "Number of training samples exceeds total samples"

    if inference_config is None:
        inference_config = InferenceConfig()

    # Column-wise embedding -> Row-wise interaction
    partial_outputs = {}
    col_embeddings = self.col_embedder(
        X,
        train_size=None if embed_with_test else train_size,
        feature_shuffles=feature_shuffles,
        mgr_config=inference_config.COL_CONFIG,
    )
    partial_outputs['col_embeddings'] = col_embeddings
    representations = self.row_interactor(col_embeddings, mgr_config=inference_config.ROW_CONFIG,)
    partial_outputs['representations'] = representations

    # Dataset-wise in-context learning
    out = self.icl_predictor(
        representations,
        y_train=y_train,
        return_logits=return_logits,
        softmax_temperature=softmax_temperature,
        mgr_config=inference_config.ICL_CONFIG,
    )

    return out, partial_outputs


# Update forward method of model
model = clf.model_
model._inference_forward = types.MethodType(model_new_inference_forward, model)
def _batch_forward(self, Xs, ys, shuffle_patterns=None):
    batch_size = self.batch_size or Xs.shape[0]
    n_batches = np.ceil(Xs.shape[0] / batch_size)
    Xs = np.array_split(Xs, n_batches)
    ys = np.array_split(ys, n_batches)
    if shuffle_patterns is None:
        shuffle_patterns = [None] * n_batches
    else:
        shuffle_patterns = np.array_split(shuffle_patterns, n_batches)

    outputs = []
    partial_outputs = []
    for X_batch, y_batch, pattern_batch in zip(Xs, ys, shuffle_patterns):
        X_batch = torch.from_numpy(X_batch).float().to(self.device_)
        y_batch = torch.from_numpy(y_batch).float().to(self.device_)
        if pattern_batch is not None:
            pattern_batch = pattern_batch.tolist()

        with torch.no_grad():
            out, partial_out = self.model_(
                X_batch,
                y_batch,
                feature_shuffles=pattern_batch,
                return_logits=True if self.average_logits else False,
                softmax_temperature=self.softmax_temperature,
                inference_config=self.inference_config_,
            )
        outputs.append(out.float().cpu().numpy())
        partial_outputs.append(partial_out)

    return np.concatenate(outputs, axis=0), partial_outputs

outputs = []
partial_outputs = []
for norm_method, (x_test_s, y_test_s) in tqdm(data.items()):
    shuffle_patterns = clf.ensemble_generator_.feature_shuffle_patterns_[norm_method]
    outputs_, partial_outputs_ = _batch_forward(clf, x_test_s, y_test_s, shuffle_patterns)
    outputs.append(outputs_)
    partial_outputs.append(partial_outputs_)
outputs = np.concatenate(outputs, axis=0)

partial_outputs[0][0].keys()
col_embeddings = partial_outputs[0][0]["col_embeddings"]
repr_embeddings = partial_outputs[0][0]["representations"]

#%%
import torch
import numpy as np
from tqdm import tqdm
import types

# 1 Preprocess test data
x_test = clf._validate_data(x_test, reset=False, dtype=None, skip_check_array=True)
x_test = clf.X_encoder_.transform(x_test)
data = clf.ensemble_generator_.transform(x_test)

# 2 New forward supporting CLS + all embeddings
def model_new_inference_forward(
        self,
        X: torch.Tensor,
        y_train: torch.Tensor,
        feature_shuffles=None,
        embed_with_test=False,
        return_logits=True,
        softmax_temperature=0.9,
        inference_config=None,
):
    train_size = y_train.shape[1]
    assert train_size <= X.shape[1], "Number of training samples exceeds total samples"

    if inference_config is None:
        inference_config = InferenceConfig()

    partial_outputs = {}

    # Column embeddings
    col_embeddings = self.col_embedder(
        X,
        train_size=None if embed_with_test else train_size,
        feature_shuffles=feature_shuffles,
        mgr_config=inference_config.COL_CONFIG,
    )
    partial_outputs['col_embeddings'] = col_embeddings

    # Row embeddings (CLS + all)
    cls_embeddings, all_embeddings = self.row_interactor._aggregate_embeddings(col_embeddings)
    partial_outputs['cls_embeddings'] = cls_embeddings
    partial_outputs['all_embeddings'] = all_embeddings

    # ICL prediction uses only CLS embeddings
    out = self.icl_predictor(
        cls_embeddings,
        y_train=y_train,
        return_logits=return_logits,
        softmax_temperature=softmax_temperature,
        mgr_config=inference_config.ICL_CONFIG,
    )

    return out, partial_outputs


def new_aggregate_embeddings(self, embeddings, key_mask=None):
    """
    Aggregate row embeddings.

    Returns:
        cls_flat: CLS token embeddings, flattened for ICL (B, T, C*E)
        all_embeddings: all token embeddings (B, T, H+C, E)
    """
    # Pass through transformer
    outputs = self.tf_row(embeddings, key_padding_mask=key_mask)  # (B, T, H+C, hidden_dim)

    # LayerNorm
    outputs = self.out_ln(outputs)

    # CLS tokens: first self.num_cls tokens
    cls_tokens = outputs[:, :, :self.num_cls, :]  # (B, T, C, E)

    # Flatten CLS tokens to match original model output
    cls_flat = cls_tokens.reshape(outputs.shape[0], outputs.shape[1], -1)  # (B, T, C*E)

    # Return both
    return cls_flat, outputs


# 3 Attach new forward to model
model = clf.model_
model._inference_forward = types.MethodType(model_new_inference_forward, model)
row = clf.model_.row_interactor
row._aggregate_embeddings = types.MethodType(new_aggregate_embeddings, row)

# 4 Simplified batch inference
def _batch_forward(self, X, y, shuffle_patterns=None):
    batch_size = self.batch_size or X.shape[0]
    n_batches = int(np.ceil(len(X) / batch_size))
    X_batches = np.array_split(X, n_batches)
    y_batches = np.array_split(y, n_batches)
    if shuffle_patterns is not None:
        shuffle_batches = np.array_split(shuffle_patterns, n_batches)
    else:
        shuffle_batches = [None] * n_batches

    outputs, partial_outputs = [], []
    for Xb, yb, sp in zip(X_batches, y_batches, shuffle_batches):
        Xb_tensor = torch.from_numpy(Xb).float().to(self.device_)
        yb_tensor = torch.from_numpy(yb).float().to(self.device_)
        sp_list = sp.tolist() if sp is not None else None

        with torch.no_grad():
            out, partial_out = self.model_(
                Xb_tensor,
                yb_tensor,
                feature_shuffles=sp_list,
                return_logits=True if self.average_logits else False,
                softmax_temperature=self.softmax_temperature,
                inference_config=self.inference_config_,
            )

        outputs.append(out.cpu().numpy())
        partial_outputs.append(partial_out)

    return np.concatenate(outputs, axis=0), partial_outputs

# 5 Run inference over ensemble / normalized splits
all_outputs, all_partials = [], []
for norm_method, (x_test_s, y_test_s) in tqdm(data.items()):
    shuffle_patterns = clf.ensemble_generator_.feature_shuffle_patterns_[norm_method]
    outs, partials = _batch_forward(clf, x_test_s, y_test_s, shuffle_patterns)
    all_outputs.append(outs)
    all_partials.append(partials)

all_outputs = np.concatenate(all_outputs, axis=0)

# Example access
col_embeddings = all_partials[0][0]['col_embeddings']
cls_embeddings = all_partials[0][0]['cls_embeddings']
all_embeddings = all_partials[0][0]['all_embeddings']

#%%
#### With ASSIGNMENTS

df_col_emb = []
for i in tqdm(range(col_embeddings.shape[1])):
    for j in range(col_embeddings.shape[2]):
        df_col_emb.append({
            "sample": i,
            "column": j,
            "is_cls": j >= col_embeddings.shape[2]-4,
            "embedding": col_embeddings[0, i, j].cpu().numpy().tolist()
        })
df_col_emb = pd.DataFrame(df_col_emb)

# Flatten embeddings
flat_col_embeddings = np.stack(df_col_emb["embedding"].to_list())

# Run t-SNE using openTSNE
tsne = TSNE(
    n_components=2,
    initialization="pca",
    perplexity=30,
    metric="cosine",
    n_jobs=100,
    random_state=42
)
tsne_flat_col_embeddings = tsne.fit(flat_col_embeddings)

df_col_emb["tsne_x"] = tsne_flat_col_embeddings[:, 0]
df_col_emb["tsne_y"] = tsne_flat_col_embeddings[:, 1]

# ----- NEW PART: Add DISTRIBUTION NAMES for each column -----

# Only keep real features (skip CLS)
df_features = df_col_emb[df_col_emb["column"].astype(int) >= 4].copy()

# Map column index → assignment (column 4 maps to assignments[0], etc.)
df_features["feature_idx"] = df_features["column"].astype(int) - 4
feature_assignments = assignments[1]  # length 76

df_features["assignment"] = df_features["feature_idx"].map(lambda j: feature_assignments[j])
df_features["assignment"] = df_features["assignment"].astype(str)

from lets_plot import *
LetsPlot.setup_html()  # if not done already

# (ggplot(df_features) +
#  geom_point(aes(x='tsne_x', y='tsne_y', color='assignment'), size=2) +
#  ggtitle("t-SNE of Feature Embeddings Colored by Distribution"))

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Prepare categorical mapping
categories = pd.Categorical(df_features["assignment"])
codes = categories.codes
names = categories.categories

# Plot
plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    df_features["tsne_x"],
    df_features["tsne_y"],
    c=codes,
    cmap="tab20",   # can choose other colormaps if >20 categories
    s=20,
    alpha=0.8
)

# Create colorbar with distribution names
cbar = plt.colorbar(scatter, ticks=np.arange(len(names)))
cbar.ax.set_yticklabels(names)   # set actual names
cbar.set_label("Feature Distribution")

plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("t-SNE of Feature Embeddings Colored by Distribution")
plt.show()

#%%

df_col_emb = []
for i in tqdm(range(col_embeddings.shape[1])):
    for j in range(col_embeddings.shape[2]):
        df_col_emb.append({
            "sample": i,
            "column": j,
            "is_cls": j >= col_embeddings.shape[2]-4,
            "embedding": col_embeddings[0, i, j].cpu().numpy().tolist()
        })
df_col_emb = pd.DataFrame(df_col_emb)
flat_col_embeddings = [df_col_emb.loc[i]["embedding"] for i in df_col_emb.index]
flat_col_embeddings = np.array(flat_col_embeddings)

# Run t-SNE using openTSNE
tsne = TSNE(
    n_components=2,
    initialization="pca",
    perplexity=30,
    metric="cosine",        # common for embeddings
    n_jobs=100,               # parallel
    random_state=42
)
tsne_flat_col_embeddings = tsne.fit(flat_col_embeddings)

df_col_emb["tsne_x"] = None
df_col_emb["tsne_y"] = None
for i in tqdm(df_col_emb.index):
    df_col_emb.at[i, "tsne_x"] = tsne_flat_col_embeddings[i][0]
    df_col_emb.at[i, "tsne_y"] = tsne_flat_col_embeddings[i][1]

from lets_plot import *
df_col_emb["sample"] = df_col_emb["sample"].astype("str")
df_col_emb["column"] = df_col_emb["column"].astype("str")

(ggplot(df_col_emb) +
 geom_point(aes(x='tsne_x', y='tsne_y', color='column'), size=2)).show()

#%%
df_repr_cls_emb = []
for i in tqdm(range(repr_embeddings.shape[1])):
    emb_ = repr_embeddings[0, i].cpu().numpy()
    df_repr_cls_emb.append({
        "sample": i,
        "cls": "1",
        "embedding": emb_[:128].tolist()
    })
    df_repr_cls_emb.append({
        "sample": i,
        "cls": "2",
        "embedding": emb_[128:256].tolist()
    })
    df_repr_cls_emb.append({
        "sample": i,
        "cls": "3",
        "embedding": emb_[256:384].tolist()
    })
    df_repr_cls_emb.append({
        "sample": i,
        "cls": "4",
        "embedding": emb_[384:].tolist()
    })
df_repr_cls_emb = pd.DataFrame(df_repr_cls_emb)
flat_repr_cls_embeddings = [df_repr_cls_emb.loc[i]["embedding"] for i in df_repr_cls_emb.index]
flat_repr_cls_embeddings = np.array(flat_repr_cls_embeddings)

# Run t-SNE using openTSNE
tsne = TSNE(
    n_components=2,
    perplexity=30,
    metric="cosine",        # common for embeddings
    n_jobs=100,               # parallel
    random_state=42
)
tsne_flat_repr_cls_embeddings = tsne.fit(flat_repr_cls_embeddings)

df_repr_cls_emb["tsne_x"] = None
df_repr_cls_emb["tsne_y"] = None
for i in tqdm(df_repr_cls_emb.index):
    df_repr_cls_emb.at[i, "tsne_x"] = tsne_flat_repr_cls_embeddings[i][0]
    df_repr_cls_emb.at[i, "tsne_y"] = tsne_flat_repr_cls_embeddings[i][1]
df_repr_cls_emb["sample"] = df_repr_cls_emb["sample"].astype("str")

(ggplot(df_repr_cls_emb) +
 geom_point(aes(x='tsne_x', y='tsne_y', color='cls'), size=2)).show()

#%%
# Plot
plt.figure(figsize=(8, 6))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
plt.title("openTSNE Visualization of Embeddings")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()

#%%
#### With ASSIGNMENTS

df_col_emb = []
for i in tqdm(range(all_embeddings.shape[1])):
    for j in range(all_embeddings.shape[2]):
        df_col_emb.append({
            "sample": i,
            "column": j,
            "is_cls": j >= all_embeddings.shape[2]-4,
            "embedding": all_embeddings[0, i, j].cpu().numpy().tolist()
        })
df_col_emb = pd.DataFrame(df_col_emb)

# Flatten embeddings
flat_col_embeddings = np.stack(df_col_emb["embedding"].to_list())

# Run t-SNE using openTSNE
tsne = TSNE(
    n_components=2,
    initialization="pca",
    perplexity=30,
    metric="cosine",
    n_jobs=100,
    random_state=42
)
tsne_flat_col_embeddings = tsne.fit(flat_col_embeddings)

df_col_emb["tsne_x"] = tsne_flat_col_embeddings[:, 0]
df_col_emb["tsne_y"] = tsne_flat_col_embeddings[:, 1]

# ----- NEW PART: Add DISTRIBUTION NAMES for each column -----

# Only keep real features (skip CLS)
df_features = df_col_emb[df_col_emb["column"].astype(int) >= 4].copy()

# Map column index → assignment (column 4 maps to assignments[0], etc.)
df_features["feature_idx"] = df_features["column"].astype(int) - 4
feature_assignments = assignments[1]  # length 76

df_features["assignment"] = df_features["feature_idx"].map(lambda j: feature_assignments[j])
df_features["assignment"] = df_features["assignment"].astype(str)

from lets_plot import *
LetsPlot.setup_html()  # if not done already

# (ggplot(df_features) +
#  geom_point(aes(x='tsne_x', y='tsne_y', color='assignment'), size=2) +
#  ggtitle("t-SNE of Feature Embeddings Colored by Distribution"))

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Prepare categorical mapping
categories = pd.Categorical(df_features["assignment"])
codes = categories.codes
names = categories.categories

# Plot
plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    df_features["tsne_x"],
    df_features["tsne_y"],
    c=codes,
    cmap="tab20",   # can choose other colormaps if >20 categories
    s=20,
    alpha=0.8
)

# Create colorbar with distribution names
cbar = plt.colorbar(scatter, ticks=np.arange(len(names)))
cbar.ax.set_yticklabels(names)   # set actual names
cbar.set_label("Feature Distribution")

plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("t-SNE of Feature Embeddings Colored by Distribution")
plt.show()

