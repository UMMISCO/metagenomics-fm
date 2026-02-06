#%%
import os
import sys
sys.path.append('/data/projects/deepintegromics/analyses/3.tabpfn/tab_icl/tabicl/tests')
from testing_data.pasolli.pasolli import open_pasolli
from testing_data.metacardis.metacardis import open_metacardis
from testing_data.preprocessing.filter_or_logic import open_and_filter

#%%
# ===================================================================
# 0. IMPORTS
# ===================================================================
import torch
import numpy as np
from tqdm import tqdm
import types

sys.path.append('/data/projects/deepintegromics/analyses/3.tabpfn/tab_icl/tabicl/src')
from tabicl_original.model.layers import InducedSelfAttentionBlock
# ===================================================================
# 1. ORIGINAL FUNCTION — sparse2dense (unchanged)
# ===================================================================
def sparse2dense(
        sparse_tensor: torch.Tensor,
        row_lengths: torch.Tensor,
        max_len=None,
        dtype=torch.float32,
):
    assert sparse_tensor.dim() == 1
    assert row_lengths.sum() == len(sparse_tensor)

    num_rows = len(row_lengths)
    max_len = max_len or row_lengths.max().item()

    dense = torch.zeros(num_rows, max_len, dtype=dtype, device=sparse_tensor.device)
    indices = torch.arange(max_len, device=sparse_tensor.device)
    mask = indices.unsqueeze(0) < row_lengths.unsqueeze(1)
    dense[mask] = sparse_tensor.to(dtype)
    return dense

# ===================================================================
# 2. YOUR CUSTOM FORWARD FUNCTIONS (unchanged)
# ===================================================================
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
    from tabicl_original.model.inference_config import InferenceConfig

    train_size = y_train.shape[1]
    if inference_config is None:
        inference_config = InferenceConfig()

    partial_outputs = {}

    col_embeddings = self.col_embedder(
        X,
        train_size=None if embed_with_test else train_size,
        feature_shuffles=feature_shuffles,
        mgr_config=inference_config.COL_CONFIG,
    )
    partial_outputs['col_embeddings'] = col_embeddings

    cls_embeddings, all_embeddings = self.row_interactor._aggregate_embeddings(col_embeddings)
    partial_outputs['cls_embeddings'] = cls_embeddings
    partial_outputs['all_embeddings'] = all_embeddings

    out = self.icl_predictor(
        cls_embeddings,
        y_train=y_train,
        return_logits=return_logits,
        softmax_temperature=softmax_temperature,
        mgr_config=inference_config.ICL_CONFIG,
    )

    return out, partial_outputs


def new_aggregate_embeddings(self, embeddings, key_mask=None):
    outputs = self.tf_row(embeddings, key_padding_mask=key_mask)
    outputs = self.out_ln(outputs)

    cls_tokens = outputs[:, :, :self.num_cls, :]
    cls_flat = cls_tokens.reshape(outputs.shape[0], outputs.shape[1], -1)

    return cls_flat, outputs

def patched_isab_forward(self, X, *args, **kwargs):
    """
    Forward override for ISAB to capture the FIRST MAB output (H = MAB(I, X)).
    """
    # Expand inducing points (name may differ; see note below)
    I = self.I                      # <-- If yours is named differently, tell me
    I = I.unsqueeze(0).expand(X.size(0), -1, -1)

    # FIRST ATTENTION BLOCK (inducing -> data)
    H = self.mab1(I, X)
    self.first_block_output = H     # <-- store it for extraction later

    # SECOND ATTENTION BLOCK (normal ISAB output)
    Y = self.mab2(X, H)

    return Y

# ===================================================================
# 3. BATCH FORWARD FUNCTION (unchanged)
# ===================================================================
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

# ===================================================================
# 4. PATCH MODEL AFTER FIT (important)
# ===================================================================
def patch_model_after_fit(clf):
    model = clf.model_
    model._inference_forward = types.MethodType(model_new_inference_forward, model)

    row = model.row_interactor
    row._aggregate_embeddings = types.MethodType(new_aggregate_embeddings, row)

    # --- PATCH ISAB blocks to expose MAB1 output ---
    tf = model.col_embedder.tf_col
    for layer in tf.blocks:
        if isinstance(layer, InducedSelfAttentionBlock):
            mab1 = layer.multihead_attn1

            # Save original forward
            mab1._orig_forward = mab1.forward

            # Patch forward ONLY
            mab1.forward = types.MethodType(hooked_mab1_forward, mab1)

def hooked_mab1_forward(self, query, key, value, *args, **kwargs):
    """
    Wrap MultiheadAttentionBlock.forward to capture its output.
    """
    out = self._orig_forward(query, key, value, *args, **kwargs)
    self._last_mab1_output = out
    return out

# ===================================================================
# 5. MAIN FUNCTION: EXTRACT EMBEDDINGS FROM ONE X_raw
# ===================================================================
def extract_embeddings_from_X(X, y, clf):
    """
    Fit the classifier on this dataset (train half), patch the model,
    extract CLS / COL / ALL embeddings for the test half.
    """

    # select correct batch element (your current pipeline always uses index 1)
    Xs = X
    ys = y

    # -------------------------
    # Train / test split
    # -------------------------
    from sklearn.utils import shuffle

    split = int(len(Xs) * 0.99)
    Xs, ys = shuffle(Xs,ys)
    x_train, x_test = Xs[:split], Xs[split:]
    y_train, y_test = ys[:split], ys[split:]

    # -------------------------
    # FIT MODEL ON THIS DATASET
    # -------------------------
    clf.fit(x_train, y_train)
    patch_model_after_fit(clf)

    # -------------------------
    # Preprocess test data
    # -------------------------
    x_test = clf._validate_data(x_test, reset=False, dtype=None, skip_check_array=True)
    x_test = clf.X_encoder_.transform(x_test)

    data = clf.ensemble_generator_.transform(x_test)
    # -------------------------
    # RUN INFERENCE
    # -------------------------
    all_partials = []
    for norm_method, (x_s, y_s) in data.items():
        shuffle = clf.ensemble_generator_.feature_shuffle_patterns_[norm_method]
        _, partials = _batch_forward(clf, x_s, y_s, shuffle)
        all_partials.append(partials)

    # -------------------------
    # EXTRACT EMBEDDINGS
    # -------------------------
    parts = all_partials[0][0]

    col = parts['col_embeddings'].cpu().numpy()
    cls = parts['cls_embeddings'].cpu().numpy()
    all_ = parts['all_embeddings'].cpu().numpy()

    tf = clf.model_.col_embedder.tf_col
    # First ISAB, first MAB (inducing-point attention)
    inducing_embeddings = tf.blocks[0].multihead_attn1._last_mab1_output

    y_all = np.concatenate([y_train, y_test])

    return col, cls, all_ , y_all, inducing_embeddings
#%%

#TEST
import numpy as np
import pandas as pd
from openTSNE import TSNE
from lets_plot import *

sys.path.append('/data/projects/deepintegromics/analyses/3.tabpfn/tab_icl/tabicl/src')
from tabicl_original.model.inference_config import InferenceConfig
from tabicl_original.sklearn.classifier import TabICLClassifier

clf = TabICLClassifier(
    n_estimators=1,
    norm_methods=["none", "power"],
    preprocess=True,
    model_path="/data/projects/deepintegromics/analyses/3.tabpfn/tab_icl/tabicl/src/tabicl/checkpoints/checkpoints_tabicl_original_retrained_no_seed/step-2000.ckpt",
    allow_auto_download=False
)

X, y = open_pasolli('abundance_cirrhosis--stagevalidation')
X, y = open_and_filter(X, 0.2, 0.01)
col1, cls1, all1, y_all1, inducing1 = extract_embeddings_from_X(X, y, clf)

#%%
#TEST INDUCING POINTS
from openTSNE import TSNE
from lets_plot import *
LetsPlot.setup_html()
import torch.nn.functional as F

X = inducing1.sum(dim=1)
X = F.normalize(X, dim=1)         # normalize
X = X.cpu().numpy()   # MUST be numpy

tsne = TSNE(
    n_components=2,
    perplexity=40,
    metric="cosine",
    initialization="pca",
    n_jobs=8,
    random_state=42,
)

X_embedded = tsne.fit(X)
n_features = X_embedded.shape[0]

df = pd.DataFrame({
    "x": X_embedded[:, 0],
    "y": X_embedded[:, 1],
    "feature_id": np.arange(n_features),
})

p = (ggplot(df, aes("x", "y", color="feature_id")) + \
    geom_point(size=3) )
p.show()

#%%
#TEST INDUCING POINTS
# ============================================================
# 0. IMPORTS
# ============================================================
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from openTSNE import TSNE
from sklearn.cluster import KMeans
from lets_plot import *

LetsPlot.setup_html()
assert inducing1.dim() == 3

# ============================================================
# 2. AGGREGATE INDUCING POINTS → ONE VECTOR PER FEATURE
# ============================================================
X = inducing1.sum(dim=1)          # (197, 128)
X = F.normalize(X, dim=1)         # IMPORTANT
X_np = X.cpu().numpy()            # openTSNE needs NumPy

# ============================================================
# 3. t-SNE WITH openTSNE
# ============================================================
tsne = TSNE(
    n_components=2,
    perplexity=30,
    metric="euclidean",
    random_state=42,
    n_jobs=8,
    verbose=True,
)

X_embedded = tsne.fit(X_np)        # (197, 2)

# ============================================================
# 4. CLUSTER IN EMBEDDING SPACE (NOT t-SNE SPACE)
# ============================================================
k = 6   # try 4–10
labels = KMeans(
    n_clusters=k,
    random_state=42,
    n_init="auto",
).fit_predict(X_np)

# ============================================================
# 5. PLOT (COLOR BY CLUSTER)
# ============================================================
df = pd.DataFrame({
    "x": X_embedded[:, 0],
    "y": X_embedded[:, 1],
    "cluster": labels.astype(str),
})

plot = (
    ggplot(df, aes("x", "y", color="cluster"))
    + geom_point(size=4, alpha=0.85)
    + labs(
        title="t-SNE of Inducing-Point Aggregated Column Embeddings",
        color="Cluster"
    )
    + theme_minimal()
)

plot.show()

#%%
#TEST INDUCING POINT OF ALL PASOLLI DATASET
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from openTSNE import TSNE
from sklearn.cluster import KMeans
from lets_plot import *

LetsPlot.setup_html()

dataset_names = [
    "abundance_cirrhosis--stagediscovery",
    "abundance_cirrhosis--stagevalidation",
    "abundance_obesity",
    "abundance_ibd",
    "abundance_t2d",
    "abundance_WT2D"
]

all_X = []
all_labels = []

for name in dataset_names:
    X, y = open_pasolli(name)
    X, y = open_and_filter(X, 0.2, 0.01)
    _, _, _, _, inducing = extract_embeddings_from_X(X, y, clf)

    # Aggregate inducing points
    X_agg = inducing.sum(dim=1)
    X_agg = F.normalize(X_agg, dim=1)
    all_X.append(X_agg.cpu().numpy())

    # Keep track of dataset
    all_labels.extend([name] * X_agg.shape[0])

# Concatenate all datasets
X_combined = np.vstack(all_X)

# t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=30,
    metric="cosine",
    random_state=42,
    n_jobs=8,
    verbose=True,
)
X_embedded = tsne.fit(X_combined)

# Optionally cluster in original embedding space
k = 6
cluster_labels = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(X_combined)

# Prepare dataframe for plotting
df = pd.DataFrame({
    "x": X_embedded[:, 0],
    "y": X_embedded[:, 1],
    "dataset": all_labels,
    "cluster": cluster_labels.astype(str)
})

# Plot: color by dataset
plot = (
        ggplot(df, aes("x", "y", color="cluster"))
        + geom_point(size=4, alpha=0.85)
        + labs(title="t-SNE of Inducing-Point Aggregated Column Embeddings (All Datasets)")
        + theme_minimal()
)

plot.show()

#%%

""" If you want the T-sne of the feature cell embeddings colored by the dataset"""

import numpy as np
import pandas as pd
from openTSNE import TSNE
from lets_plot import *

sys.path.append('/data/projects/deepintegromics/analyses/3.tabpfn/tab_icl/tabicl/src')
from tabicl_original.model.inference_config import InferenceConfig
from tabicl_original.sklearn.classifier import TabICLClassifier

names = [
    "abundance_cirrhosis--stagediscovery",
    "abundance_cirrhosis--stagevalidation",
    "abundance_obesity",
    "abundance_ibd",
    "abundance_t2d",
    "abundance_WT2D"
]

clf = TabICLClassifier(
    n_estimators=1,
    norm_methods=["none", "power"],
    preprocess=True,
    model_path="/data/projects/deepintegromics/analyses/3.tabpfn/tab_icl/tabicl/src/tabicl/checkpoints/checkpoints_tabicl_original_retrained_no_seed/step-2000.ckpt",
    allow_auto_download=False
)

# Initialize lets-plot
LetsPlot.setup_html()

# Collect embeddings from all datasets
all_flat_embeddings = []
all_dataset_labels = []

for name in names:
    X, y = open_pasolli(name)
    X, y = open_and_filter(X, 0.2, 0.01)
    col1, cls1, all1, y_all1 = extract_embeddings_from_X(X, y, clf)

    col_emb = col1[0]  # (samples, features, emb_dim)

    # Skip first 4 CLS embeddings (columns 0-3)
    col_emb_features = col_emb[:, 4:, :]  # (samples, features-4, emb_dim)

    n_samples, n_features, emb_dim = col_emb_features.shape
    flat_embeddings = col_emb_features.reshape(-1, emb_dim)  # (samples*(features-4), emb_dim)

    dataset_labels = np.array([name] * flat_embeddings.shape[0])

    all_flat_embeddings.append(flat_embeddings)
    all_dataset_labels.append(dataset_labels)

# Combine all datasets
all_flat_embeddings = np.vstack(all_flat_embeddings)
all_dataset_labels = np.concatenate(all_dataset_labels)

# Run t-SNE
tsne = TSNE(
    n_components=2,
    initialization="pca",
    perplexity=30,
    metric="cosine",
    n_jobs=8,
    random_state=42
)
tsne_results = tsne.fit(all_flat_embeddings)

# Build DataFrame
df_tsne = pd.DataFrame({
    "tsne_x": tsne_results[:, 0],
    "tsne_y": tsne_results[:, 1],
    "dataset": all_dataset_labels
})

df_tsne.to_csv("/data/projects/deepintegromics/analyses/3.tabpfn/tsne.csv")
# Plot
p = (
        ggplot(df_tsne) +
        geom_point(aes(x="tsne_x", y="tsne_y", color="dataset"), size=3, alpha=0.8) +
        ggtitle("t-SNE of Feature Embeddings Across All Datasets") +
        xlab("t-SNE 1") +
        ylab("t-SNE 2")
)

p.show()

#%%
""" If you want the T-sne of the samples having one mean embedding (average across the features of that datasets) colored by the label"""
import numpy as np
import pandas as pd
from openTSNE import TSNE
from lets_plot import *

sys.path.append('/data/projects/deepintegromics/analyses/3.tabpfn/tab_icl/tabicl/src')
from tabicl_original.model.inference_config import InferenceConfig
from tabicl_original.sklearn.classifier import TabICLClassifier

names = [
    "abundance_cirrhosis--stagediscovery",
    "abundance_cirrhosis--stagevalidation",
    "abundance_obesity",
    "abundance_ibd",
    "abundance_t2d",
    "abundance_WT2D"
]

clf = TabICLClassifier(
    n_estimators=1,
    norm_methods=["none", "power"],
    preprocess=True,
    model_path="/data/projects/deepintegromics/analyses/3.tabpfn/tab_icl/tabicl/src/tabicl/checkpoints/checkpoints_tabicl_original_retrained_no_seed/step-2000.ckpt",
    allow_auto_download=False
)

# Initialize lets-plot
LetsPlot.setup_html()

all_sample_embeddings = []
all_labels = []

for dataset_idx, name in enumerate(names):
    X, y = open_pasolli(name)
    X, y = open_and_filter(X, 0.2, 0.01)

    col1, cls1, all1, y_all1 = extract_embeddings_from_X(X, y, clf)

    # Skip first 4 CLS embeddings, mean across remaining features
    sample_emb = all1[0][:, 4:, :].mean(axis=1)  # shape: (n_samples, emb_dim)

    # Encode labels: 0 = control, disease gets unique number per dataset
    y_encoded = np.where(y_all1 == 0, 0, 1 + dataset_idx * 2)

    all_sample_embeddings.append(sample_emb)
    all_labels.append(y_encoded)

# Combine all datasets
all_sample_embeddings = np.vstack(all_sample_embeddings)
all_labels = np.concatenate(all_labels)

# Run t-SNE
tsne = TSNE(
    n_components=2,
    initialization="pca",
    perplexity=30,
    metric="cosine",
    n_jobs=8,
    random_state=42
)
tsne_results = tsne.fit(all_sample_embeddings)

# Build DataFrame
df_tsne = pd.DataFrame({
    "tsne_x": tsne_results[:, 0],
    "tsne_y": tsne_results[:, 1],
    "label": all_labels.astype(str)  # categorical
})

# Plot with distinct colors for each label
p = (
        ggplot(df_tsne) +
        geom_point(aes(x="tsne_x", y="tsne_y", color="label"), size=4, alpha=0.8) +
        ggtitle("t-SNE of Sample Embeddings Across All Datasets") +
        xlab("t-SNE 1") +
        ylab("t-SNE 2")
)

p.show()

#%%
import numpy as np
import pandas as pd
from openTSNE import TSNE
from lets_plot import *

sys.path.append('/data/projects/deepintegromics/analyses/3.tabpfn/tab_icl/tabicl/src')
from tabicl_original.model.inference_config import InferenceConfig
from tabicl_original.sklearn.classifier import TabICLClassifier

names = [
    "abundance_cirrhosis--stagediscovery",
    "abundance_cirrhosis--stagevalidation",
    "abundance_obesity",
    "abundance_ibd",
    "abundance_t2d",
    "abundance_WT2D"
]

clf = TabICLClassifier(
    n_estimators=1,
    norm_methods=["none", "power"],
    preprocess=True,
    model_path="/data/projects/deepintegromics/analyses/3.tabpfn/tab_icl/tabicl/src/tabicl/checkpoints/checkpoints_tabicl_original_retrained_no_seed/step-2000.ckpt",
    allow_auto_download=False
)

LetsPlot.setup_html()

for name in names:

    print(f"Running t-SNE for dataset: {name}")

    # Load dataset
    X, y = open_pasolli(name)
    X, y = open_and_filter(X, 0.2, 0.01)

    # Run model to extract embeddings
    col1, cls1, all1, y_all1 = extract_embeddings_from_X(X, y, clf)

    # Mean embedding across features (skip 4 CLS tokens)
    sample_emb = col1[0][:, 4:, :].mean(axis=1)  # (n_samples, emb_dim)

    # Class labels: 0 or 1
    labels = y_all1.astype(int)

    # Run t-SNE for this dataset only
    tsne = TSNE(
        n_components=2,
        initialization="pca",
        perplexity=30,
        metric="cosine",
        n_jobs=8,
        random_state=42
    )
    tsne_results = tsne.fit(sample_emb)

    # Build dataframe
    df_tsne = pd.DataFrame({
        "tsne_x": tsne_results[:, 0],
        "tsne_y": tsne_results[:, 1],
        "label": labels.astype(str)
    })

    # Plot t-SNE
    p = (
        ggplot(df_tsne) +
        geom_point(aes(x="tsne_x", y="tsne_y", color="label"), size=4, alpha=0.8) +
        ggtitle(f"t-SNE of Sample Embeddings — {name}") +
        xlab("t-SNE 1") +
        ylab("t-SNE 2")
    )

    p.show()

#%%
import numpy as np
import pandas as pd
from openTSNE import TSNE
from lets_plot import *

sys.path.append('/data/projects/deepintegromics/analyses/3.tabpfn/tab_icl/tabicl/src')
from tabicl_original.model.inference_config import InferenceConfig
from tabicl_original.sklearn.classifier import TabICLClassifier

names = [
    "abundance_cirrhosis--stagediscovery",
    "abundance_cirrhosis--stagevalidation",
    "abundance_obesity",
    "abundance_ibd",
    "abundance_t2d",
    "abundance_WT2D"
]

clf = TabICLClassifier(
    n_estimators=1,
    norm_methods=["none", "power"],
    preprocess=True,
    model_path="/data/projects/deepintegromics/analyses/3.tabpfn/tab_icl/tabicl/src/tabicl/checkpoints/checkpoints_tabicl_original_retrained_no_seed/step-2000.ckpt",
    allow_auto_download=False
)

LetsPlot.setup_html()

for name in names:

    print(f"Running t-SNE for dataset: {name}")

    # Load dataset
    X, y = open_pasolli(name)
    X, y = open_and_filter(X, 0.2, 0.01)

    # Extract embeddings
    col1, cls1, all1, y_all1 = extract_embeddings_from_X(X, y, clf)

    # ----- CONCATENATE THE 4 CLS TOKENS -----
    # col1[0] has shape (n_samples, seq_len, emb_dim)
    # CLS tokens are usually the first 4 positions
    cls_concat = col1[0][:, :4, :].reshape(col1[0].shape[0], -1)
    # Now shape is (n_samples, 4 * emb_dim)

    labels = y_all1.astype(int)

    # Run t-SNE
    tsne = TSNE(
        n_components=2,
        initialization="pca",
        perplexity=30,
        metric="cosine",
        n_jobs=8,
        random_state=42
    )
    tsne_results = tsne.fit(cls_concat)

    # DataFrame
    df_tsne = pd.DataFrame({
        "tsne_x": tsne_results[:, 0],
        "tsne_y": tsne_results[:, 1],
        "label": labels.astype(str)
    })

    # Plot
    p = (
        ggplot(df_tsne) +
        geom_point(aes("tsne_x", "tsne_y", color="label"), size=4, alpha=0.8) +
        ggtitle(f"t-SNE of Concatenated CLS Embeddings — {name}") +
        xlab("t-SNE 1") +
        ylab("t-SNE 2")
    )

    p.show()
