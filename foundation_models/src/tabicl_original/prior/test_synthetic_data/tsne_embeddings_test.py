#%%
# ===================================================================
# 0. IMPORTS
# ===================================================================
import torch
import numpy as np
from tqdm import tqdm
import types


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
    """
    Patch the model AFTER clf.fit() was called.
    """
    model = clf.model_
    model._inference_forward = types.MethodType(model_new_inference_forward, model)

    row = model.row_interactor
    row._aggregate_embeddings = types.MethodType(new_aggregate_embeddings, row)


# ===================================================================
# 5. MAIN FUNCTION: EXTRACT EMBEDDINGS FROM ONE X_raw
# ===================================================================
def extract_embeddings_from_X(X_raw, clf):
    """
    Fit the classifier on this dataset (train half), patch the model,
    extract CLS / COL / ALL embeddings for the test half.
    """

    # -------------------------
    # Build dense matrix
    # -------------------------
    Xsparse = X_raw['X']
    batch_size = X_raw['batch_size']
    seq_len = X_raw['seq_lens'][0]
    d = X_raw['d']
    y = X_raw['y']
    assignments = X_raw['feature_assignments']

    X_dense = sparse2dense(
        Xsparse,
        d.repeat_interleave(seq_len),
        dtype=torch.float32
    ).view(batch_size, seq_len, -1)

    # select correct batch element (your current pipeline always uses index 1)
    Xs = X_dense[1]
    ys = y[1]

    # -------------------------
    # Train / test split
    # -------------------------
    split = int(len(Xs) * 0.5)
    x_train, x_test = Xs[:split], Xs[split:]
    y_train, y_test = ys[:split], ys[split:]

    # -------------------------
    # FIT MODEL ON THIS DATASET
    # -------------------------
    clf.fit(x_train, y_train)

    # -------------------------
    # PATCH MODEL NOW
    # -------------------------
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

    return col, cls, all_, assignments


#%%
sys.path.append('/data/projects/deepintegromics/analyses/3.tabpfn/tab_icl/tabicl/src')
from tabicl_original.model.inference_config import InferenceConfig
from tabicl_original.sklearn.classifier import TabICLClassifier

clf = TabICLClassifier(n_estimators=1, norm_methods=["none", "power"],preprocess=True, model_path="/data/projects/deepintegromics/analyses/3.tabpfn/tab_icl/tabicl/src/tabicl/checkpoints/checkpoints_tabicl_original_retrained_no_seed/step-2000.ckpt", allow_auto_download=False)
#%%
X1 = torch.load('/data/projects/deepintegromics/analyses/3.tabpfn/tab_icl/tabicl/src/tabicl/checkpoints/checkpoints_try/our_batch/batch_000000.pt')
X2 = torch.load('/data/projects/deepintegromics/analyses/3.tabpfn/tab_icl/tabicl/src/tabicl/checkpoints/checkpoints_try/original_batch/batch_000000.pt')


#%%
col1, cls1, all1, assignments1 = extract_embeddings_from_X(X1, clf)
col2, cls2, all2, assignments2 = extract_embeddings_from_X(X2, clf)

#%%
def plot_feature_tsne(col_embeddings, assignments, n_cls=4, perplexity=30, n_jobs=50):
    """
    Plot t-SNE of feature embeddings.

    Supports both PyTorch tensors and NumPy arrays.
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from openTSNE import TSNE

    # ================================================================
    # 1. Build dataframe of embeddings
    # ================================================================

    df = []

    # Convert PyTorch → NumPy if needed (but keep original for indexing)
    if hasattr(col_embeddings, "detach"):
        col_emb_np = col_embeddings.detach().cpu().numpy()
    else:
        col_emb_np = col_embeddings  # already numpy

    B, n_samples, n_cols, emb_dim = col_emb_np.shape

    for i in tqdm(range(n_samples), desc="Collecting embeddings"):
        for j in range(n_cols):
            vec = col_emb_np[0, i, j]
            vec = np.asarray(vec)  # ensure numpy array

            df.append({
                "sample": i,
                "column": j,
                "is_cls": j >= n_cols - n_cls,
                "embedding": vec.tolist()
            })

    df = pd.DataFrame(df)

    flat_embeddings = np.stack(df["embedding"].to_list())

    # ================================================================
    # 2. Run t-SNE
    # ================================================================

    tsne = TSNE(
        n_components=2,
        initialization="pca",
        perplexity=perplexity,
        metric="cosine",
        n_jobs=n_jobs,
        random_state=42
    )

    tsne_results = tsne.fit(flat_embeddings)

    df["tsne_x"] = tsne_results[:, 0]
    df["tsne_y"] = tsne_results[:, 1]

    # ================================================================
    # 3. Only real features (exclude CLS)
    # ================================================================

    df_feat = df[df["is_cls"] == False].copy()

    df_feat["feature_idx"] = df_feat["column"] - n_cls
    df_feat["assignment"] = df_feat["feature_idx"].map(lambda k: assignments[k])
    df_feat["assignment"] = df_feat["assignment"].astype(str)

    # ================================================================
    # 4. Plot
    # ================================================================

    plt.figure(figsize=(12, 8))

    categories = pd.Categorical(df_feat["assignment"])
    codes = categories.codes
    names = categories.categories

    scatter = plt.scatter(
        df_feat["tsne_x"],
        df_feat["tsne_y"],
        c=codes,
        cmap="tab20",
        s=50,
        alpha=0.85
    )

    cbar = plt.colorbar(scatter, ticks=np.arange(len(names)))
    cbar.ax.set_yticklabels(names)
    cbar.set_label("Feature Distribution")

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("t-SNE of Feature Embeddings Colored by Distribution")
    plt.tight_layout()
    plt.show()

    return df_feat

#%%
df_plot = plot_feature_tsne(col1, assignments1[1], n_cls=4)
df_plot = plot_feature_tsne(col2, assignments2[1], n_cls=4)

#%%
def plot_feature_tsne(col_embeddings, assignments1, assignments2, n_cls=4,
                      perplexity=50, n_jobs=50):
    """
    Plot t-SNE of feature embeddings for a concatenated dataset.
    Color = feature distribution (4 from dataset1, 4 from dataset2).

    assignments1 : list of feature distributions for dataset1 (length = n_features)
    assignments2 : list of feature distributions for dataset2 (length = n_features)
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from openTSNE import TSNE

    # ================================================================
    # 1. Convert to numpy
    # ================================================================

    if hasattr(col_embeddings, "detach"):
        col_np = col_embeddings.detach().cpu().numpy()
    else:
        col_np = col_embeddings

    B, total_samples, n_cols, emb_dim = col_np.shape

    # number of real features (removing CLS tokens)
    n_features = n_cols - n_cls

    # number of samples in dataset1 (the rest are dataset2)
    # NOTE: this requires col_embeddings = concat([col1, col2], axis=1)
    n_samples_1 = total_samples // 2

    # ================================================================
    # 2. Build list of embedding entries
    # ================================================================

    rows = []

    for i in tqdm(range(total_samples), desc="Collecting embeddings"):
        for j in range(n_cols):

            vec = col_np[0, i, j]

            rows.append({
                "sample": i,
                "column": j,
                "is_cls": j >= n_cols - n_cls,
                "embedding": vec.tolist()
            })

    df = pd.DataFrame(rows)

    # Flatten embeddings for t-SNE
    flat_embeddings = np.stack(df["embedding"].to_list())

    # ================================================================
    # 3. Run t-SNE
    # ================================================================

    tsne = TSNE(
        n_components=2,
        initialization="pca",
        perplexity=perplexity,
        metric="cosine",
        n_jobs=n_jobs,
        random_state=42
    )

    tsne_results = tsne.fit(flat_embeddings)

    df["tsne_x"] = tsne_results[:, 0]
    df["tsne_y"] = tsne_results[:, 1]

    # ================================================================
    # 4. Keep only REAL features (exclude CLS tokens)
    # ================================================================

    df_feat = df[df["is_cls"] == False].copy()
    df_feat["feature_idx"] = df_feat["column"] - n_cls

    # ================================================================
    # 5. Assign distributions properly (dataset1 OR dataset2)
    # ================================================================

    def choose_assignment(sample, feature_idx):
        if sample < n_samples_1:
            return assignments1[feature_idx]
        else:
            return assignments2[feature_idx]

    df_feat["assignment"] = df_feat.apply(
        lambda r: choose_assignment(r["sample"], r["feature_idx"]),
        axis=1
    )

    df_feat["assignment"] = df_feat["assignment"].astype(str)

    # ================================================================
    # 6. Plot
    # ================================================================

    plt.figure(figsize=(12, 8))

    categories = pd.Categorical(df_feat["assignment"])
    codes = categories.codes
    names = categories.categories

    scatter = plt.scatter(
        df_feat["tsne_x"], df_feat["tsne_y"],
        c=codes,
        cmap="tab20",
        s=30,
        alpha=0.9
    )

    cbar = plt.colorbar(scatter, ticks=np.arange(len(names)))
    cbar.ax.set_yticklabels(names)
    cbar.set_label("Feature Distribution")

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("t-SNE of Feature Embeddings (Both Datasets)")
    plt.tight_layout()
    plt.show()

    return df_feat
#%%
def to_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else x

col1_np = to_numpy(col1)
col2_np = to_numpy(col2)

n_cols_min = min(col1_np.shape[2], col2_np.shape[2])
col1_np = col1_np[:, :, :n_cols_min, :]
col2_np = col2_np[:, :, :n_cols_min, :]

col_concat = np.concatenate([col1_np, col2_np], axis=1)

df_all = plot_feature_tsne(
    col_concat,
    assignments1[1],   # 4 distributions for dataset 1
    assignments2[1],   # 4 distributions for dataset 2
    n_cls=4
)
#%%
def to_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else x

col1_np = to_numpy(all1)
col2_np = to_numpy(all2)

n_cols_min = min(col1_np.shape[2], col2_np.shape[2])
col1_np = col1_np[:, :, :n_cols_min, :]
col2_np = col2_np[:, :, :n_cols_min, :]

col_concat = np.concatenate([col1_np, col2_np], axis=1)

df_all = plot_feature_tsne(
    col_concat,
    assignments1[1],   # 4 distributions for dataset 1
    assignments2[1],   # 4 distributions for dataset 2
    n_cls=4
)

#%%
##CLS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openTSNE import TSNE
from lets_plot import *
repr_embeddings = cls2
df_repr_cls_emb = []
for i in tqdm(range(repr_embeddings.shape[1])):
    emb_ = repr_embeddings[0, i]
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


# Plot using raw matplotlib
plt.figure(figsize=(8, 6))
plt.scatter(tsne_flat_repr_cls_embeddings[:, 0],
            tsne_flat_repr_cls_embeddings[:, 1],
            s=5)
plt.title("openTSNE Visualization of CLS Embeddings")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.show()
