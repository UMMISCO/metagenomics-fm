import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, List, Union

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from testing.testing_data.pasolli.pasolli import open_pasolli
from testing.testing_data.metacardis.metacardis import open_metacardis
from testing.testing_data.preprocessing.filter_or_logic import open_and_filter

# =============================================================================
# VISUALIZATION MODULE
# =============================================================================

class PerturbationVisualizer:
    """
    Three scatter plots for perturbation analysis.
    Each point is always one (sample, feature) pair → n_samples × n_features points total.
    Only shared features between original and perturbed are plotted (missing ≠ zero).
    """

    @staticmethod
    def _paired_vals(
        original: pd.DataFrame,
        X_pert: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Return (x_vals, y_vals, shared_cols) aligned by sorted shared columns."""
        shared_cols = sorted(original.columns.intersection(X_pert.columns))
        x_vals = original[shared_cols].values.flatten().astype(float)
        y_vals = X_pert[shared_cols].values.flatten().astype(float)
        return x_vals, y_vals, shared_cols

    # ------------------------------------------------------------------
    # Plot 1: one subplot per perturbation
    # ------------------------------------------------------------------
    def plot_per_perturbation(
        self,
        original: pd.DataFrame,
        perturbations: List[Tuple[str, pd.DataFrame]],
        subplot_size: Tuple[int, int] = (5, 5),
        save_path: Optional[str] = None,
        title: Optional[str] = None,
    ) -> None:
        """One subplot per perturbation. x = original value, y = perturbed value.
        Total points per panel = n_samples × n_shared_features."""
        n = len(perturbations)
        ncols = min(n, 3)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(subplot_size[0] * ncols, subplot_size[1] * nrows),
            squeeze=False,
        )
        palette = sns.color_palette("husl", n)

        for idx, (label, X_pert) in enumerate(perturbations):
            ax = axes[idx // ncols][idx % ncols]
            x_vals, y_vals, shared_cols = self._paired_vals(original, X_pert)
            n_pts = len(x_vals)  # n_samples × n_shared_features

            ax.scatter(x_vals, y_vals, alpha=0.3, s=8,
                       color=palette[idx], edgecolors='none',
                       label=f"n_pts={n_pts} (s={X_pert.shape[0]}, f={X_pert.shape[1]})")
            lim_max = max(x_vals.max(), y_vals.max())
            ax.plot([0, lim_max], [0, lim_max], color='black',
                    linewidth=1, linestyle='--', label='y = x')
            ax.set_xlabel("Original value", fontsize=14)
            ax.set_ylabel("Perturbed value", fontsize=14)
            ax.set_title(label, fontsize=14, fontweight='bold')
            ax.tick_params(axis='both', labelsize=13)
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend(fontsize=12, frameon=True)

        for idx in range(n, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    # ------------------------------------------------------------------
    # Plot 2: all perturbations overlaid, colour = perturbation level
    # ------------------------------------------------------------------
    def plot_overlay(
        self,
        original: pd.DataFrame,
        perturbations: List[Tuple[str, pd.DataFrame]],
        figsize: Tuple[int, int] = (8, 7),
        save_path: Optional[str] = None,
        title: Optional[str] = None,
    ) -> None:
        """All perturbations overlaid. Colour = perturbation level.
        Total points per layer = n_samples × n_shared_features."""
        fig, ax = plt.subplots(figsize=figsize)
        palette = sns.color_palette("husl", len(perturbations))

        for idx, (label, X_pert) in enumerate(perturbations):
            x_vals, y_vals, _ = self._paired_vals(original, X_pert)
            n_pts = len(x_vals)
            ax.scatter(x_vals, y_vals, alpha=0.3, s=8,
                       color=palette[idx], edgecolors='none',
                       label=f"{label} — {n_pts} pts (f={X_pert.shape[1]})")

        all_max = max(
            original.values.astype(float).max(),
            max(X.values.astype(float).max() for _, X in perturbations),
        )
        ax.plot([0, all_max], [0, all_max], color='black',
                linewidth=1.2, linestyle='--', label='y = x (no change)')
        ax.set_xlabel("Original value", fontsize=11)
        ax.set_ylabel("Perturbed value", fontsize=11)
        ax.set_title(title or "Original vs. Perturbations (overlay)",
                     fontsize=12, fontweight='bold')
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=8, frameon=True, markerscale=2)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    # ------------------------------------------------------------------
    # Plot 3: colour = sample class, black ring = protected feature
    # ------------------------------------------------------------------
    def plot_class_and_protected(
        self,
        original: pd.DataFrame,
        perturbations: List[Tuple[str, pd.DataFrame]],
        y_labels: pd.Series,
        protected_features: List[str],
        subplot_size: Tuple[int, int] = (6, 6),
        save_path: Optional[str] = None,
        title: Optional[str] = None,
    ) -> None:
        """One subplot per perturbation.
        Colour = sample class label (repeated per feature).
        Black ring = protected feature."""
        n = len(perturbations)
        ncols = min(n, 3)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(subplot_size[0] * ncols, subplot_size[1] * nrows),
            squeeze=False,
        )

        classes = sorted(y_labels.unique())
        class_palette = dict(zip(classes, sns.color_palette("Set2", len(classes))))

        for idx, (label, X_pert) in enumerate(perturbations):
            ax = axes[idx // ncols][idx % ncols]
            shared_cols = sorted(original.columns.intersection(X_pert.columns))
            n_samples = original.shape[0]

            x_vals = original[shared_cols].values.flatten().astype(float)
            y_vals = X_pert[shared_cols].values.flatten().astype(float)

            # Each (sample, feature) pair gets the colour of that sample's class
            colours = np.array([class_palette[c] for c in y_labels for _ in shared_cols])
            # Protected mask: True where the feature is protected
            is_protected = np.tile([f in protected_features for f in shared_cols], n_samples)

            # Non-protected behind
            ax.scatter(x_vals[~is_protected], y_vals[~is_protected],
                       c=colours[~is_protected], alpha=0.2, s=8, edgecolors='none')
            # Protected on top with black ring
            ax.scatter(x_vals[is_protected], y_vals[is_protected],
                       c=colours[is_protected], alpha=0.7, s=25,
                       edgecolors='black', linewidths=0.6)

            lim_max = max(x_vals.max(), y_vals.max())
            ax.plot([0, lim_max], [0, lim_max], color='black',
                    linewidth=1, linestyle='--')

            if idx == 0:
                for cls, col in class_palette.items():
                    ax.scatter([], [], color=col, label=str(cls), s=20)
                ax.scatter([], [], edgecolors='black', facecolors='grey',
                           s=25, linewidths=0.6, label='protected feature')
                ax.legend(fontsize=7, frameon=True, title='Class')

            ax.set_xlabel("Original value", fontsize=14)
            ax.set_ylabel("Perturbed value", fontsize=14)
            ax.set_title(label, fontsize=14, fontweight='bold')
            ax.grid(True, linestyle="--", alpha=0.3)

        for idx in range(n, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.suptitle(title or "Original vs. Perturbations (by class & protected features)",
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    # ------------------------------------------------------------------
    # Plot 4: distribution of (perturbed - original) split by class
    # ------------------------------------------------------------------
    def plot_delta_by_class(
        self,
        original: pd.DataFrame,
        perturbations: List[Tuple[str, pd.DataFrame]],
        y_labels: pd.Series,
        figsize: Tuple[int, int] = (12, 5),
        save_path: Optional[str] = None,
        title: Optional[str] = None,
    ) -> None:
        """
        Boxplot of (perturbed - original) per (perturbation level x class).
        Each observation is one (sample, feature) delta value.
        If boxes overlap -> perturbation is class-agnostic.
        If boxes separate -> perturbation introduces class bias.
        """
        classes = sorted(y_labels.unique())
        class_palette = dict(zip(classes, sns.color_palette("Set2", len(classes))))

        # Build long-form: one row per (sample, feature, perturbation)
        records = []
        for label, X_pert in perturbations:
            shared_cols = sorted(original.columns.intersection(X_pert.columns))
            for sample_idx, cls in zip(original.index, y_labels):
                orig_row = original.loc[sample_idx, shared_cols].values.astype(float)
                pert_row = X_pert.loc[sample_idx, shared_cols].values.astype(float)
                # Exclude only (0, 0) pairs — keep if non-zero in either original or perturbed
                nonzero_mask = (orig_row != 0) | (pert_row != 0)
                for delta in (pert_row[nonzero_mask] - orig_row[nonzero_mask]):
                    records.append({'perturbation': label, 'class': str(cls), 'delta': delta})

        df_long = pd.DataFrame(records)
        pert_labels = [label for label, _ in perturbations]
        n_classes = len(classes)
        width = 0.8 / n_classes
        offsets = np.linspace(-(0.8 - width) / 2, (0.8 - width) / 2, n_classes)

        fig, ax = plt.subplots(figsize=figsize)

        for cls_idx, cls in enumerate(classes):
            df_cls = df_long[df_long['class'] == str(cls)]
            positions = np.arange(len(pert_labels)) + offsets[cls_idx]
            data_per_pert = [
                df_cls[df_cls['perturbation'] == lbl]['delta'].values
                for lbl in pert_labels
            ]
            ax.boxplot(
                data_per_pert,
                positions=positions,
                widths=width * 0.9,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color='black', linewidth=1.5),
                boxprops=dict(facecolor=class_palette[cls], alpha=0.7),
                whiskerprops=dict(linewidth=1),
                capprops=dict(linewidth=1),
            )
            ax.scatter([], [], color=class_palette[cls], label=f'Class {cls}', s=40)

        ax.axhline(0, color='black', linewidth=1.2, linestyle='--', label='no change')
        ax.set_xticks(np.arange(len(pert_labels)))
        ax.set_xticklabels(pert_labels, rotation=20, ha='right', fontsize=8)
        ax.set_ylabel("Perturbed - Original (excl. double-zero pairs)", fontsize=11)
        ax.set_xlabel("Perturbation level", fontsize=11)
        ax.set_title(title or "Delta abundance by class", fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, frameon=True)
        ax.grid(True, linestyle="--", alpha=0.3, axis='y')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()


    # ------------------------------------------------------------------
    # Plot 5: per-class feature trajectory across perturbation levels
    # ------------------------------------------------------------------
    def plot_feature_trajectories(
        self,
        original: pd.DataFrame,
        perturbations: List[Tuple[str, pd.DataFrame]],
        y_labels: pd.Series,
        protected_features: List[str],
        figsize: Tuple[int, int] = (16, 5),
        save_path: Optional[str] = None,
        title: Optional[str] = None,
    ) -> None:
        """
        One subplot per class.
        X-axis = perturbation level (k=0, k=10, ...).
        Y-axis = mean abundance of each feature for samples of that class.
        Each dot = one feature at one k level, placed inside the boxplot with jitter.
        Dots of the same feature are connected by a line across k levels.
        Protected features in red, others in grey.
        Y-axis capped at 0.02 to show low-abundance features clearly.
        """
        classes = sorted(y_labels.unique())
        all_datasets = [("k=0", original)] + [(label, X) for label, X in perturbations]
        shared_cols = sorted(set.intersection(*[set(X.columns) for _, X in all_datasets]))
        n_k = len(all_datasets)

        fig, axes = plt.subplots(1, len(classes), figsize=figsize, sharey=True)
        if len(classes) == 1:
            axes = [axes]

        x_positions = np.arange(n_k)
        x_labels = [label for label, _ in all_datasets]
        rng = np.random.default_rng(0)

        for ax, cls in zip(axes, classes):
            sample_mask = y_labels == cls

            # mean abundance per feature per k level
            feat_matrix = np.array([
                [X.loc[sample_mask, f].values.astype(float).mean()
                 for _, X in all_datasets]
                for f in shared_cols
            ])  # shape: (n_features, n_k)

            # --- Boxplot at each k ---
            ax.boxplot(
                [feat_matrix[:, k] for k in range(n_k)],
                positions=x_positions,
                widths=0.5,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color='black', linewidth=2, zorder=5),
                boxprops=dict(facecolor='#d5e8f7', alpha=0.5, zorder=1),
                whiskerprops=dict(linewidth=1, zorder=1),
                capprops=dict(linewidth=1, zorder=1),
            )

            # --- Lines + jittered dots per feature ---
            for f_idx, feat in enumerate(shared_cols):
                is_protected = feat in protected_features
                colour = '#c0392b' if is_protected else '#555555'
                alpha  = 0.9 if is_protected else 0.6
                lw     = 1.8 if is_protected else 0.7
                ms     = 25  if is_protected else 14
                zorder = 4   if is_protected else 3

                y_vals = feat_matrix[f_idx]
                jitter = rng.uniform(-0.18, 0.18, size=n_k)

                # line on exact x positions
                ax.plot(x_positions, y_vals, color=colour,
                        alpha=alpha * 0.5, linewidth=lw, zorder=zorder)
                # dots with jitter inside box
                ax.scatter(x_positions + jitter, y_vals, color=colour,
                           alpha=alpha, s=ms, zorder=zorder + 1, edgecolors='none')

            ax.set_yscale('symlog', linthresh=1e-4)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, rotation=30, ha='right', fontsize=15)
            ax.tick_params(axis='y', labelsize=15)
            ax.set_title(f"Class {cls}", fontsize=18, fontweight='bold')
            ax.set_xlabel("Perturbation level", fontsize=16)
            ax.grid(True, linestyle="--", alpha=0.3, axis='y')

            if ax == axes[0]:
                ax.set_ylabel("Mean feature abundance (symlog scale)", fontsize=16)
                ax.scatter([], [], color='#c0392b', s=60, label='protected feature')
                ax.scatter([], [], color='#555555', s=40, alpha=0.7, label='other feature')
                ax.legend(fontsize=15, frameon=True)

        if title:
            fig.suptitle(title, fontsize=18, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()