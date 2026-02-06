from __future__ import annotations

import random
import numpy as np

import torch
from torch import nn

from .zero_inflated_priors import ZeroInflatedDataGenerator

class GaussianNoise(nn.Module):
    def __init__(self, std):
        super().__init__()
        self.std = std

    def forward(self, X):
        return X + torch.normal(torch.zeros_like(X), self.std)


class XSampler:
    """Input sampler for generating features for prior datasets.

    Supports multiple feature distribution types:
    - Normal: Standard normal distribution
    - Multinomial: Categorical features with random number of categories
    - Zipf: Power law distributed features
    - Mixed: Random combination of the above

    Parameters
    ----------
    seq_len : int
        Length of sequence to generate

    num_features : int
        Number of features to generate

    pre_stats : bool
        Whether to pre-generate statistics for the input features

    sampling : str, default='mixed'
        Feature sampling strategy ('normal', 'mixed', 'uniform')

    device : str, default='cpu'
        Device to store tensors on
    """

    def __init__(self, seq_len, num_features, pre_stats=False, sampling="mixed", device="cpu"):
        self.seq_len = seq_len
        self.num_features = num_features
        self.pre_stats = pre_stats
        self.sampling = sampling
        self.device = device

        if pre_stats:
            self._pre_stats()

    def _pre_stats(self):
        means = np.random.normal(0, 1, self.num_features)
        stds = np.abs(np.random.normal(0, 1, self.num_features) * means)
        self.means = torch.tensor(means, dtype=torch.float, device=self.device).unsqueeze(0).repeat(self.seq_len, 1)
        self.stds = torch.tensor(stds, dtype=torch.float, device=self.device).unsqueeze(0).repeat(self.seq_len, 1)

    def sample(self, return_numpy=False):
        """Generate features according to the specified sampling strategy.

        Returns
        -------
        X : torch.Tensor
            Generated features of shape (seq_len, num_features)
        """
        samplers = {"normal": self.sample_normal_all, "mixed": self.sample_mixed, "uniform": self.sample_uniform}
        if self.sampling not in samplers:
            raise ValueError(f"Invalid sampling method: {self.sampling}")
        X = samplers[self.sampling]()

        return X.cpu().numpy() if return_numpy else X

    def sample_normal_all(self):
        if self.pre_stats:
            X = torch.normal(self.means, self.stds.abs()).float()
        else:
            X = torch.normal(0.0, 1.0, (self.seq_len, self.num_features), device=self.device).float()
        return X

    # #with lognormal instead of normal distribution
    # def sample_normal_all(self):
    #     """Generate Zero-Inflated LogNormal–distributed features for ALL features."""
    #
    #     # Sample means & stds for all features
    #     causes_mean, causes_std = causes_sampler_f(self.num_features)
    #     causes_mean = torch.tensor(causes_mean, device=self.device).view(1, -1)
    #     causes_std = torch.tensor(causes_std, device=self.device).view(1, -1)
    #
    #     # Zero-inflation
    #     percentage_zero_lognorm = torch.normal(mean=0.8, std=0.1, size=(1,), device=self.device).item()
    #
    #     generator_lognorm = ZeroInflatedDataGenerator(
    #         n_samples=self.seq_len,
    #         zero_inflation_prob=percentage_zero_lognorm,
    #         device=self.device,
    #         num_causes=self.num_features
    #     )
    #
    #     # Output: (seq_len, num_features)
    #     X = generator_lognorm.generate_zil_data(
    #         causes_mean, causes_std
    #     ).view(self.seq_len, self.num_features)
    #
    #     return X.float()

    def sample_uniform(self):
        """Generate uniformly distributed features."""
        return torch.rand((self.seq_len, self.num_features), device=self.device)


    # # with lognormal instead of normal distribution
    # def sample_normal(self, n=None):
    #     """Generate a single Zero-Inflated LogNormal feature."""
    #
    #     # Sample one feature mean & std
    #     causes_mean, causes_std = causes_sampler_f(1)
    #     causes_mean = torch.tensor(causes_mean, device=self.device).view(1, 1)
    #     causes_std = torch.tensor(causes_std, device=self.device).view(1, 1)
    #
    #     percentage_zero_lognorm = torch.normal(mean=0.8, std=0.1, size=(1,), device=self.device).item()
    #
    #     generator_lognorm = ZeroInflatedDataGenerator(
    #         n_samples=self.seq_len,
    #         zero_inflation_prob=percentage_zero_lognorm,
    #         device=self.device,
    #         num_causes=1
    #     )
    #
    #     # Output shape: (seq_len, 1)
    #     X = generator_lognorm.generate_zil_data(
    #         causes_mean, causes_std
    #     ).view(self.seq_len)
    #
    #     return X.float()

    def sample_normal(self, n=None):
        """Generate normally distributed features.

        Parameters
        ----------
        n : int
            Index of the feature to generate
        """

        if self.pre_stats:
            return torch.normal(self.means[:, n], self.stds[:, n].abs()).float()
        else:
            return torch.normal(0.0, 1.0, (self.seq_len,), device=self.device).float()

    def sample_multinomial(self):
        """Generate categorical features."""
        n_categories = random.randint(2, 20)
        probs = torch.rand(n_categories, device=self.device)
        x = torch.multinomial(probs, self.seq_len, replacement=True)
        x = x.float()
        return (x - x.mean()) / x.std()

    def sample_zipf(self):
        """Generate Zipf-distributed features."""
        x = np.random.zipf(2.0 + random.random() * 2, (self.seq_len,))
        x = torch.tensor(x, device=self.device).clamp(max=10)
        x = x.float()
        return x - x.mean()

    def sample_mixed(self):
        """Generate features with mixed distributions."""
        X = []
        zipf_p, multi_p, normal_p = random.random() * 0.66, random.random() * 0.66, random.random() * 0.66
        for n in range(self.num_features):
            if random.random() > normal_p:
                x = self.sample_normal(n)
            elif random.random() > multi_p:
                x = self.sample_multinomial()
            elif random.random() > zipf_p:
                x = self.sample_zipf()
            else:
                x = torch.rand((self.seq_len,), device=self.device)
            X.append(x)
        return torch.stack(X, -1)

    # def sample_mixed(
    #         self,
    #         p_normal=0.25,
    #         p_multinomial=0.25,
    #         p_zipf=0.25,
    #         p_uniform=0.25,
    # ):
    #     """
    #     Generate features with mixed distributions using *exact proportional allocation*.
    #
    #     Uses the original distributions:
    #         - Normal
    #         - Multinomial
    #         - Zipf
    #         - Uniform
    #
    #     Guarantees each distribution appears at least once when num_features >= 4.
    #
    #     Returns:
    #         X : (seq_len, num_features)
    #         assignments : list[str]
    #     """
    #
    #     distributions = ["normal", "multinomial", "zipf", "uniform"]
    #     probs = np.array([p_normal, p_multinomial, p_zipf, p_uniform], dtype=float)
    #
    #     # ============================================================
    #     # CASE 1 — fewer features than distributions
    #     # ============================================================
    #     if self.num_features < 4:
    #         assignments = []
    #         for _ in range(self.num_features):
    #             r = random.random()
    #             if r < p_normal:
    #                 assignments.append("normal")
    #             elif r < p_normal + p_multinomial:
    #                 assignments.append("multinomial")
    #             elif r < p_normal + p_multinomial + p_zipf:
    #                 assignments.append("zipf")
    #             else:
    #                 assignments.append("uniform")
    #
    #         X = []
    #         for dist in assignments:
    #             if dist == "normal":
    #                 x = self.sample_normal(0)
    #             elif dist == "multinomial":
    #                 x = self.sample_multinomial()
    #             elif dist == "zipf":
    #                 x = self.sample_zipf()
    #             else:  # uniform
    #                 x = torch.rand((self.seq_len,), device=self.device)
    #             X.append(x)
    #
    #         return torch.stack(X, dim=-1), assignments
    #
    #     # ============================================================
    #     # CASE 2 — num_features >= 4 → proportional allocation
    #     # ============================================================
    #
    #     # Normalize
    #     probs /= probs.sum()
    #
    #     # Raw fractional counts
    #     raw = probs * self.num_features
    #     counts = np.floor(raw).astype(int)
    #
    #     # Ensure minimum of 1 each
    #     for i in range(4):
    #         if counts[i] == 0:
    #             counts[i] = 1
    #
    #     # Fix if too large
    #     while counts.sum() > self.num_features:
    #         idx = np.argmax(counts)
    #         if counts[idx] > 1:
    #             counts[idx] -= 1
    #         else:
    #             break
    #
    #     # Fix if too small
    #     while counts.sum() < self.num_features:
    #         frac = raw - np.floor(raw)
    #         idx = np.argmax(frac)
    #         counts[idx] += 1
    #
    #     # ============================================================
    #     # Build assignment list
    #     # ============================================================
    #     assignments = []
    #     for dist, c in zip(distributions, counts):
    #         assignments += [dist] * c
    #
    #     random.shuffle(assignments)
    #
    #     # ============================================================
    #     # Generate actual features
    #     # ============================================================
    #     X = []
    #     for i, dist in enumerate(assignments):
    #
    #         if dist == "normal":
    #             x = self.sample_normal(i)
    #
    #         elif dist == "multinomial":
    #             x = self.sample_multinomial()
    #
    #         elif dist == "zipf":
    #             x = self.sample_zipf()
    #
    #         else:  # uniform
    #             x = torch.rand((self.seq_len,), device=self.device)
    #
    #         X.append(x)
    #
    #     X = torch.stack(X, dim=-1)
    #     self.last_assignments = assignments
    #
    #     return X, assignments


def causes_sampler_f(num_causes):
    mu, sigma = 0, 1
    means = np.random.normal(mu, sigma, (num_causes))
    std = np.abs(np.random.normal(mu, sigma, (num_causes)) * means)

    return means, std
