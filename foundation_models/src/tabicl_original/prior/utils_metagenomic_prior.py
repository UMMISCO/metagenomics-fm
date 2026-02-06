from __future__ import annotations

import random
import numpy as np
import pdb
import math

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
    --------------------
    Giulia's Distributions
    - Zero Inflated LogNormal Uniform
    - Poisson
    - Negative Binomial
    - Zipf without removing the mean (only positive values)

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
        samplers = {"zipf": self.sample_zipf, "poisson": self.sample_poisson, "zilnu": self.sample_zilnu, "normal": self.sample_normal_all, "mixed": self.sample_mixed, "uniform": self.sample_uniform}
        if self.sampling not in samplers:
            raise ValueError(f"Invalid sampling method: {self.sampling}")
        X, assignments = samplers[self.sampling]()
        # print(self.sampling, '-------------')

        if return_numpy:
            X = X.cpu().numpy()

        return X, assignments

    def sample_zilnu(self):
        """Generate Zero-Inflated LogNormal + Uniform distributed features (2D output)."""

        # Sample means & stds for all features
        causes_mean, causes_std = causes_sampler_f(self.num_features)
        causes_mean = torch.tensor(causes_mean, device=self.device).view(1, -1)  # shape (1, n_lognorm)
        causes_std = torch.tensor(causes_std, device=self.device).view(1, -1)

        # Determine how many features will be lognormal
        percentage_lognorm = torch.normal(mean=0.7, std=0.1, size=(1,)).item() * self.num_features
        n_lognorm = min(max(1, math.floor(percentage_lognorm)), self.num_features - 1)

        # Lognormal part
        percentage_zero_lognorm = torch.normal(mean=0.8, std=0.1, size=(1,)).item()
        generator_lognorm = ZeroInflatedDataGenerator(
            n_samples=self.seq_len,
            zero_inflation_prob=percentage_zero_lognorm,
            device=self.device,
            num_causes=n_lognorm
        )
        causes_zero_lognorm = generator_lognorm.generate_zil_data(
            causes_mean[:, :n_lognorm],
            causes_std[:, :n_lognorm]
        ).view(self.seq_len, -1)  # -> (seq_len, n_lognorm)

        # Uniform part
        percentage_zero_uniform = torch.normal(mean=0.2, std=0.1, size=(1,)).item()
        generator_uniform = ZeroInflatedDataGenerator(
            n_samples=self.seq_len,
            zero_inflation_prob=percentage_zero_uniform,
            device=self.device,
            num_causes=self.num_features - n_lognorm
        )
        causes_zero_uniform = generator_uniform.generate_ziu_data().view(self.seq_len, -1)  # -> (seq_len, n_uniform)

        # Concatenate along features axis (now dim=1)
        X = torch.cat((causes_zero_lognorm, causes_zero_uniform), dim=1)  # -> (seq_len, num_features)

        return X

    def sample_poisson(self, mu: float = 0.0, sigma: float = 1.0):
        """Generate Poisson-distributed features."""
        log_lambda = torch.normal(mean=mu, std=sigma, size=(self.seq_len, self.num_features), device=self.device)
        lam = torch.exp(log_lambda).clamp_max(1e6)
        x = torch.poisson(lam)
        assert (x < 0).sum() == 0
        return x

    def sample_normal_all(self):
        if self.pre_stats:
            X = torch.normal(self.means, self.stds.abs()).float()
        else:
            X = torch.normal(0.0, 1.0, (self.seq_len, self.num_features), device=self.device).float()
        return X

    def sample_uniform(self):
        """Generate uniformly distributed features."""
        return torch.rand((self.seq_len, self.num_features), device=self.device)

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
        # return x - x.mean()
        return x

    # def sample_mixed(self):
    #     """Generate features with mixed distributions."""
    #     X = []
    #     zipf_p, multi_p, normal_p = random.random() * 0.66, random.random() * 0.66, random.random() * 0.66
    #     for n in range(self.num_features):
    #         if random.random() > normal_p:
    #             x = self.sample_normal(n)
    #         elif random.random() > multi_p:
    #             x = self.sample_multinomial()
    #         elif random.random() > zipf_p:
    #             x = self.sample_zipf()
    #         else:
    #             x = torch.rand((self.seq_len,), device=self.device)
    #         X.append(x)
    #     # breakpoint()
    #     return torch.stack(X, -1)

    def sample_mixed(
            self,
            p_poisson=0.25,
            p_ziln=0.25,
            p_zipf=0.25,
            p_negbinom=0.25,
    ):
        """
        Generate features with FIXED proportions of:
           - Poisson
           - Zero-Inflated LogNormal
           - Zipf
           - Negative Binomial

        Guarantees all distributions appear if possible.
        If num_features < 4, selects a subset of distributions.

        RETURNS:
            X : (seq_len, num_features) tensor
            assignments : list[str] of length num_features
        """

        distributions = ["Poisson", "ZILN", "Zipf", "NegBinom"]
        probs = [p_poisson, p_ziln, p_zipf, p_negbinom]

        # -----------------------------
        # Handle small num_features
        # -----------------------------
        if self.num_features < 4:
            # pick num_features unique distributions
            chosen = np.random.choice(distributions, size=self.num_features, replace=False)
            counts = [1 if d in chosen else 0 for d in distributions]

        else:

            # Normalize probabilities
            probs = np.array([p_poisson, p_ziln, p_zipf, p_negbinom])
            probs = probs / probs.sum()

            # Expected fractional counts
            raw = probs * self.num_features

            # Floor all counts
            counts = np.floor(raw).astype(int)

            # Compute how many features are missing due to flooring
            missing = self.num_features - counts.sum()

            # Distribute remaining features based on largest fractional parts
            frac = raw - np.floor(raw)
            order = np.argsort(-frac)  # descending fractional values

            for i in order[:missing]:
                counts[i] += 1

        # -----------------------------
        # Build assignment list
        # -----------------------------
        assignments = []
        for dist, count in zip(distributions, counts):
            assignments += [dist] * count

        random.shuffle(assignments)
        # length = num_features guaranteed

        # -----------------------------
        # Generate features
        # -----------------------------
        X = []
        for dist in assignments:

            if dist == "Poisson":
                x = self.sample_poisson(mu=0.0, sigma=1.0)[:, 0]

            elif dist == "ZILN":
                n = 1
                means, stds = causes_sampler_f(n)
                means = torch.tensor(means, device=self.device).view(1, -1)
                stds = torch.tensor(stds, device=self.device).view(1, -1)
                prob = float(torch.normal(mean=0.8, std=0.1, size=(1,), device=self.device).clamp(0.0, 1.0))
                gen = ZeroInflatedDataGenerator(self.seq_len, prob, device=self.device, num_causes=n)
                x = gen.generate_zil_data(means, stds).view(self.seq_len)

            elif dist == "Zipf":
                x = self.sample_zipf()

            else:  # Negative Binomial
                r_param = 5
                p_param = random.uniform(0.3, 0.7)
                x = torch.tensor(
                    np.random.negative_binomial(r_param, p_param, size=self.seq_len),
                    device=self.device
                ).float()

            X.append(x)

        # Final shape (seq_len, num_features)
        X = torch.stack(X, dim=-1)

        assert X.shape[1] == self.num_features, \
            f"Expected {self.num_features}, got {X.shape[1]}"

        self.last_assignments = assignments
        return X, assignments


def causes_sampler_f(num_causes):
    mu, sigma = 0, 1
    means = np.random.normal(mu, sigma, (num_causes))
    std = np.abs(np.random.normal(mu, sigma, (num_causes)) * means)

    return means, std
