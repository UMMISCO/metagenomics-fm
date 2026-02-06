#%%
import torch
from typing import Union
import random

#--------------------------------------
# Zero-Inflated Data Generator
#--------------------------------------
class ZeroInflatedDataGenerator:

    def __init__(self, n_samples, zero_inflation_prob, device='cpu', num_causes=1):
        self.n_samples = n_samples
        self.zero_prob = min(max(zero_inflation_prob, 0), 1)
        self.device = device
        self.num_causes = num_causes

    def generate_bernoulli_data(self, shape_data):
        return torch.bernoulli(torch.full(shape_data, 1 - self.zero_prob, device=self.device))

    def generate_zil_data(self, mu, sigma):
        z = torch.randn((self.n_samples, self.num_causes), device=self.device)
        xlog = torch.exp(mu + sigma * z).float()
        bern = self.generate_bernoulli_data(z.shape)
        return torch.where(bern == 0, torch.tensor(0.0, device=self.device), xlog)

#--------------------------------------
# Causes sampler (for ZILN)
#--------------------------------------
def causes_sampler_f(num_causes):
    mu, sigma = 0, 1
    means = np.random.normal(mu, sigma, num_causes)
    stds = np.abs(np.random.normal(mu, sigma, num_causes) * means)
    return means, stds

#--------------------------------------
# Poisson sampler (standalone)
#--------------------------------------
def sample_poisson(seq_len, num_features, mu=0.0, sigma=1.0, device='cpu'):
    log_lambda = torch.normal(mean=mu, std=sigma, size=(seq_len, num_features), device=device)
    lam = torch.exp(log_lambda).clamp_max(1e6)
    x = torch.poisson(lam)
    return x

#--------------------------------------
# Zipf sampler (standalone)
#--------------------------------------
def sample_zipf(seq_len, device='cpu'):
    a = 2.0 + random.random() * 2.0
    x = np.random.zipf(a, size=seq_len)
    x = torch.tensor(x, device=device).float().clamp(max=10)
    return x.float()

#--------------------------------------
# Mixed sampler
#--------------------------------------
def sample_mixed(
    seq_len=1024,
    num_features=100,
    device='cpu',
    p_poisson=0.25,
    p_ziln=0.25,
    p_zipf=0.25,
    p_negbinom=0.25,
):
    """
    Generate a dataset containing all 4 distributions with fixed proportions.
    Ensures:
      - Poisson, ZILN, Zipf, NegBinom all appear
      - Proportions match provided percentages (rounded)
    """

    # Normalize percentages
    total = p_poisson + p_ziln + p_zipf + p_negbinom
    p_poisson /= total
    p_ziln /= total
    p_zipf /= total
    p_negbinom /= total

    # Determine number of features
    n_poisson   = max(1, int(round(num_features * p_poisson)))
    n_ziln      = max(1, int(round(num_features * p_ziln)))
    n_zipf      = max(1, int(round(num_features * p_zipf)))
    n_negbinom  = max(1, int(round(num_features * p_negbinom)))

    # Adjust rounding to match exact feature count
    total_assigned = n_poisson + n_ziln + n_zipf + n_negbinom
    diff = total_assigned - num_features

    if diff > 0:
        # remove from largest block
        counts = [n_poisson, n_ziln, n_zipf, n_negbinom]
        i = np.argmax(counts)
        counts[i] -= diff
        n_poisson, n_ziln, n_zipf, n_negbinom = counts

    elif diff < 0:
        # add to smallest block
        counts = [n_poisson, n_ziln, n_zipf, n_negbinom]
        i = np.argmin(counts)
        counts[i] += abs(diff)
        n_poisson, n_ziln, n_zipf, n_negbinom = counts

    # Build feature-type assignment
    assignments = (
        ["Poisson"]   * n_poisson +
        ["ZILN"]      * n_ziln +
        ["Zipf"]      * n_zipf +
        ["NegBinom"]  * n_negbinom
    )
    random.shuffle(assignments)

    # Now generate the features
    X = []
    labels = []

    for dist in assignments:
        if dist == "Poisson":
            x = sample_poisson(seq_len, 1, device=device)[:, 0]

        elif dist == "ZILN":
            means, stds = causes_sampler_f(1)
            means = torch.tensor(means, device=device).view(1, -1)
            stds = torch.tensor(stds, device=device).view(1, -1)
            prob = float(torch.normal(mean=0.8, std=0.1, size=(1,)).clamp(0.0, 1.0))
            gen = ZeroInflatedDataGenerator(seq_len, prob, device=device, num_causes=1)
            x = gen.generate_zil_data(means, stds).view(seq_len)

        elif dist == "Zipf":
            x = sample_zipf(seq_len, device=device)

        elif dist == "NegBinom":
            r_param = 5
            p_param = random.uniform(0.3, 0.7)
            x = torch.tensor(
                np.random.negative_binomial(r_param, p_param, size=seq_len),
                device=device
            ).float()

        X.append(x)
        labels.append(dist)

    X = torch.stack(X, dim=-1)
    return X, labels


#%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

seq_len = 1024
num_features = 50

# Sample dataset
X, labels = sample_mixed(seq_len=seq_len, num_features=num_features)
X_np = X.numpy()

# Plot histograms grouped by distribution
unique_labels = list(set(labels))
plt.figure(figsize=(12, 8))

for i, dist in enumerate(unique_labels):
    idxs = [j for j, l in enumerate(labels) if l == dist]
    plt.subplot(2, 2, i+1)
    for j in idxs:
        plt.hist(X_np[:, j], bins=30, alpha=0.4, label=f"Feature {j}")
    plt.title(f"{dist} Features")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
