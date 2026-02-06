import torch
import numpy as np

from typing import Union


class ZeroInflatedDataGenerator:

    def __init__(self, n_samples: int, zero_inflation_prob: float, device: Union[int, str], num_causes: int):

        zero_inflation_prob = min(max(zero_inflation_prob, 0), 1)  #ensure to have zero_prob < 1 and > 0
        assert 0. <= zero_inflation_prob <= 1., "zero_inflation_prob is not a probability"

        self.n_samples = n_samples
        self.zero_prob = zero_inflation_prob if zero_inflation_prob is not None else torch.rand(1).item()
        self.device = device
        self.num_causes = num_causes

    def generate_bernoulli_data(self, shape_data):
        self.zero_prob = float(self.zero_prob)  # Ensure this is a float
        return torch.bernoulli(torch.full(shape_data, 1 - self.zero_prob, device=self.device))

    def generate_zil_data(self, mu, sigma):
        z = torch.randn((self.n_samples, 1, self.num_causes), device=self.device)
        xlog = torch.exp(mu + sigma * z).float()
        bernoulli_samples = self.generate_bernoulli_data(z.shape)
        causes = torch.where(bernoulli_samples == torch.tensor(0.), torch.tensor(0.), xlog)
        return causes

    def generate_zip_data(self, min_value: float=1, max_value: float=11):
        lambda_value = torch.FloatTensor(1).uniform_(min_value, max_value).item()
        poisson_data = torch.poisson(torch.full((self.n_samples, 1, self.num_causes), lambda_value)).to(self.device).float()
        bernoulli_samples = self.generate_bernoulli_data(poisson_data.shape)
        causes = torch.where(bernoulli_samples == torch.tensor(0.), torch.tensor(0.), poisson_data)
        return causes

    def generate_zinb_data(self, mu, theta):
        p = mu / (mu + theta)
        r = theta
        neg_binom_data = torch.distributions.NegativeBinomial(r, probs=1 - p).sample().to(self.device).float()
        bernoulli_samples = self.generate_bernoulli_data(neg_binom_data.shape)
        causes = torch.where(bernoulli_samples == torch.tensor(0.), torch.tensor(0.), neg_binom_data)
        return causes

    def generate_ziu_data(self):
        z = torch.rand((self.n_samples, 1, self.num_causes), device=self.device).float()
        bernoulli_samples = self.generate_bernoulli_data(z.shape)
        causes = torch.where(bernoulli_samples == torch.tensor(0.), torch.tensor(0.), z)
        return causes

