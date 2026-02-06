import math
import random

import numpy as np
import torch
from torch import nn

from mothernet.utils import default_device
from mothernet.distributions import parse_distributions, sample_distributions
from mothernet.config_utils import str2bool
from mothernet.priors.zero_inflated_priors import ZeroInflatedDataGenerator
from mothernet.priors.utils import min_max_scaler_torch

def normalize_rows_to_value(matrix, value: int=1):
    assert len(matrix.shape) == 2, f"The matrix has shape {matrix.shape}"
    # Ensure all values are positive by shifting if necessary
    min_val = matrix.min()
    if min_val < 0:
        matrix = matrix - min_val

    # Normalize each row so that the sum is approximately 100
    row_sums = matrix.sum(axis=1)
    normalized_matrix = (matrix.T / row_sums * value).T

    return normalized_matrix

class GaussianNoise(nn.Module):
    def __init__(self, std, device):
        super().__init__()
        self.std = std
        self.device = device

    def forward(self, x):
        return x + torch.normal(torch.zeros_like(x), self.std)


def causes_sampler_f(num_causes, study_condition: bool=False):

    if not study_condition:
        mu, sigma = 0, 1
    else:
        assert len(ARR_STDS_STUDY) ==  len(ARR_MEANS_STUDY), "Array means and std for study condition of different dimensions"
        index = np.random.randint(0, len(ARR_STDS_STUDY) - 1)
        mu = ARR_MEANS_STUDY[index]
        sigma = ARR_STDS_STUDY[index]

    # print(mu, sigma)
    means = np.random.normal(mu, sigma, (num_causes))
    std = np.abs(np.random.normal(mu, sigma, (num_causes)) * means)
    
    return means, std


def causes_sampler_f_nb(num_causes):
    mu_mean, sigma_mean = 1, 1  # Shifted to ensure positive means
    means = np.abs(np.random.normal(mu_mean, sigma_mean, num_causes))

    shape_param, scale_param = 2, 1  # These can be adjusted based on the expected overdispersion
    theta = np.random.gamma(shape_param, scale_param, num_causes)

    return means, theta


class MLP(torch.nn.Module):
    def __init__(self, device, num_features, num_outputs, n_samples, sampling, *, num_layers, prior_mlp_hidden_dim, prior_mlp_activations,normalize_by_rows, normalize_by_columns,
                 noise_std, y_is_effect, pre_sample_weights, prior_mlp_dropout_prob, pre_sample_causes, prior_mlp_scale_weights_sqrt, random_feature_rotation, add_uninformative_features,
                 is_causal, num_causes, block_wise_dropout, init_std, sort_features, in_clique):
        super(MLP, self).__init__()
        self.device = device
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.n_samples = n_samples
        self.sampling = sampling
        self.prior_mlp_scale_weights_sqrt = prior_mlp_scale_weights_sqrt
        self.random_feature_rotation = random_feature_rotation
        self.pre_sample_causes = pre_sample_causes
        self.add_uninformative_features = add_uninformative_features
        self.is_causal = is_causal
        self.num_causes = num_causes
        self.prior_mlp_hidden_dim = prior_mlp_hidden_dim
        self.prior_mlp_activations = prior_mlp_activations
        self.num_layers = num_layers
        self.noise_std = noise_std
        self.y_is_effect = y_is_effect
        self.pre_sample_weights = pre_sample_weights
        self.prior_mlp_dropout_prob = prior_mlp_dropout_prob
        self.block_wise_dropout = block_wise_dropout
        self.init_std = init_std
        self.sort_features = str2bool(sort_features) if isinstance(sort_features, str) else sort_features
        self.in_clique = in_clique
        self.normalize_by_rows = normalize_by_rows
        self.normalize_by_columns = normalize_by_columns
        # self.pre_sample_causes = self.is_causal if not is_causal else pre_sample_causes


        with torch.no_grad():
            assert (self.num_layers >= 2)
            if self.is_causal:
                self.prior_mlp_hidden_dim = max(self.prior_mlp_hidden_dim, num_outputs + 2 * num_features)
            else:
                self.num_causes = num_features

            # This means that the mean and standard deviation of each cause is determined in advance
            if self.pre_sample_causes:
                self.causes_mean, self.causes_std = causes_sampler_f(self.num_causes, study_condition=False)
                self.causes_mean = torch.tensor(self.causes_mean, device=device).unsqueeze(0).unsqueeze(0).tile(
                    (n_samples, 1, 1))
                self.causes_std = torch.tensor(self.causes_std, device=device).unsqueeze(0).unsqueeze(0).tile(
                    (n_samples, 1, 1))

                self.causes_mean_normal, self.causes_std_normal = causes_sampler_f(self.num_causes, study_condition=False)
                self.causes_mean_normal = torch.tensor(self.causes_mean_normal, device=device).unsqueeze(0).unsqueeze(0).tile(
                    (n_samples, 1, 1))
                self.causes_std_normal = torch.tensor(self.causes_std_normal, device=device).unsqueeze(0).unsqueeze(0).tile(
                    (n_samples, 1, 1))
                self.causes_mean_nb, self.causes_std_nb = causes_sampler_f_nb(self.num_causes)
                self.causes_mean_nb = torch.tensor(self.causes_mean_nb, device=device).unsqueeze(0).unsqueeze(0).tile((n_samples, 1, 1))
                self.causes_std_nb = torch.tensor(self.causes_std_nb, device=device).unsqueeze(0).unsqueeze(0).tile((n_samples, 1, 1))

            def generate_module(layer_idx, out_dim):
                # Determine std of each noise term in initialization, so that is shared in runs
                # torch.abs(torch.normal(torch.zeros((out_dim)), self.noise_std)) - Change std for each dimension?
                noise = (GaussianNoise(torch.abs(torch.normal(torch.zeros(size=(1, out_dim), device=device), float(self.noise_std))), device=device)
                         if self.pre_sample_weights else GaussianNoise(float(self.noise_std), device=device))
                return [
                    nn.Sequential(*[self.prior_mlp_activations(), nn.Linear(self.prior_mlp_hidden_dim, out_dim), noise])
                ]

            self.layers = [nn.Linear(self.num_causes, self.prior_mlp_hidden_dim, device=device)]
            self.layers += [module for layer_idx in range(self.num_layers-1) for module in generate_module(layer_idx, self.prior_mlp_hidden_dim)]
            if not self.is_causal:
                self.layers += generate_module(-1, num_outputs)
            self.layers = nn.Sequential(*self.layers)

            # Initialize Model parameters
            for i, (n, p) in enumerate(self.layers.named_parameters()):
                if self.block_wise_dropout:
                    if len(p.shape) == 2:  # Only apply to weight matrices and not bias
                        nn.init.zeros_(p)
                        # TODO: N blocks should be a setting
                        n_blocks = random.randint(1, math.ceil(math.sqrt(min(p.shape[0], p.shape[1]))))
                        w, h = p.shape[0] // n_blocks, p.shape[1] // n_blocks
                        keep_prob = (n_blocks*w*h) / p.numel()
                        for block in range(0, n_blocks):
                            nn.init.normal_(p[w * block: w * (block+1), h * block: h * (block+1)], std=self.init_std /
                                            keep_prob**(1/2 if self.prior_mlp_scale_weights_sqrt else 1))
                else:
                    if len(p.shape) == 2:  # Only apply to weight matrices and not bias
                        dropout_prob = self.prior_mlp_dropout_prob if i > 0 else 0.0  # Don't apply dropout in first layer
                        dropout_prob = min(dropout_prob, 0.99)
                        nn.init.normal_(p, std=self.init_std / (1. - dropout_prob**(1/2 if self.prior_mlp_scale_weights_sqrt else 1)))
                        p *= torch.bernoulli(torch.zeros_like(p) + 1. - dropout_prob)

    def forward(self):
        n_samples = self.n_samples
        device = self.device
        num_outputs = self.num_outputs
        num_features = self.num_features

        def sample_normal():
            if self.pre_sample_causes:
                # print(self.causes_mean_normal.shape, self.causes_std_normal.shape, self.num_causes, self.num_features)
                # exit()
                causes = torch.normal(self.causes_mean_normal, self.causes_std_normal.abs()).float()
            else:
                causes = torch.normal(0., 1., (n_samples, 1, self.num_causes), device=device).float()
            return causes

        if self.sampling == 'normal':
            causes = sample_normal()
        elif self.sampling == 'mixed':
            # zipf_p, multi_p, normal_p = random.random() * 0.66, random.random() * 0.66, random.random() * 0.66
            multi_p, normal_p = random.random() * 0.66, random.random() * 0.66

            def sample_cause(n):
                if random.random() > normal_p:
                    if self.pre_sample_causes:
                        return torch.normal(self.causes_mean_normal[:, :, n], self.causes_std_normal[:, :, n].abs()).float()
                    else:
                        return torch.normal(0., 1., (n_samples, 1), device=device).float()
                elif random.random() > multi_p:
                    x = torch.multinomial(torch.rand((random.randint(2, 10))), n_samples, replacement=True).to(device).unsqueeze(-1).float()
                    x = (x - torch.mean(x)) / torch.std(x)
                    return x
                else:
                    x = torch.minimum(torch.tensor(np.random.zipf(2.0 + random.random() * 2, size=(n_samples)),
                                                   device=device).unsqueeze(-1).float(), torch.tensor(10.0, device=device))
                    return x - torch.mean(x)
            causes = torch.cat([sample_cause(n).unsqueeze(-1) for n in range(self.num_causes)], -1)
        elif self.sampling == 'uniform':
            causes = torch.rand((n_samples, 1, self.num_causes), device=device)
        elif self.sampling == 'lognorm':
            if self.pre_sample_causes:
                mu = self.causes_mean
                sigma = self.causes_std.abs()
            else:
                mu = 1.0279785596707818  # Mean of the underlying normal distribution 1.0279785596707818 3.261679481125489
                sigma = 3.261679481125489
            # print(mu, sigma)
            z = torch.randn((n_samples, 1, self.num_causes), device=device)
            causes = torch.exp(mu + sigma * z).float()
        elif self.sampling == 'ziln':
            if self.pre_sample_causes:
                mu = self.causes_mean
                sigma = self.causes_std.abs()
            else:
                mu = 1.0279785596707818  # Mean of the underlying normal distribution 1.0279785596707818 3.261679481125489
                sigma = 3.261679481125489
            generator = ZeroInflatedDataGenerator(n_samples=n_samples, zero_inflation_prob=torch.rand(1).item(), device=device, num_causes=self.num_causes)
            causes = generator.generate_zil_data(mu, sigma)

        elif self.sampling == 'zinb':
            if self.pre_sample_causes:
                mu = self.causes_mean_nb
                sigma = self.causes_std_nb
            else:
                mu = 1.0279785596707818  # Mean of the underlying normal distribution 1.0279785596707818 3.261679481125489
                sigma = 3.261679481125489
            generator = ZeroInflatedDataGenerator(n_samples=n_samples, zero_inflation_prob=torch.rand(1).item(), device=device, num_causes=self.num_causes)
            causes = generator.generate_zinb_data(mu, sigma)
        elif self.sampling == 'zip':
            generator = ZeroInflatedDataGenerator(n_samples=n_samples, zero_inflation_prob=torch.rand(1).item(), device=device, num_causes=self.num_causes)
            causes = generator.generate_zip_data()
        elif self.sampling == 'log-uniform':
            percentage_lognorm = torch.normal(mean=.7, std=.1, size=(1,)).item() * self.num_causes
            percentage_lognorm = min(math.floor(percentage_lognorm), self.num_causes)  # within valid range (?)
            causes_mean_lognorm = self.causes_mean[:, :,  :percentage_lognorm]
            causes_std_lognorm = self.causes_std[:, :,  :percentage_lognorm]
            # if math.floor(percentage_lognorm) > self.num_causes:
            #     percentage_lognorm = percentage_lognorm - 1
            z = torch.randn((n_samples, 1, percentage_lognorm), device=device)
            # print(causes_mean_lognorm.shape, self.num_causes, z.shape, percentage_lognorm)
            causes_lognorm = torch.exp(causes_mean_lognorm + causes_std_lognorm * z).float()
            causes_uniform = torch.rand((n_samples, 1, self.num_causes - percentage_lognorm), device=device).float()
            # print(causes_lognorm.shape, causes_uniform.shape)
            causes = torch.cat((causes_lognorm, causes_uniform), dim=2)

        elif self.sampling == 'zero-log-uniform':

            percentage_lognorm = torch.normal(mean=.7, std=.1, size=(1,)).item() * self.num_causes
            if math.floor(percentage_lognorm) > self.num_causes:
                percentage_lognorm = self.num_causes - 1
            causes_mean_lognorm = self.causes_mean[:, :, :math.floor(percentage_lognorm)]
            causes_std_lognorm = self.causes_std[:, :, :math.floor(percentage_lognorm)]
            percentage_zero_lognorm = torch.normal(mean=.8, std=.1, size=(1,)).item()
            generator_lognorm = ZeroInflatedDataGenerator(n_samples=n_samples,
                                                          zero_inflation_prob=percentage_zero_lognorm, device=device,
                                                          num_causes=causes_mean_lognorm.shape[-1])
            # assert causes_mean_lognorm.shape[-1] == causes_std_lognorm.shape[-1] == math.floor(percentage_lognorm), "The length are different"

            causes_zero_lognorm = generator_lognorm.generate_zil_data(causes_mean_lognorm, causes_std_lognorm)

            percentage_zero_uniform = torch.normal(mean=.2, std=.1, size=(1,)).item()
            generator_uniform = ZeroInflatedDataGenerator(n_samples=n_samples,
                                                          zero_inflation_prob=percentage_zero_uniform, device=device,
                                                          num_causes=self.num_causes - math.floor(percentage_lognorm))
            causes_zero_uniform = generator_uniform.generate_ziu_data()

            causes = torch.cat((causes_zero_lognorm, causes_zero_uniform), dim=2)


        else:
            raise ValueError(f'Sampling is set to invalid setting: {self.sampling}.')

        outputs = [causes]
        for layer in self.layers:
            outputs.append(layer(outputs[-1]))
        outputs = outputs[2:]

        if self.is_causal:
            # Sample nodes from graph if model is causal
            outputs_flat = torch.cat(outputs, -1)

            if self.in_clique:
                random_perm = random.randint(0, outputs_flat.shape[-1] - num_outputs - num_features) + \
                    torch.randperm(num_outputs + num_features, device=device)
            else:
                random_perm = torch.randperm(outputs_flat.shape[-1]-1, device=device)

            random_idx_y = list(range(-num_outputs, -0)) if self.y_is_effect else random_perm[0:num_outputs]
            random_idx = random_perm[num_outputs:num_outputs + num_features]

            if self.sort_features:
                random_idx, _ = torch.sort(random_idx)
            y = outputs_flat[:, :, random_idx_y]

            x = outputs_flat[:, :, random_idx]
        else:
            y = outputs[-1][:, :, :]
            x = causes

        if bool(torch.any(torch.isnan(x)).detach().cpu().numpy()) or bool(torch.any(torch.isnan(y)).detach().cpu().numpy()):
            print('Nan caught in MLP model x:', torch.isnan(x).sum(), ' y:', torch.isnan(y).sum())

            x[:] = 0.0
            y[:] = -100  # default ignore index for CE

        # random feature rotation
        if self.random_feature_rotation:
            x = x[..., (torch.arange(x.shape[-1], device=device)+random.randrange(x.shape[-1])) % x.shape[-1]]

        if self.add_uninformative_features and random.random() < 0.5:
            bounce = random.randint(1, num_features)
            n_uninformative = random.randint(0, bounce)
            if n_uninformative > 0:
                # we pick the last couple to be uninformative; since we shuffle anyway it doesn't matter
                x_uninformative = x[:, :, -n_uninformative:]
                shuffle_indices = torch.cat([torch.randperm(n_samples, device=device).unsqueeze(1).unsqueeze(2) for _ in range(n_uninformative)], 2)
                x_uninformative = torch.gather(x_uninformative, 0, shuffle_indices)
                x[:, :, -n_uninformative:] = x_uninformative
                x = x[:, :, torch.randperm(num_features, device=device)]

        if self.normalize_by_rows:
            x = normalize_rows_to_value(x.squeeze()).unsqueeze(1)

        elif self.normalize_by_columns:
            x = min_max_scaler_torch(x)
        return x, y


class MLPPrior:
    def __init__(self, config=None):
        self.config = parse_distributions(config or {})

    def get_batch(self, batch_size, n_samples, num_features, device=default_device, num_outputs=1, epoch=None, single_eval_pos=None):
        sample = [MLP(device, num_features, num_outputs, n_samples, **sample_distributions(self.config)).to(device)() for _ in range(0, batch_size)]
        x, y = zip(*sample)

        y = torch.cat(y, 1).detach().squeeze(2)
        x = torch.cat(x, 1).detach()
        # print(x.shape, y.shape)
        return x, y, y
