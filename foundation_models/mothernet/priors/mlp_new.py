import math
import random
import pdb
import numpy as np
import torch
from torch import nn

from mothernet.utils import default_device
from mothernet.distributions import parse_distributions, sample_distributions
from mothernet.config_utils import str2bool
from mothernet.priors.zero_inflated_priors import ZeroInflatedDataGenerator
from mothernet.priors.utils import min_max_scaler_torch, normalize_rows_to_value

class GaussianNoise(nn.Module):
    def __init__(self, std, device):
        super().__init__()
        self.std = std
        self.device = device

    def forward(self, x):
        return x + torch.normal(torch.zeros_like(x), self.std)

def causes_sampler_f(num_causes):

    mu, sigma = 0, 1
    means = np.random.normal(mu, sigma, num_causes)
    std = np.abs(np.random.normal(mu, sigma, num_causes) * means)
    
    return means, std

def causes_sampler_f_nb(num_causes):
    mu_mean, sigma_mean = 1, 1  # Shifted to ensure positive means
    means = np.abs(np.random.normal(mu_mean, sigma_mean, num_causes))

    shape_param, scale_param = 2, 1  # These can be adjusted based on the expected over-dispersion
    theta = np.random.gamma(shape_param, scale_param, num_causes)

    return means, theta

def tensor_transformation(mean, std, samples, device):
    mean = torch.tensor(mean, device=device).unsqueeze(0).unsqueeze(0).tile(
        (samples, 1, 1))
    std = torch.tensor(std, device=device).unsqueeze(0).unsqueeze(0).tile(
        (samples, 1, 1))
    return mean, std

def sampling_perc_from_gamma(probs: list, num_causes):
    assert sum(probs) <= 1.0

    k=100
    alpha = [p * k for p in probs]
    beta = [(1-p) * k for p in probs]
    perc_neg_binomial = torch.distributions.Beta(alpha[0], beta[0]).sample().item() * num_causes
    perc_log_norm = torch.distributions.Beta(alpha[1], beta[1]).sample().item() * num_causes
    perc_uniform = torch.distributions.Beta(alpha[2], beta[2]).sample().item() * num_causes

    perc_neg_binomial = max(0, min(math.floor(perc_neg_binomial), num_causes))
    perc_log_norm = max(0, min(math.floor(perc_log_norm), num_causes))
    perc_uniform = max(0, min(math.floor(perc_uniform), num_causes))
    total_perc = perc_neg_binomial + perc_log_norm + perc_uniform

    # Scale down if total > num_causes
    if total_perc > num_causes:
        scaling_factor = num_causes / total_perc
        perc_neg_binomial = math.floor(perc_neg_binomial * scaling_factor)
        perc_log_norm = math.floor(perc_log_norm * scaling_factor)
        perc_uniform = math.floor(perc_uniform * scaling_factor)

    perc_poisson = max(0 , num_causes - (perc_neg_binomial + perc_log_norm + perc_uniform))

    return perc_neg_binomial, perc_log_norm, perc_uniform, perc_poisson


def generator_x(samples, p, device, perc, causes_mean, causes_std, generator_class, method_name, instance):
    """Args:
        instance: bool
        samples (int): Number of samples to generate.
        p (float): Zero inflation probability (or any relevant probability parameter).
        device (str): Device to run the computation.
        perc (float): Proportion or percentage used to determine the number of causes.
        causes_mean (torch.Tensor): Mean tensor for the causes.
        causes_std (torch.Tensor): Standard deviation tensor for the causes.
        generator_class (class): A class used to generate the data (e.g. ZeroInflatedDataGenerator).
        method_name (str): Name of the method to call on the class to generate data (e.g., 'generate_zil_data').
    Returns:
        torch.Tensor: Generated data using the specified class and method."""
    generator_instance = generator_class(n_samples=samples, zero_inflation_prob=p, device=device,
                                         num_causes=math.floor(perc))
    if instance:
        causes_mean = causes_mean[:, :, :math.floor(perc)]
        causes_std = causes_std[:, :, :math.floor(perc)]
        generate_method = getattr(generator_instance, method_name)
        causes_distribution = generate_method(causes_mean, causes_std)
    else:
        generate_method = getattr(generator_instance, method_name)
        causes_distribution = generate_method()
    return causes_distribution


class MLP(torch.nn.Module):
    def __init__(self, device, num_features, num_outputs, n_samples, sampling, *, num_layers, prior_mlp_hidden_dim, prior_mlp_activations, normalize_by_rows, normalize_by_columns,
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
        # print(self.num_features, 'num_features')

        with torch.no_grad():
            assert (self.num_layers >= 2)
            if self.is_causal:
                self.prior_mlp_hidden_dim = max(self.prior_mlp_hidden_dim, num_outputs + 2 * num_features)
            else:
                self.num_causes = num_features

            # This means that the mean and standard deviation of each cause is determined in advance
            if self.pre_sample_causes:

                self.causes_mean, self.causes_std = causes_sampler_f(self.num_causes)
                self.causes_mean, self.causes_std = tensor_transformation(self.causes_mean, self.causes_std,n_samples, device)

                self.causes_mean_normal, self.causes_std_normal = causes_sampler_f(self.num_causes)
                self.causes_mean_normal, self.causes_std_normal = tensor_transformation(self.causes_mean_normal, self.causes_std_normal, n_samples, device)

                self.causes_mean_nb, self.causes_std_nb = causes_sampler_f_nb(self.num_causes)
                self.causes_mean_nb, self.causes_std_nb = tensor_transformation(self.causes_mean_nb, self.causes_std_nb, n_samples, device)

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
                causes = torch.normal(self.causes_mean_normal, self.causes_std_normal.abs()).float()
            else:
                causes = torch.normal(0., 1., (n_samples, 1, self.num_causes), device=device).float()
            return causes

        if self.sampling == 'normal':
            causes = sample_normal()

        elif self.sampling == 'zero-log-uniform':
            percentage_lognorm = math.floor(torch.normal(mean=.7, std=.1, size=(1,)).item() * self.num_causes)
            if percentage_lognorm > self.num_causes:
                percentage_lognorm = self.num_causes - 1
            causes_mean_lognorm = self.causes_mean[:, :, :percentage_lognorm]
            causes_std_lognorm = self.causes_std[:, :, :percentage_lognorm]
            percentage_zero_lognorm = torch.normal(mean=.8, std=.1, size=(1,)).item()
            percentage_zero_lognorm = percentage_zero_lognorm if percentage_zero_lognorm <= 1 else 0.99
            generator_lognorm = ZeroInflatedDataGenerator(n_samples=n_samples,
                                                          zero_inflation_prob=percentage_zero_lognorm, device=device,
                                                          num_causes=percentage_lognorm)
            causes_zero_lognorm = generator_lognorm.generate_zil_data(causes_mean_lognorm, causes_std_lognorm)

            percentage_zero_uniform = torch.normal(mean=.2, std=.1, size=(1,)).item()
            percentage_zero_uniform = percentage_zero_uniform if percentage_zero_uniform > 0 else 0.1
            generator_uniform = ZeroInflatedDataGenerator(n_samples=n_samples,
                                                          zero_inflation_prob=percentage_zero_uniform, device=device,
                                                          num_causes=self.num_causes - percentage_lognorm)
            causes_zero_uniform = generator_uniform.generate_ziu_data()
            causes = torch.cat((causes_zero_lognorm, causes_zero_uniform), dim=2)

        elif self.sampling == 'log-uniform-neg-poisson':
            probs = [0.45, 0.35, 0.10]   #Neg_binomoal, Log_norm, Uniform percentage
            perc_neg_binomial, perc_log_norm, perc_uniform, perc_poisson = sampling_perc_from_gamma(probs, self.num_causes)
            causes_log_norm = generator_x(samples=n_samples, p=.0, device=device, perc=perc_log_norm, causes_mean=self.causes_mean, causes_std=self.causes_std, generator_class=ZeroInflatedDataGenerator, method_name='generate_zil_data', instance=True)
            causes_neg_bin = generator_x(samples=n_samples, p=.0, device=device, perc=perc_neg_binomial, causes_mean=self.causes_mean_nb, causes_std=self.causes_std_nb, generator_class=ZeroInflatedDataGenerator, method_name='generate_zinb_data', instance=True)
            causes_uniform = generator_x(samples=n_samples, p=.0, device=device, perc=perc_uniform, causes_mean=self.causes_mean, causes_std=self.causes_std, generator_class=ZeroInflatedDataGenerator, method_name='generate_ziu_data', instance=False)
            causes_poisson = generator_x(samples=n_samples, p=.0, device=device, perc=perc_poisson, causes_mean=self.causes_mean, causes_std=self.causes_std, generator_class=ZeroInflatedDataGenerator, method_name='generate_zip_data', instance=False)
            # print(causes_log_norm.shape, causes_poisson.shape, causes_neg_bin.shape, causes_uniform.shape, 'ln-p-nb-u')
            causes = torch.cat((causes_log_norm, causes_neg_bin, causes_uniform, causes_poisson), dim=2)

        elif self.sampling == 'zero-log-uniform-neg-poisson':

            probs = [0.45, 0.35, 0.10]   #Neg_binomoal, Log_norm, Uniform percentage
            perc_neg_binomial, perc_log_norm, perc_uniform, perc_poisson = sampling_perc_from_gamma(probs, self.num_causes)
            perc_zero_ln_bn = torch.normal(mean=.8, std=.1, size=(1,)).item()
            # print(perc_zero_ln_bn, perc_log_norm)
            perc_zero_u_p = torch.normal(mean=.2, std=.1, size=(1,)).item()
            causes_log_norm = generator_x(samples=n_samples, p=perc_zero_ln_bn, device=device, perc=perc_log_norm, causes_mean=self.causes_mean, causes_std=self.causes_std, generator_class=ZeroInflatedDataGenerator, method_name='generate_zil_data', instance=True)
            causes_neg_bin = generator_x(samples=n_samples, p=perc_zero_ln_bn, device=device, perc=perc_neg_binomial, causes_mean=self.causes_mean_nb, causes_std=self.causes_std_nb, generator_class=ZeroInflatedDataGenerator, method_name='generate_zinb_data', instance=True)
            causes_uniform = generator_x(samples=n_samples, p=perc_zero_u_p, device=device, perc=perc_uniform, causes_mean=self.causes_mean, causes_std=self.causes_std, generator_class=ZeroInflatedDataGenerator, method_name='generate_ziu_data', instance=False)
            causes_poisson = generator_x(samples=n_samples, p=perc_zero_u_p, device=device, perc=perc_poisson, causes_mean=self.causes_mean, causes_std=self.causes_std, generator_class=ZeroInflatedDataGenerator, method_name='generate_zip_data', instance=False)
            causes = torch.cat((causes_log_norm, causes_neg_bin, causes_uniform, causes_poisson), dim=2)

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
        if not isinstance(config, dict):
            config = config.dump()
        self.config = parse_distributions(config or {})

    def get_batch(self, batch_size, n_samples, num_features, device=default_device, num_outputs=1, epoch=None, single_eval_pos=None):
        sample = [MLP(device, num_features, num_outputs, n_samples, **sample_distributions(self.config)).to(device)() for _ in range(0, batch_size)]
        x, y = zip(*sample)

        y = torch.cat(y, 1).detach().squeeze(2)
        x = torch.cat(x, 1).detach()
        return x, y, y
