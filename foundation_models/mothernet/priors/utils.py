import random

import torch
from torch import nn

from mothernet.distributions import zipf_sampler_f


def order_by_y(x, y):
    order = torch.argsort(y if random.randint(0, 1) else -y, dim=0)[:, 0, 0]
    order = order.reshape(2, -1).transpose(0, 1).reshape(-1)  # .reshape(n_samples)
    x = x[order]  # .reshape(2, -1).transpose(0, 1).reshape(-1).flip([0]).reshape(n_samples, 1, -1)
    y = y[order]  # .reshape(2, -1).transpose(0, 1).reshape(-1).reshape(n_samples, 1, -1)

    return x, y


def randomize_classes(x, num_classes):
    classes = torch.arange(0, num_classes, device=x.device)
    random_classes = torch.randperm(num_classes, device=x.device).type(x.type())
    x = ((x.unsqueeze(-1) == classes) * random_classes).sum(-1)
    return x


class CategoricalActivation(nn.Module):
    def __init__(self, categorical_p=0.1, ordered_p=0.7, keep_activation_size=False, num_classes_sampler=None):
        if num_classes_sampler is None:
            num_classes_sampler = zipf_sampler_f(0.8, 1, 10)
        self.categorical_p = categorical_p
        self.ordered_p = ordered_p
        self.keep_activation_size = keep_activation_size
        self.num_classes_sampler = num_classes_sampler

        super().__init__()

    def forward(self, x):
        # x shape: T, B, H

        x = nn.Softsign()(x)
        num_classes = self.num_classes_sampler()
        hid_strength = torch.abs(x).mean(0).unsqueeze(0) if self.keep_activation_size else None

        categorical_classes = torch.rand((x.shape[1], x.shape[2])) < self.categorical_p
        class_boundaries = torch.zeros((num_classes - 1, x.shape[1], x.shape[2]), device=x.device, dtype=x.dtype)
        # Sample a different index for each hidden dimension, but shared for all batches
        for b in range(x.shape[1]):
            for h in range(x.shape[2]):
                ind = torch.randint(0, x.shape[0], (num_classes - 1,))
                class_boundaries[:, b, h] = x[ind, b, h]

        for b in range(x.shape[1]):
            x_rel = x[:, b, categorical_classes[b]]
            boundaries_rel = class_boundaries[:, b, categorical_classes[b]].unsqueeze(1)
            x[:, b, categorical_classes[b]] = (x_rel > boundaries_rel).sum(dim=0).float() - num_classes / 2

        ordered_classes = torch.rand((x.shape[1], x.shape[2])) < self.ordered_p
        ordered_classes = torch.logical_and(ordered_classes, categorical_classes)
        x[:, ordered_classes] = randomize_classes(x[:, ordered_classes], num_classes)

        x = x * hid_strength if self.keep_activation_size else x

        return x

def min_max_scaler_torch(X_tensor: torch.Tensor, min_val: int=0, max_val: int=1):
    X_std = (X_tensor - X_tensor.amin(dim=0)) / (X_tensor.amax(dim=0) - X_tensor.amin(dim=0))
    X_scaled = X_std * (max_val - min_val) + min_val
    return X_scaled

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