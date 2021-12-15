import math

import torch
import torch.nn as nn


class ParametricGaussian(nn.Module):
    def __init__(self, shape, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.mu = nn.Parameter(torch.empty(shape, **factory_kwargs).uniform_(-0.2, 0.2))
        self.rho = nn.Parameter(torch.empty(shape, **factory_kwargs).uniform_(-4., -1.))

    @property
    def sigma(self):
        return self.rho.exp().log1p()

    def sample(self):
        return self.mu + self.sigma * torch.randn_like(self.mu)

    def log_prob(self, w, reduction=True):
        log_probs = (
            - math.log(math.sqrt(2 * math.pi))
            - self.sigma.log()
            - ((w - self.mu).square()) / (2 * self.sigma.square())
        )
        if reduction:
            return log_probs.sum()
        return log_probs


class ParametricGaussianMixture(nn.Module):
    def __init__(self, gaussian_1, gaussian_2, pi=0.5):
        super().__init__()
        self.gaussian_1 = gaussian_1
        self.gaussian_2 = gaussian_2
        self.pi = pi

    def log_prob(self, w):
        return (
            self.pi * self.gaussian_1.log_prob(w, False).exp()
            + (1 - self.pi) * self.gaussian_2.log_prob(w, False).exp()
        ).log().sum()
