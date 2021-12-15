import math

import torch
import torch.nn as nn


class ParametricGaussian(nn.Module):
    def __init__(self, shape, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.mu = nn.Parameter(torch.randn(shape, **factory_kwargs))
        self.rho = nn.Parameter(torch.randn(shape, **factory_kwargs))
        self.epsilon = torch.randn(shape, **factory_kwargs)

    @property
    def sigma(self):
        return self.rho.exp().log1p()

    def sample(self):
        return self.mu + self.sigma * self.epsilon.normal_()

    def log_prob(self, w):
        return (
            - math.log(math.sqrt(2 * math.pi))
            - self.sigma.log()
            - ((w - self.mu).square()) / (2 * self.sigma.square())
        ).sum()


class ParametricGaussianMixture(nn.Module):
    def __init__(self, gaussian_1, gaussian_2, pi=0.5):
        super().__init__()
        self.gaussian_1 = gaussian_1
        self.gaussian_2 = gaussian_2
        self.pi = pi

    def log_prob(self, w):
        return (
            self.pi * self.gaussian_1.log_prob(w).exp()
            + (1 - self.pi) * self.gaussian_2.log_prob(w).exp()
        ).log().sum()
