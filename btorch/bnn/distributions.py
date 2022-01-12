import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class ParametricGaussian(nn.Module):
    def __init__(self, shape, requires_grad=True):
        super().__init__()
        self.requires_grad = requires_grad
        self.mu = nn.Parameter(torch.randn(shape), requires_grad=requires_grad)
        self.rho = nn.Parameter(torch.randn(shape), requires_grad=requires_grad)

    @property
    def sigma(self):
        return self.rho.exp().log1p()

    @property
    def obj(self):
        return Normal(self.mu, self.sigma)

    def sample(self):
        return self.obj.sample()

    def rsample(self):
        return self.obj.rsample()

    def log_prob(self, x):
        return self.obj.log_prob(x)

    def update(self, other):
        self.load_state_dict(other.state_dict())
        self.mu.requires_grad = self.requires_grad
        self.rho.requires_grad = self.requires_grad
