import torch
import torch.nn as nn
import torch.nn.init as init
from torch.distributions.normal import Normal


class ParametricGaussian(nn.Module):
    def __init__(self, shape, requires_grad=True):
        super().__init__()
        self.requires_grad = requires_grad
        self.mu = nn.Parameter(torch.empty(shape), requires_grad=requires_grad)
        self.rho = nn.Parameter(torch.empty(shape), requires_grad=requires_grad)
        init.uniform_(self.mu, -0.1, 0.1)
        init.uniform_(self.rho, -5., -4.)

    @property
    def sigma(self):
        return self.rho.exp().log1p()

    @property
    def obj(self):
        return Normal(self.mu, self.sigma)

    def sample(self, sample_shape=torch.Size([])):
        return self.obj.sample(sample_shape=sample_shape)

    def rsample(self, sample_shape=torch.Size([])):
        return self.obj.rsample(sample_shape=sample_shape)

    def log_prob(self, x):
        return self.obj.log_prob(x)
