import torch
import torch.nn as nn
import torch.nn.init as init


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

    def sample(self, batch_size=None):
        if batch_size is None:
            eps = torch.randn(self.mu.shape, dtype=self.mu.dtype, device=self.mu.device)
        else:
            eps = torch.randn(batch_size, *self.mu.shape, dtype=self.mu.dtype, device=self.mu.device)
        return self.mu + eps * self.sigma
