import torch
import torch.nn as nn
import torch.nn.init as init


class _ParametricGaussian(nn.Module):
    def __init__(self, shape, requires_grad=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.requires_grad = requires_grad
        self.mu = nn.Parameter(torch.empty(shape, **factory_kwargs), requires_grad=requires_grad)
        self.rho = nn.Parameter(torch.empty(shape, **factory_kwargs), requires_grad=requires_grad)

    @property
    def sigma(self):
        return self.rho.exp().log1p()

    def sample(self, batch_size=None):
        if batch_size is None:
            eps = torch.randn(self.mu.shape, dtype=self.mu.dtype, device=self.mu.device)
        else:
            eps = torch.randn(batch_size, *self.mu.shape, dtype=self.mu.dtype, device=self.mu.device)
        return self.mu + eps * self.sigma


class ParametricGaussianPrior(_ParametricGaussian):
    def __init__(self, shape, mu_range=None, rho_range=None, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(shape, requires_grad=False, **factory_kwargs)
        self.mu_range = mu_range
        self.rho_range = rho_range
        self.reset_parameters()

    def reset_parameters(self):
        if self.mu_range is None:
            init.uniform_(self.mu, -0.1, 0.1)
        else:
            init.uniform_(self.mu, self.mu_range[0], self.mu_range[1])
        if self.rho_range is None:
            init.uniform_(self.rho, -1., 0.)
        else:
            init.uniform_(self.rho, self.rho_range[0], self.rho_range[1])


class ParametricGaussianPosterior(_ParametricGaussian):
    def __init__(self, shape, mu_range=None, rho_range=None, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(shape, requires_grad=True, **factory_kwargs)
        self.mu_range = mu_range
        self.rho_range = rho_range
        self.reset_parameters()

    def reset_parameters(self):
        if self.mu_range is None:
            init.uniform_(self.mu, -0.1, 0.1)
        else:
            init.uniform_(self.mu, self.mu_range[0], self.mu_range[1])
        if self.rho_range is None:
            init.uniform_(self.rho, -5., -4.)
        else:
            init.uniform_(self.rho, self.rho_range[0], self.rho_range[1])
