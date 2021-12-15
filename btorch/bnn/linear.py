import torch
import torch.nn as nn
import torch.nn.functional as F

from btorch.bnn.distributions import ParametricGaussian, ParametricGaussianMixture


class Linear(nn.Module):

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, prior_pi=0.5) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_distribution = ParametricGaussian((out_features, in_features), **factory_kwargs)
        self.weight_prior = ParametricGaussianMixture(
            ParametricGaussian((out_features, in_features), **factory_kwargs),
            ParametricGaussian((out_features, in_features), **factory_kwargs),
            pi=prior_pi
        )
        self.weight = self.weight_distribution.sample()
        if bias:
            self.bias_distribution = ParametricGaussian(out_features, **factory_kwargs)
            self.bias_prior = ParametricGaussianMixture(
                ParametricGaussian(out_features, **factory_kwargs),
                ParametricGaussian(out_features, **factory_kwargs),
                pi=prior_pi
            )
            self.bias = self.bias_distribution.sample()
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.weight = self.weight_distribution.sample()
        if self.bias is not None:
            self.bias = self.bias_distribution.sample()
        self.log_prior = None
        self.log_posterior = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.weight = self.weight_distribution.sample()
            self.log_prior = self.weight_prior.log_prob(self.weight)
            self.log_posterior = self.weight_distribution.log_prob(self.weight)
            if self.bias is not None:
                self.bias = self.bias_distribution.sample()
                self.log_prior += self.bias_prior.log_prob(self.bias)
                self.log_posterior += self.bias_distribution.log_prob(self.bias)
        else:
            self.weight = self.weight_distribution.mu
            if self.bias is not None:
                self.bias = self.bias_distribution.mu
            self.log_prior = None
            self.log_posterior = None
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
