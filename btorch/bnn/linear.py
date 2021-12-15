from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from btorch.bnn.distributions import ParametricGaussian, ParametricGaussianMixture


class Linear(nn.Module):

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight_distribution: nn.Module
    weight_prior: nn.Module
    bias_distribution: Optional[nn.Module]
    bias_prior: Optional[nn.Module]

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
        if bias:
            self.bias_distribution = ParametricGaussian(out_features, **factory_kwargs)
            self.bias_prior = ParametricGaussianMixture(
                ParametricGaussian(out_features, **factory_kwargs),
                ParametricGaussian(out_features, **factory_kwargs),
                pi=prior_pi
            )
        else:
            self.register_parameter('bias_distribution', None)
            self.register_parameter('bias_prior', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.log_prior = None
        self.log_posterior = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w = self.weight_distribution.sample()
            self.log_prior = self.weight_prior.log_prob(w)
            self.log_posterior = self.weight_distribution.log_prob(w)
            if self.bias_distribution is not None:
                b = self.bias_distribution.sample()
                self.log_prior += self.bias_prior.log_prob(b)
                self.log_posterior += self.bias_distribution.log_prob(b)
        else:
            w = self.weight_distribution.mu
            if self.bias_distribution is not None:
                b = self.bias_distribution.mu
            self.log_prior = None
            self.log_posterior = None
        return F.linear(x, w, b)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias_distribution is not None
        )
