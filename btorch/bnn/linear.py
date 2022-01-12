import torch.nn as nn
import torch.nn.functional as F

from btorch.bnn.distributions import ParametricGaussian


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight_prior = ParametricGaussian((out_features, in_features), requires_grad=False)
        self.weight_distribution = ParametricGaussian((out_features, in_features))
        if self.bias:
            self.bias_prior = ParametricGaussian((out_features,), requires_grad=False)
            self.bias_distribution = ParametricGaussian((out_features,))

    def forward(self, x):
        weight = self.weight_distribution.rsample()
        self.log_prior = self.weight_prior.log_prob(weight)
        self.log_posterior = self.weight_distribution.log_prob(weight)
        if self.bias:
            bias = self.bias_distribution.rsample()
            self.log_prior += self.bias_prior.log_prob(bias)
            self.log_posterior += self.bias_distribution.log_prob(bias)
        else:
            bias = None
        return F.linear(x, weight, bias)
