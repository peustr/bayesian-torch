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
        self.weight_posterior = ParametricGaussian((out_features, in_features))
        if self.bias:
            self.bias_prior = ParametricGaussian((out_features,), requires_grad=False)
            self.bias_posterior = ParametricGaussian((out_features,))

    def forward(self, x):
        bias = None
        batch_size, f_in = x.shape
        if self.training:
            weight = self.weight_posterior.sample(batch_size=batch_size)
            x = x[:, None, :] @ weight.transpose(1, 2)
            x = x.view(batch_size, self.out_features)
            if self.bias:
                bias = self.bias_posterior.sample(batch_size=batch_size)
                x += bias
            return x
        weight = self.weight_posterior.mu.data
        if self.bias:
            bias = self.bias_posterior.mu.data
        return F.linear(x, weight, bias)
