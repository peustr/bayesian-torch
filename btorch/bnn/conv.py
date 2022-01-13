import torch.nn as nn
import torch.nn.functional as F

from btorch.bnn.distributions import ParametricGaussian


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.bias = bias
        self.weight_distribution = ParametricGaussian(
            (out_channels, in_channels // groups, *self.kernel_size))
        self.weight_prior = ParametricGaussian(
            (out_channels, in_channels // groups, *self.kernel_size), requires_grad=False)
        if self.bias:
            self.bias_distribution = ParametricGaussian((out_channels,))
            self.bias_prior = ParametricGaussian((out_channels,), requires_grad=False)

    def forward(self, x, test=False):
        bias = None
        if test:
            weight = self.weight_distribution.mu.data
            if self.bias:
                bias = self.bias_distribution.mu.data
        else:
            weight = self.weight_distribution.rsample()
            self.log_prior = self.weight_prior.log_prob(weight).sum()
            self.log_posterior = self.weight_distribution.log_prob(weight).sum()
            if self.bias:
                bias = self.bias_distribution.rsample()
                self.log_prior += self.bias_prior.log_prob(bias).sum()
                self.log_posterior += self.bias_distribution.log_prob(bias).sum()
        return F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)
