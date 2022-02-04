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
        self.weight_posterior = ParametricGaussian(
            (out_channels, in_channels // groups, *self.kernel_size))
        self.weight_prior = ParametricGaussian(
            (out_channels, in_channels // groups, *self.kernel_size), requires_grad=False)
        if self.bias:
            self.bias_posterior = ParametricGaussian((out_channels,))
            self.bias_prior = ParametricGaussian((out_channels,), requires_grad=False)

    def forward(self, x):
        bias = None
        batch_size, c_in, xH, xW = x.shape
        if self.training:
            weight = self.weight_posterior.sample(batch_size=batch_size)
            x = x.view(1, batch_size * c_in, xH, xW)
            weight = weight.view(batch_size * self.out_channels, c_in, *self.kernel_size)
            if self.bias:
                bias = self.bias_posterior.sample(batch_size=batch_size)
                bias = bias.view(batch_size * self.out_channels)
            x = F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, groups=batch_size)
            _, _, new_xH, new_xW = x.shape
            x = x.view(batch_size, self.out_channels, new_xH, new_xW)
            return x
        weight = self.weight_posterior.mu.data
        if self.bias:
            bias = self.bias_posterior.mu.data
        return F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)
