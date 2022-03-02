import torch.nn as nn
import torch.nn.functional as F

from btorch.bnn.distributions import ParametricGaussianPrior, ParametricGaussianPosterior


class Conv2d(nn.Module):
    """ Bayesian implementation of torch.nn.Conv2d.

    Arguments:
        `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding`, `dilation`, `groups`, `bias`:
            Same as https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#conv2d.
        `shared_prior` (bool, optional): The same prior is shared for all weights and biases to
            optimize memory usage. If true, it will force the weight prior to be of shape (1,)
            instead of (out_channels, in_channels // groups, *kernel_size), and the bias prior to
            be of shape (1,) instead of (out_channels,). Default: `True`.
        `prior_mu`, `prior_rho`, `posterior_mu`, `posterior_rho` (2-tuple, optional): Tuples of the
            form (low, high) for weight and bias initialization. If specified, the parametric
            Gaussian prior and posterior distributions will be initialized uniformly from the
            range (low, high). Default: `None`.

    Note:
        For arguments `kernel_size`, `stride`, `padding` and `dilation` only integer values are
            currently supported.
    Note:
        Check the `ParametricGaussianPrior` and `ParametricGaussianPosterior` classes in the
            `btorch.bnn.distributions` module for the default initialization values of the
            distributions when the `*_mu` and `*_rho` arguments are `None`.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        shared_prior=True,
        prior_mu=None,
        prior_rho=None,
        posterior_mu=None,
        posterior_rho=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.bias = bias
        weight_prior_shape = (1,) if shared_prior else (out_channels, in_channels // groups, *self.kernel_size)
        self.weight_prior = ParametricGaussianPrior(
            weight_prior_shape,
            mu=prior_mu,
            rho=prior_rho,
        )
        self.weight_posterior = ParametricGaussianPosterior(
            (out_channels, in_channels // groups, *self.kernel_size),
            mu=posterior_mu,
            rho=posterior_rho,
        )
        if self.bias:
            bias_prior_shape = (1,) if shared_prior else (out_channels,)
            self.bias_prior = ParametricGaussianPrior(
                bias_prior_shape,
                mu=prior_mu,
                rho=prior_rho,
            )
            self.bias_posterior = ParametricGaussianPosterior(
                (out_channels,),
                mu=posterior_mu,
                rho=posterior_rho,
            )

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
