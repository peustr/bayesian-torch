from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair, _reverse_repeat_tuple

from btorch.bnn.distributions import ParametricGaussianPrior, ParametricGaussianPosterior


class _ConvNd(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, ...],
        stride: Tuple[int, ...],
        padding: Tuple[int, ...],
        dilation: Tuple[int, ...],
        transposed: bool,
        output_padding: Tuple[int, ...],
        groups: int,
        bias: bool,
        padding_mode: str,
        shared_prior: bool = True,
        force_sampling: bool = False,
        prior_mu=None,
        prior_rho=None,
        posterior_mu=None,
        posterior_rho=None,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_strings = {'same', 'valid'}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    "Invalid padding string {!r}, should be one of {}".format(padding, valid_padding_strings)
                )
            if padding == 'same' and any(s != 1 for s in stride):
                raise ValueError("padding='same' is not supported for strided convolutions")

        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError(
                "padding_mode must be one of {}, but got padding_mode='{}'".format(valid_padding_modes, padding_mode)
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        self.shared_prior = shared_prior
        self.force_sampling = force_sampling
        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
            if padding == 'same':
                for d, k, i in zip(dilation, kernel_size, range(len(kernel_size) - 1, -1, -1)):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (total_padding - left_pad)
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)

        if transposed:
            weight_prior_shape = (1,) if shared_prior else (in_channels, out_channels // groups, *kernel_size)
            self.weight_posterior = ParametricGaussianPosterior(
                (in_channels, out_channels // groups, *kernel_size),
                mu_range=posterior_mu,
                rho_range=posterior_rho,
                **factory_kwargs,
            )
        else:
            weight_prior_shape = (1,) if shared_prior else (out_channels, in_channels // groups, *kernel_size)
            self.weight_posterior = ParametricGaussianPosterior(
                (out_channels, in_channels // groups, *kernel_size),
                mu_range=posterior_mu,
                rho_range=posterior_rho,
                **factory_kwargs,
            )
        self.weight_prior = ParametricGaussianPrior(
            weight_prior_shape,
            mu_range=prior_mu,
            rho_range=prior_rho,
            **factory_kwargs,
        )
        if bias:
            bias_prior_shape = (1,) if shared_prior else (out_channels,)
            self.bias_prior = ParametricGaussianPrior(
                bias_prior_shape,
                mu_range=prior_mu,
                rho_range=prior_rho,
                **factory_kwargs,
            )
            self.bias_posterior = ParametricGaussianPosterior(
                (out_channels,),
                mu_range=posterior_mu,
                rho_range=posterior_rho,
                **factory_kwargs,
            )
        else:
            self.register_parameter('bias_prior', None)
            self.register_parameter('bias_posterior', None)

    def reset_parameters(self) -> None:
        self.weight_prior.reset_parameters()
        self.weight_posterior.reset_parameters()
        if self.bias:
            self.bias_prior.reset_parameters()
            self.bias_posterior.reset_parameters()

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}' ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if not self.bias:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'


class Conv2d(_ConvNd):
    """ Bayesian implementation of torch.nn.Conv2d.

    Args:
        `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding`, `dilation`, `groups`, `bias`,
            `padding_mode`: Same as https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#conv2d.
        `shared_prior` (bool, optional): The same prior is shared for all weights and biases to
            optimize memory usage. If true, it will force the weight prior to be of shape (1,)
            instead of (out_channels, in_channels // groups, *kernel_size), and the bias prior to
            be of shape (1,) instead of (out_channels,). Default: `True`.
        `force_sampling` (bool, optional): If true, samples from the parameter distribution
            during testing, otherwise uses the mean. Default: `False`.
        `prior_mu`, `prior_rho`, `posterior_mu`, `posterior_rho` (2-tuple, optional): Tuples of the
            form (low, high) for weight and bias initialization. If specified, the parametric
            Gaussian prior and posterior distributions will be initialized uniformly from the
            range (low, high). Default: `None`.

    Note:
        Check the `ParametricGaussianPrior` and `ParametricGaussianPosterior` classes in the
            `btorch.bnn.distributions` module for the default initialization values of the
            distributions when the `*_mu` and `*_rho` arguments are `None`.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        shared_prior: bool = True,
        force_sampling: bool = False,
        prior_mu=None,
        prior_rho=None,
        posterior_mu=None,
        posterior_rho=None,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs,
        )

    def _conv_forward(self, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], groups: int):
        if self.padding_mode != 'zeros':
            return F.conv2d(
                F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                weight, bias, self.stride, _pair(0), self.dilation, groups,
            )
        return F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = None
        if self.training or self.force_sampling:
            batch_size, c_in, xH, xW = x.shape
            weight = self.weight_posterior.sample(batch_size=batch_size)
            x = x.view(1, batch_size * c_in, xH, xW)
            weight = weight.view(batch_size * self.out_channels, c_in, *self.kernel_size)
            if self.bias:
                bias = self.bias_posterior.sample(batch_size=batch_size)
                bias = bias.view(batch_size * self.out_channels)
            x = self._conv_forward(x, weight, bias, batch_size)
            _, _, new_xH, new_xW = x.shape
            x = x.view(batch_size, self.out_channels, new_xH, new_xW)
            return x
        weight = self.weight_posterior.mu.data
        if self.bias:
            bias = self.bias_posterior.mu.data
        x = self._conv_forward(x, weight, bias, self.groups)
        return x
