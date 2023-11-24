import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        self.weight_var = nn.Parameter(1e-6 * torch.ones_like(self.weight))
        if bias:
            self.bias_var = nn.Parameter(1e-6 * torch.ones_like(self.bias))
        else:
            self.register_parameter("bias_var", None)

    def _conv_forward(self, input, weight, bias):
        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                weight,
                bias,
                self.stride,
                (0, 0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def forward(self, input):
        if not self.training:
            return self._conv_forward(input, self.weight, self.bias)
        weight = self.weight + torch.randn_like(self.weight) * self.weight_var.clamp(min=1e-8)
        if self.bias is not None:
            bias = self.bias + torch.randn_like(self.bias) * self.bias_var.clamp(min=1e-8)
        else:
            bias = None
        return self._conv_forward(input, weight, bias)
