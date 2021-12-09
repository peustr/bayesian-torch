import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_means = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.weight_stds = nn.Parameter(torch.ones((out_features, in_features), **factory_kwargs))
        self.weight = torch.empty((out_features, in_features), **factory_kwargs)
        if bias:
            self.bias_means = nn.Parameter(torch.empty(out_features, **factory_kwargs))
            self.bias_stds = nn.Parameter(torch.ones(out_features, **factory_kwargs))
            self.bias = torch.empty(out_features, **factory_kwargs)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight_means, a=math.sqrt(5))
        self.weight = torch.normal(self.weight_means, self.weight_stds)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_means)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_means, -bound, bound)
            self.bias = torch.normal(self.bias_means, self.bias_stds)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.weight = torch.normal(self.weight_means, self.weight_stds)
        if self.bias is not None:
            self.bias = torch.normal(self.bias_means, self.bias_stds)
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
