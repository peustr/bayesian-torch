import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.weight_var = nn.Parameter(1e-6 * torch.ones_like(self.weight))
        if bias:
            self.bias_var = nn.Parameter(1e-6 * torch.ones_like(self.bias))
        else:
            self.register_parameter("bias_var", None)

    def forward(self, input):
        if not self.training:
            return F.linear(input, self.weight, self.bias)
        weight = self.weight + torch.randn_like(self.weight) * self.weight_var.clamp(min=1e-8)
        if self.bias is not None:
            bias = self.bias + torch.randn_like(self.bias) * self.bias_var.clamp(min=1e-8)
        else:
            bias = None
        return F.linear(input, weight, bias)
