import copy

import torch

import btorch.bnn as bnn

_SUPPORTED_LAYERS = [bnn.Conv2d, bnn.Linear]


def enable_grad(module):
    for p in module.parameters():
        p.requires_grad = True


def disable_grad(module):
    for p in module.parameters():
        p.requires_grad = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_zero_mean_unit_variance_prior(model):
    prior_model = copy.deepcopy(model)
    disable_grad(prior_model)
    for lp0 in prior_model.modules():
        if type(lp0) in _SUPPORTED_LAYERS:
            lp0.weight.copy_(torch.zeros_like(lp0.weight))
            lp0.weight_var.copy_(torch.ones_like(lp0.weight_var))
            if lp0.bias is not None:
                lp0.bias.copy_(torch.zeros_like(lp0.bias))
                lp0.bias_var.copy_(torch.ones_like(lp0.bias_var))
    return prior_model
