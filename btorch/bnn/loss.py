import btorch.bnn as bnn

_SUPPORTED_LAYERS = [bnn.Conv2d, bnn.Linear]


def kl_divergence(model, prior_model):
    kl_loss = 0.0
    nl = 0.0
    for lp, lp0 in zip(model.modules(), prior_model.modules()):
        if type(lp) in _SUPPORTED_LAYERS and type(lp0) in _SUPPORTED_LAYERS:
            kl_loss += _kl_div(lp.weight, lp.weight_var, lp0.weight, lp0.weight_var)
            if lp.bias is not None and lp0.bias is not None:
                kl_loss += _kl_div(lp.bias, lp.bias_var, lp0.bias, lp0.bias_var)
            nl += 1.0
    return kl_loss / nl


def _kl_div(m, s, m0, s0):
    var_ratio = (s / s0).pow(2)
    return 0.5 * (var_ratio + ((m - m0) / s0).pow(2) - 1.0 - var_ratio.log()).mean()
