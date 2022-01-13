def KL_divergence(bnn, update_prior=True):
    kl_divergence = 0.
    for module in bnn.modules():
        if hasattr(module, 'log_prior') and hasattr(module, 'log_posterior'):
            kl_divergence += module.log_posterior - module.log_prior
            if update_prior:
                module.weight_prior.update(module.weight_distribution)
                if module.bias:
                    module.bias_prior.update(module.bias_distribution)
    return kl_divergence
