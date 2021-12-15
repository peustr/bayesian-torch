def KL_divergence(bnn):
    kl_divergence = 0.
    for module in bnn.modules():
        if hasattr(module, 'log_prior') and hasattr(module, 'log_posterior'):
            kl_divergence += module.log_variational_posterior - module.log_prior
    return kl_divergence
