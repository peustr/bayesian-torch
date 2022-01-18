def KL_divergence(bnn):
    bnn_modules = [m for m in bnn.modules() if(hasattr(m, 'log_prior') and hasattr(m, 'log_posterior'))]
    kl_divergence = bnn_modules[0].log_posterior - bnn_modules[0].log_prior
    for module in bnn_modules[1:]:
        kl_divergence += module.log_posterior - module.log_prior
    return kl_divergence
