import torch.nn as nn
import torch.nn.functional as F

from btorch.bnn.distributions import ParametricGaussianPrior, ParametricGaussianPosterior


class Linear(nn.Module):
    """ Bayesian implementation of torch.nn.Linear.

    Arguments:
        `in_features`, `out_features`, `bias`: Same as
            https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#linear.
        `shared_prior` (bool, optional): The same prior is shared for all weights and biases to
            optimize memory usage. If true, it will force the weight prior to be of shape (1,)
            instead of (out_features, in_features), and the bias prior to
            be of shape (1,) instead of (out_features,). Default: `True`.
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
        in_features,
        out_features,
        bias=True,
        shared_prior=True,
        force_sampling=False,
        prior_mu=None,
        prior_rho=None,
        posterior_mu=None,
        posterior_rho=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.shared_prior = shared_prior
        self.force_sampling = force_sampling
        weight_prior_shape = (1,) if shared_prior else (out_features, in_features)
        self.weight_prior = ParametricGaussianPrior(
            weight_prior_shape,
            mu=prior_mu,
            rho=prior_rho,
        )
        self.weight_posterior = ParametricGaussianPosterior(
            (out_features, in_features),
            mu=posterior_mu,
            rho=posterior_rho,
        )
        if self.bias:
            bias_prior_shape = (1,) if shared_prior else (out_features,)
            self.bias_prior = ParametricGaussianPrior(
                bias_prior_shape,
                mu=prior_mu,
                rho=prior_rho,
            )
            self.bias_posterior = ParametricGaussianPosterior(
                (out_features,),
                mu=posterior_mu,
                rho=posterior_rho,
            )

    def forward(self, x):
        bias = None
        batch_size, f_in = x.shape
        if self.training or self.force_sampling:
            weight = self.weight_posterior.sample(batch_size=batch_size)
            x = x[:, None, :] @ weight.transpose(1, 2)
            x = x.view(batch_size, self.out_features)
            if self.bias:
                bias = self.bias_posterior.sample(batch_size=batch_size)
                x += bias
            return x
        weight = self.weight_posterior.mu.data
        if self.bias:
            bias = self.bias_posterior.mu.data
        return F.linear(x, weight, bias)
