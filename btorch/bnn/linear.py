import torch
import torch.nn as nn
import torch.nn.functional as F

from btorch.bnn.distributions import ParametricGaussianPrior, ParametricGaussianPosterior


class Linear(nn.Module):
    """ Bayesian implementation of torch.nn.Linear.

    Args:
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
        in_features: int,
        out_features: int,
        bias: bool = True,
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
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.shared_prior = shared_prior
        self.force_sampling = force_sampling
        weight_prior_shape = (1,) if shared_prior else (out_features, in_features)
        self.weight_prior = ParametricGaussianPrior(
            weight_prior_shape,
            mu_range=prior_mu,
            rho_range=prior_rho,
            **factory_kwargs,
        )
        self.weight_posterior = ParametricGaussianPosterior(
            (out_features, in_features),
            mu_range=posterior_mu,
            rho_range=posterior_rho,
            **factory_kwargs,
        )
        if bias:
            bias_prior_shape = (1,) if shared_prior else (out_features,)
            self.bias_prior = ParametricGaussianPrior(
                bias_prior_shape,
                mu_range=prior_mu,
                rho_range=prior_rho,
                **factory_kwargs,
            )
            self.bias_posterior = ParametricGaussianPosterior(
                (out_features,),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = None
        if self.training or self.force_sampling:
            batch_size, f_in = x.shape
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

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias
        )
