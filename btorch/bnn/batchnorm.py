from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from btorch.bnn.distributions import ParametricGaussianPrior, ParametricGaussianPosterior


class _NormBase(nn.Module):
    """Common base of _InstanceNorm and _BatchNorm"""

    _version = 2

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
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
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.shared_prior = shared_prior
        self.force_sampling = force_sampling
        if self.affine:
            prior_shape = (1,) if shared_prior else (num_features,)
            self.weight_prior = ParametricGaussianPrior(
                prior_shape,
                mu_range=prior_mu,
                rho_range=prior_rho,
                **factory_kwargs,
            )
            self.weight_posterior = ParametricGaussianPosterior(
                (num_features,),
                mu_range=posterior_mu,
                rho_range=posterior_rho,
                **factory_kwargs,
            )
            self.bias_prior = ParametricGaussianPrior(
                prior_shape,
                mu_range=prior_mu,
                rho_range=prior_rho,
                **factory_kwargs,
            )
            self.bias_posterior = ParametricGaussianPosterior(
                (num_features,),
                mu_range=posterior_mu,
                rho_range=posterior_rho,
                **factory_kwargs,
            )
        else:
            self.register_parameter("weight_prior", None)
            self.register_parameter("weight_posterior", None)
            self.register_parameter("bias_prior", None)
            self.register_parameter("bias_posterior", None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, **factory_kwargs))
            self.register_buffer('running_var', torch.ones(num_features, **factory_kwargs))
            self.running_mean: Optional[torch.Tensor]
            self.running_var: Optional[torch.Tensor]
            self.register_buffer(
                'num_batches_tracked',
                torch.tensor(0, dtype=torch.long, **{k: v for k, v in factory_kwargs.items() if k != 'dtype'})
            )
            self.num_batches_tracked: Optional[torch.Tensor]
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            # running_mean/running_var/num_batches... are registered at runtime depending
            # if self.track_running_stats is on
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            self.weight_prior.reset_parameters()
            self.weight_posterior.reset_parameters()
            self.bias_prior.reset_parameters()
            self.bias_posterior.reset_parameters()

    def _check_input_dim(self, x):
        raise NotImplementedError

    def extra_repr(self):
        return (
            "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats}".format(**self.__dict__)
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + "num_batches_tracked"
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class _BatchNorm(_NormBase):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        shared_prior: bool = True,
        force_sampling: bool = False,
        prior_mu=None,
        prior_rho=None,
        posterior_mu=None,
        posterior_rho=None,
        device=None,
        dtype=None
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, shared_prior, force_sampling,
            prior_mu, prior_rho, posterior_mu, posterior_rho, **factory_kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(x)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked.add_(1)
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        if self.affine:
            if self.training or self.force_sampling:
                weight = self.weight_posterior.sample()
                bias = self.bias_posterior.sample()
            else:
                weight = self.weight_posterior.mu.data
                bias = self.bias_posterior.mu.data
        else:
            weight = None
            bias = None

        return F.batch_norm(
            x,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            weight,
            bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )


class BatchNorm1d(_BatchNorm):
    """ Bayesian implementation of torch.nn.BatchNorm1d.

    Args:
        `num_features`, `eps`, `momentum`, `affine`, `track_running_stats`: Same as
            https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html.
        `shared_prior` (bool, optional): The same prior is shared for all weights and biases to
            optimize memory usage. If true, it will force the weight prior to be of shape (1,)
            instead of (num_features,), and the bias prior to be of shape (1,) instead of (num_features,).
            Default: `True`.
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
    def _check_input_dim(self, x):
        if x.dim() != 2 and x.dim() != 3:
            raise ValueError("expected 2D or 3D input (got {}D input)".format(x.dim()))


class BatchNorm2d(_BatchNorm):
    """ Bayesian implementation of torch.nn.BatchNorm2d.

    Args:
        `num_features`, `eps`, `momentum`, `affine`, `track_running_stats`: Same as
            https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html.
        `shared_prior` (bool, optional): The same prior is shared for all weights and biases to
            optimize memory usage. If true, it will force the weight prior to be of shape (1,)
            instead of (num_features,), and the bias prior to be of shape (1,) instead of (num_features,).
            Default: `True`.
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
    def _check_input_dim(self, x):
        if x.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(x.dim()))
