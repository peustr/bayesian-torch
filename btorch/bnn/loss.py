def kl_divergence(bnn, reduction='mean'):
    assert reduction in ('sum', 'mean')
    n, loss = 0., 0.
    for m in bnn.modules():
        if (
            hasattr(m, 'weight_posterior') and
            hasattr(m, 'weight_prior') and
            m.weight_posterior is not None and
            m.weight_prior is not None
        ):
            n += 1.
            loss += _kl(m.weight_posterior, m.weight_prior)
            if (
                hasattr(m, 'bias_posterior') and
                hasattr(m, 'bias_prior') and
                m.bias_posterior is not None and
                m.bias_prior is not None
            ):
                loss += _kl(m.bias_posterior, m.bias_prior)
    if reduction == 'mean':
        loss /= n
    return loss


def _kl(p, q):
    return (
        0.5 * (
            (p.sigma / q.sigma) ** 2 + (q.mu - p.mu) ** 2 / (q.sigma ** 2) - 1. + 2. * (q.sigma / p.sigma).log()
        )
    ).mean()
