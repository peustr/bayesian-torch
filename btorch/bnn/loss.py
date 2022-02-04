def kl_divergence(bnn, reduction='mean'):
    assert reduction in ('sum', 'mean')
    n, loss = 0., 0.
    for m in bnn.modules():
        if hasattr(m, 'weight_posterior') and hasattr(m, 'weight_prior'):
            n += 1.
            loss += _kl(m.weight_posterior, m.weight_prior)
            if m.bias:
                loss += _kl(m.bias_posterior, m.bias_prior)
    if reduction == 'mean':
        loss /= n
    return loss


def _kl(p, q):
    return (
        0.5 * (
            (q.sigma / p.sigma) ** 2 + (p.mu - q.mu) ** 2 / (p.sigma ** 2) - 1. + 2. * (p.sigma / q.sigma).log()
        )
    ).sum()
