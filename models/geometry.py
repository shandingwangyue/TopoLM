"""Shared diagonal-Gaussian 2-Wasserstein geometry."""

import torch


def pairwise_squared_distance(x, y):
    """Squared Euclidean distances without materializing [..., N, M, D]."""
    return (x.square().sum(-1, keepdim=True)
            + y.square().sum(-1).unsqueeze(-2)
            - 2 * x @ y.transpose(-2, -1)).clamp_min(0)


def diagonal_wasserstein_sq(mu_a, logvar_a, mu_b, logvar_b, *, pairwise=False):
    """W2² = ||mu_a-mu_b||² + ||sqrt(var_a)-sqrt(var_b)||².

    logvar is the log diagonal covariance, not log standard deviation.
    Both attention implementations use the same numerical variance bounds.
    """
    std_a = (0.5 * logvar_a.clamp(-20, 20)).exp()
    std_b = (0.5 * logvar_b.clamp(-20, 20)).exp()
    if pairwise:
        return pairwise_squared_distance(mu_a, mu_b) + pairwise_squared_distance(std_a, std_b)
    return (mu_a - mu_b).square().sum(-1) + (std_a - std_b).square().sum(-1)


def diagonal_wasserstein_interpolate(mu_a, variance_a, mu_b, variance_b, weight=0.5):
    """Diagonal Gaussian W2 geodesic: interpolate means and std deviations.

    weight=1 returns A; weight=0 returns B. This is not a mixture density.
    """
    if not 0 <= weight <= 1:
        raise ValueError("weight must be in [0, 1]")
    mean = weight * mu_a + (1 - weight) * mu_b
    variance = (weight * variance_a.sqrt() + (1 - weight) * variance_b.sqrt()).square()
    return mean, variance
