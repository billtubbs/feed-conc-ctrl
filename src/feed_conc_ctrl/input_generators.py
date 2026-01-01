"""Input signal generators for simulation.

This module provides functions for generating bounded random walk signals
based on Nicolau (2002).

Reference:
    J. Nicolau, "Stationary Processes That Look Like Random Walks -
    The Bounded Random Walk Process in Discrete and Continuous Time",
    Econometric Theory, 18, 2002, 99-118.
"""

import numpy as np


def brw_reversion_bias(x, alpha1, alpha2, beta, tau):
    """Calculate reversion bias for bounded random walk.

    This is the function 'a(x)' from Nicolau (2002) used in the
    difference equation of the bounded random walk (BRW)
    (see Eq. 1 in the paper).

    Args:
        x: Current state value
        alpha1: First exponential parameter
        alpha2: Second exponential parameter
        beta: Scaling parameter (called 'k' in Nicolau's paper)
        tau: Target/equilibrium value

    Returns:
        Reversion bias value
    """
    a = np.exp(beta) * (
        np.exp(-alpha1 * (x - tau)) - np.exp(alpha2 * (x - tau))
    )
    return a


def sample_bounded_random_walk(
    sd_e, beta, alpha1, alpha2, n, tau, phi=0.5, xkm1=None
):
    """Simulate bounded random walk stochastic process.

    Simulates the Bounded Random Walk stochastic process proposed by
    J. Nicolau (2002). This process generates samples that look like
    random walks but are stationary and bounded around a target value.

    Args:
        sd_e: Standard deviation of the stochastic noise
        beta: Scaling parameter for reversion bias
        alpha1: First exponential parameter for reversion bias
        alpha2: Second exponential parameter for reversion bias
        n: Number of samples to generate
        tau: Target/equilibrium value
        phi: Regularization parameter (default: 0.5)
        xkm1: Initial state value (default: tau)

    Returns:
        Array of n samples from the bounded random walk process
    """
    # Set initial state if not provided
    if xkm1 is None:
        xkm1 = tau

    # Generate white noise
    e = np.random.randn(n)
    p = np.zeros(n)

    # Simulate
    for i in range(n):
        # Stochastic input
        alpha = sd_e * e[i]

        # Reversion bias
        bias = brw_reversion_bias(xkm1, alpha1, alpha2, beta, tau)

        # Regularization step (to avoid instability)
        if abs(bias) < 2 * abs(xkm1 - tau):
            x = xkm1 + bias + alpha
        else:
            x = tau + phi * (xkm1 - tau) + alpha

        # Bounded process output
        p[i] = x
        xkm1 = x

    return p
