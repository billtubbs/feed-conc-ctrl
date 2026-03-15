# import numba
import numpy as np


# @numba.njit
def brw_reversion_bias(x, alpha1, alpha2, beta, tau):
    """Calculate reversion bias for bounded random walk.

    Args:
        x: Current state value
        alpha1: First exponential parameter
        alpha2: Second exponential parameter
        beta: Scaling parameter
        tau: Target/equilibrium value

    Returns:
        Reversion bias value
    """
    a = np.exp(beta - alpha1 * (x - tau)) - np.exp(beta + alpha2 * (x - tau))
    return a


def compute_brw_parameters_with_steepness(r1, r2, a1, a2):
    """Compute BRW parameters with independent control.

    This parameterization defines:
    - r1: offset from zero where a(r1) = +1 (typically r1 < 0, below zero)
    - r2: offset from zero where a(r2) = -1 (typically r2 > 0, above zero)
    - a1: (|derivative|) at r1 (positive value)
    - a2: (|derivative|) at r2 (positive value)

    Args:
        r1: Offset from zero where bias = +1 (negative for below zero), scalar or 1-D array
        r2: Offset from zero where bias = -1 (positive for above zero), scalar or 1-D array
        a1: (|derivative|) at r1 (positive value), scalar or 1-D array
        a2: (|derivative|) at r2 (positive value), scalar or 1-D array

    Returns:
        tuple: (alpha1, alpha2, beta)
    """
    r1 = np.asarray(r1, dtype=np.float64)
    r2 = np.asarray(r2, dtype=np.float64)
    a1 = np.asarray(a1, dtype=np.float64)
    a2 = np.asarray(a2, dtype=np.float64)

    alpha1 = a2
    alpha2 = a1
    term = np.exp(-alpha1 * r2) - np.exp(alpha2 * r2)

    small = np.abs(term) < 1e-10
    if np.any(small):
        print("Warning: Numerical issues in parameter computation")

    alpha1_fallback = 2.0 / np.abs(r2)
    alpha2_fallback = 2.0 / np.abs(r1)
    beta_fallback = np.log(1.0 / (2.0 * np.sinh(alpha1_fallback * r2)))

    with np.errstate(invalid="ignore", divide="ignore"):
        beta_normal = np.log(-1.0 / term)

    alpha1 = np.where(small, alpha1_fallback, alpha1)
    alpha2 = np.where(small, alpha2_fallback, alpha2)
    beta = np.where(small, beta_fallback, beta_normal)

    return alpha1, alpha2, beta


# @numba.njit
def simulate_brw(e, sd_e, xkm1, alpha1, alpha2, beta, tau, phi):
    """
    Simulate multiple bounded random walks (vectorized).

    Args:
        e: array of shape (n_samples, n_walks) of standard normal samples
        sd_e: array of shape (n_walks,), std of noise
        xkm1: 1-D array of initial states for each walk, length n_walks
        alpha1, alpha2, beta: BRW parameters
        tau: target value
        phi: regularization parameter

    Returns:
        Array of shape (n_samples, n_walks)
    """
    p = np.zeros(e.shape)

    for i in range(e.shape[0]):
        bias = brw_reversion_bias(xkm1, alpha1, alpha2, beta, tau)
        alpha = sd_e * e[i]
        # Vectorized regularization
        x = np.where(
            np.abs(bias) < 2 * np.abs(xkm1 - tau),
            xkm1 + bias + alpha,
            tau + phi * (xkm1 - tau) + alpha,
        )
        p[i] = x
        xkm1 = x  # update for next step
    return p


def _infer_n_walks(n_walks, **params):
    """Infer or validate n_walks from parameter shapes.

    Args:
        n_walks: Explicit number of walks, or None to infer.
        **params: Named float parameters (scalar or 1-D array).

    Returns:
        Resolved n_walks (int).
    """
    lengths = set()
    for val in params.values():
        arr = np.atleast_1d(np.asarray(val, dtype=np.float64))
        if arr.shape[0] > 1:
            lengths.add(arr.shape[0])
    if len(lengths) > 1:
        raise ValueError(
            f"Array arguments have inconsistent lengths: {sorted(lengths)}"
        )
    inferred = lengths.pop() if lengths else 1
    if n_walks is None:
        return inferred
    if inferred > 1 and inferred != n_walks:
        raise ValueError(
            f"n_walks={n_walks} conflicts with array argument "
            f"length {inferred}"
        )
    return n_walks


def _broadcast_param(val, n_walks, name):
    """Return val as a 1-D float64 array of length n_walks.

    Scalars (or length-1 arrays) are tiled; length-n_walks arrays
    are returned as-is.

    Args:
        val: Scalar or 1-D array.
        n_walks: Target length.
        name: Parameter name for error messages.

    Returns:
        np.ndarray of shape (n_walks,).
    """
    arr = np.atleast_1d(np.asarray(val, dtype=np.float64))
    if arr.shape[0] == 1:
        return np.full(n_walks, arr[0], dtype=np.float64)
    if arr.shape[0] == n_walks:
        return arr
    raise ValueError(
        f"{name} must be a scalar or have shape ({n_walks},), "
        f"got shape {arr.shape}"
    )


def sample_bounded_random_walk(
    sd_e,
    r1,
    r2,
    a1,
    a2,
    size,
    phi=0.5,
    xkm1=None,
    rng=None,
    seed=0,
    n_walks=None,
):
    """
    Bounded random walk with uniform sampling.

    Supports generation of multiple independent walks by passing array
    parameters or specifying n_walks explicitly.

    Args:
        sd_e: Standard deviation of the stochastic noise (scalar or
            array of shape (n_walks,))
        r1: Offset from zero where bias = +1 (scalar or array of
            shape (n_walks,))
        r2: Offset from zero where bias = -1 (scalar or array of
            shape (n_walks,))
        a1: |derivative| at r1 (scalar or array of shape (n_walks,))
        a2: |derivative| at r2 (scalar or array of shape (n_walks,))
        size: Number of samples to generate
        phi: Regularization parameter
        xkm1: Initial state value (optional, scalar or array of
            shape (n_walks,))
        rng: np.random.Generator (optional)
        seed: Random seed if rng is None
        n_walks: Number of independent walks. Inferred from the
            longest array argument if None.

    Returns:
        Array of samples from bounded random walk:
        - If n_walks == 1: shape (n_samples,)
        - If n_walks > 1: shape (n_samples, n_walks)
    """
    n_walks = _infer_n_walks(n_walks, sd_e=sd_e, r1=r1, r2=r2, a1=a1, a2=a2)
    sd_e = _broadcast_param(sd_e, n_walks, "sd_e")
    r1 = _broadcast_param(r1, n_walks, "r1")
    r2 = _broadcast_param(r2, n_walks, "r2")
    a1 = _broadcast_param(a1, n_walks, "a1")
    a2 = _broadcast_param(a2, n_walks, "a2")

    # Compute BRW parameters
    alpha1, alpha2, beta = compute_brw_parameters_with_steepness(
        r1, r2, a1, a2
    )

    tau = 0.0

    if rng is None:
        rng = np.random.default_rng(seed=seed)

    # Always generate independent noise for each walk
    e = rng.normal(size=(size, n_walks))

    # Set initial state if not provided
    if xkm1 is None:
        xkm1 = np.full(n_walks, tau, dtype=np.float64)
    else:
        xkm1 = _broadcast_param(xkm1, n_walks, "xkm1")

    p = simulate_brw(e, sd_e, xkm1, alpha1, alpha2, beta, tau, phi)

    return p.squeeze()


# @numba.njit
def simulate_brw_irregular(
    e, t_eval, sd_e, xkm1, alpha1, alpha2, beta, tau, phi
):
    """
    Simulate bounded random walk at irregular sample times.

    Args:
        e: array of shape (n_samples,) of standard normal samples
        t_eval: array of sample times (monotonic increasing)
        sd_e: scalar noise std
        xkm1: scalar initial state
        alpha1, alpha2, beta: BRW parameters (scalars)
        tau: target value (scalar)
        phi: regularization parameter

    Returns:
        Array of shape (n_samples,)
    """
    n = len(e)
    p = np.zeros(n)

    for i in range(n):
        if i == 0:
            dt = t_eval[0] if t_eval[0] > 0 else 1.0
        else:
            dt = t_eval[i] - t_eval[i - 1]

        bias = (
            brw_reversion_bias(np.array([xkm1]), alpha1, alpha2, beta, tau)[0]
            * dt
        )
        alpha = sd_e * np.sqrt(dt) * e[i]

        if np.abs(bias) < 2 * np.abs(xkm1 - tau):
            x = xkm1 + bias + alpha
        else:
            x = tau + phi * (xkm1 - tau) + alpha

        p[i] = x
        xkm1 = x

    return p


def sample_bounded_random_walk_irregular(
    sd_e,
    r1,
    r2,
    a1,
    a2,
    t_eval,
    phi=0.5,
    xkm1=None,
    rng=None,
    seed=0,
    n_walks=None,
):
    """
    Bounded random walk with irregular sampling times.

    Supports generation of multiple independent walks by passing array
    parameters or specifying n_walks explicitly.

    Args:
        sd_e: Standard deviation of the stochastic noise (scalar or
            array of shape (n_walks,))
        r1: Offset from zero where bias = +1 (scalar or array of
            shape (n_walks,))
        r2: Offset from zero where bias = -1 (scalar or array of
            shape (n_walks,))
        a1: |derivative| at r1 (scalar or array of shape (n_walks,))
        a2: |derivative| at r2 (scalar or array of shape (n_walks,))
        t_eval: Array of sample times (must be monotonically increasing)
        phi: Regularization parameter
        xkm1: Initial state value (optional, scalar or array of
            shape (n_walks,))
        rng: np.random.Generator (optional)
        seed: Random seed if rng is None
        n_walks: Number of independent walks. Inferred from the
            longest array argument if None.

    Returns:
        Array of samples from bounded random walk:
        - If n_walks == 1: shape (n_samples,)
        - If n_walks > 1: shape (n_samples, n_walks)
    """
    # Ensure t_eval is NumPy array
    t_eval = np.asarray(t_eval, dtype=np.float64)
    n = len(t_eval)

    n_walks = _infer_n_walks(n_walks, sd_e=sd_e, r1=r1, r2=r2, a1=a1, a2=a2)
    sd_e = _broadcast_param(sd_e, n_walks, "sd_e")
    r1 = _broadcast_param(r1, n_walks, "r1")
    r2 = _broadcast_param(r2, n_walks, "r2")
    a1 = _broadcast_param(a1, n_walks, "a1")
    a2 = _broadcast_param(a2, n_walks, "a2")

    # Compute BRW parameters
    alpha1, alpha2, beta = compute_brw_parameters_with_steepness(
        r1, r2, a1, a2
    )

    tau = 0.0

    if rng is None:
        rng = np.random.default_rng(seed=seed)

    # Generate independent noise for each walk
    e = rng.normal(size=(n, n_walks))

    # Set initial state if not provided
    if xkm1 is None:
        xkm1 = np.full(n_walks, tau, dtype=np.float64)
    else:
        xkm1 = _broadcast_param(xkm1, n_walks, "xkm1")

    # Simulate
    p = np.zeros((n, n_walks))

    for i in range(n):
        # Calculate dt
        if i == 0:
            dt = t_eval[0] if t_eval[0] > 0 else 1.0
        else:
            dt = t_eval[i] - t_eval[i - 1]

        # Reversion bias (vectorized) scaled by dt
        bias_val = np.exp(beta - alpha1 * (xkm1 - tau)) - np.exp(
            beta + alpha2 * (xkm1 - tau)
        )
        bias = bias_val * dt

        # Stochastic input scaled by sqrt(dt)
        alpha = sd_e * np.sqrt(dt) * e[i]

        # Regularization step (vectorized)
        x = np.where(
            np.abs(bias) < 2 * np.abs(xkm1 - tau),
            xkm1 + bias + alpha,
            tau + phi * (xkm1 - tau) + alpha,
        )

        p[i] = x
        xkm1 = x

    return p.squeeze()


# Example usage and testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Common parameters
    sd_e = 1.0
    y_nop = 100.0  # Normal operating point
    r1 = -10.0  # When x is 10 units below zero, bias = +1 (pushes up)
    r2 = 10.0  # When x is 10 units above zero, bias = -1 (pushes down)
    size = 100000
    plot_size = 500

    # parameters - symmetric
    a1_low = 1.0
    a2_low = 1.0

    a1_high = 2.0
    a2_high = 2.0

    # Generate bounded random walks
    seed = 42

    # Generate both samples with same seed (same noise), then shift to y_nop
    samples_low = (
        sample_bounded_random_walk(
            sd_e, r1, r2, a1_low, a2_low, size, seed=seed
        )
        + y_nop
    )
    samples_high = (
        sample_bounded_random_walk(
            sd_e, r1, r2, a1_high, a2_high, size, seed=seed
        )
        + y_nop
    )

    # Compute parameters for both
    alpha1_low, alpha2_low, beta_low = compute_brw_parameters_with_steepness(
        r1, r2, a1_low, a2_low
    )
    alpha1_high, alpha2_high, beta_high = (
        compute_brw_parameters_with_steepness(r1, r2, a1_high, a2_high)
    )

    print(f"Common parameters:")
    print(f"  r1 = {r1}, r2 = {r2}, y_nop = {y_nop}")
    print(f"  sd_e = {sd_e}")

    print(f"\nLow (a1=a2={a1_low}):")
    print(
        f"  alpha1 = {alpha1_low:.4f}, alpha2 = {alpha2_low:.4f}, "
        f"beta = {beta_low:.4f}"
    )

    print(f"\nHigh (a1=a2={a1_high}):")
    print(
        f"  alpha1 = {alpha1_high:.4f}, alpha2 = {alpha2_high:.4f}, "
        f"beta = {beta_high:.4f}"
    )

    # Analyze the bias functions
    x_range = np.linspace(1.1 * r1, 1.1 * r2, 1000)
    bias_values_low = np.array(
        [
            brw_reversion_bias(x, alpha1_low, alpha2_low, beta_low, 0.0)
            for x in x_range
        ]
    )
    bias_values_high = np.array(
        [
            brw_reversion_bias(x, alpha1_high, alpha2_high, beta_high, 0.0)
            for x in x_range
        ]
    )

    # Plot results with your preferred settings
    n_plots = 3
    fig = plt.figure(figsize=(9, 1 + 1.5 * n_plots))
    gs = fig.add_gridspec(3, 2)

    # Time series comparison
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(
        samples_low[:plot_size],
        "b-",
        linewidth=1,
        alpha=0.7,
        label=f"Low (a={a1_low})",
    )
    ax1.plot(
        samples_high[:plot_size],
        "r-",
        linewidth=1,
        alpha=0.7,
        label=f"High (a={a1_high})",
    )
    ax1.axhline(
        y_nop,
        color="g",
        linestyle="--",
        linewidth=2,
        alpha=0.5,
        label=f"y_nop = {y_nop}",
    )
    ax1.axhline(
        y_nop + r1, color="gray", linestyle=":", linewidth=1, alpha=0.5
    )
    ax1.axhline(
        y_nop + r2, color="gray", linestyle=":", linewidth=1, alpha=0.5
    )
    ax1.set_xlabel("Sample")
    ax1.set_ylabel("Value")
    ax1.set_title(f"Bounded Random Walk Comparison ({size:,d} samples)")
    ax1.legend()
    ax1.grid(True)

    # Histograms side by side
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(
        samples_low,
        bins=50,
        density=True,
        alpha=0.7,
        edgecolor="black",
        color="blue",
    )
    ax2.axvline(y_nop, color="g", linestyle="--", linewidth=2, alpha=0.5)
    ax2.axvline(
        y_nop + r1, color="gray", linestyle=":", linewidth=1, alpha=0.5
    )
    ax2.axvline(
        y_nop + r2, color="gray", linestyle=":", linewidth=1, alpha=0.5
    )
    ax2.set_xlabel("Value")
    ax2.set_ylabel("Density")
    ax2.set_title(f"Distribution (a={a1_low})")
    ax2.grid(True)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(
        samples_high,
        bins=50,
        density=True,
        alpha=0.7,
        edgecolor="black",
        color="red",
    )
    ax3.axvline(y_nop, color="g", linestyle="--", linewidth=2, alpha=0.5)
    ax3.axvline(
        y_nop + r1, color="gray", linestyle=":", linewidth=1, alpha=0.5
    )
    ax3.axvline(
        y_nop + r2, color="gray", linestyle=":", linewidth=1, alpha=0.5
    )
    ax3.set_xlabel("Value")
    ax3.set_ylabel("Density")
    ax3.set_title(f"Distribution (a={a1_high})")
    ax3.grid(True)

    # Bias function comparison (shown relative to zero)
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(
        x_range + y_nop,
        bias_values_low,
        "b-",
        linewidth=2,
        label=f"Low (a={a1_low})",
    )
    ax4.plot(
        x_range + y_nop,
        bias_values_high,
        "r-",
        linewidth=2,
        label=f"High (a={a1_high})",
    )
    ax4.axhline(0, color="k", linestyle="-", linewidth=0.5)
    ax4.axhline(1, color="orange", linestyle="--", linewidth=1, alpha=0.5)
    ax4.axhline(-1, color="orange", linestyle="--", linewidth=1, alpha=0.5)
    ax4.axvline(y_nop, color="g", linestyle="--", linewidth=2, alpha=0.5)
    ax4.axvline(
        y_nop + r1, color="gray", linestyle=":", linewidth=1, alpha=0.5
    )
    ax4.axvline(
        y_nop + r2, color="gray", linestyle=":", linewidth=1, alpha=0.5
    )
    ax4.scatter([y_nop + r1, y_nop + r2], [1, -1], color="orange")
    ax4.set_xlabel("x")
    ax4.set_ylabel("$a(x)$ - bias")
    ax4.set_title("Reversion Bias Function Comparison")
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.show()

    print(f"\nLow sample statistics:")
    print(f"  Mean: {np.mean(samples_low):.4f}")
    print(f"  Std: {np.std(samples_low):.4f}")
    print(f"  Min: {np.min(samples_low):.4f}")
    print(f"  Max: {np.max(samples_low):.4f}")

    print(f"\nHigh sample statistics:")
    print(f"  Mean: {np.mean(samples_high):.4f}")
    print(f"  Std: {np.std(samples_high):.4f}")
    print(f"  Min: {np.min(samples_high):.4f}")
    print(f"  Max: {np.max(samples_high):.4f}")
