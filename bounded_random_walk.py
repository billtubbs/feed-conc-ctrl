import numpy as np


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


def compute_brw_parameters_with_steepness(r1, r2, a1, a2, tau):
    """Compute BRW parameters with independent control.

    This parameterization defines:
    - r1: offset from tau where a(tau + r1) = +1 (typically r1 < 0, below tau)
    - r2: offset from tau where a(tau + r2) = -1 (typically r2 > 0, above tau)
    - a1: (|derivative|) at r1 (positive value)
    - a2: (|derivative|) at r2 (positive value)

    Args:
        r1: Offset from tau where bias = +1 (negative for below tau)
        r2: Offset from tau where bias = -1 (positive for above tau)
        a1: (|derivative|) at r1 (positive value)
        a2: (|derivative|) at r2 (positive value)
        tau: Target/equilibrium value

    Returns:
        tuple: (alpha1, alpha2, beta)
    """
    # From the bias function constraints:
    # a(τ + r1) = exp(β - α₁·r1) - exp(β + α₂·r1) = 1
    # a(τ + r2) = exp(β - α₁·r2) - exp(β + α₂·r2) = -1

    # From the derivative constraints:
    # a'(τ + r1) = -α₁·exp(β - α₁·r1) - α₂·exp(β + α₂·r1)
    # a'(τ + r2) = -α₁·exp(β - α₁·r2) - α₂·exp(β + α₂·r2)

    # For r1 < 0 (below tau): exp(β + α₂·r1) dominates
    # For r2 > 0 (above tau): exp(β - α₁·r2) dominates

    # Approximation approach:
    # At r1: the exp(β + α₂·r1) term dominates both bias and derivative
    # At r2: the exp(β - α₁·r2) term dominates both bias and derivative

    # From dominant terms:
    # At r1: exp(β + α₂·r1) ≈ -1, and -α₂·exp(β + α₂·r1) ≈ a1
    # This gives: α₂ ≈ a1

    # At r2: exp(β - α₁·r2) ≈ -1, and -α₁·exp(β - α₁·r2) ≈ -a2
    # This gives: α₁ ≈ a2

    alpha1 = a2
    alpha2 = a1

    # Solve for β using the bias constraint at r2:
    # exp(β - α₁·r2) - exp(β + α₂·r2) = -1
    # exp(β)[exp(-α₁·r2) - exp(α₂·r2)] = -1

    term = np.exp(-alpha1 * r2) - np.exp(alpha2 * r2)

    if abs(term) < 1e-10:
        # Fallback if term is too small
        print("Warning: Numerical issues in parameter computation")
        alpha1 = 2.0 / abs(r2)
        alpha2 = 2.0 / abs(r1)
        beta = np.log(1.0 / (2.0 * np.sinh(alpha1 * r2)))
    else:
        beta = np.log(-1.0 / term)

    return alpha1, alpha2, beta


def sample_bounded_random_walk_with_steepness(
    sd_e, r1, r2, a1, a2, tau, size, phi=0.5, xkm1=None, rng=None, seed=0
):
    """Simulate bounded random walk with control.

    Args:
        sd_e: Standard deviation of the stochastic noise
        r1: Offset from tau where bias = +1 (typically negative, e.g., -5)
        r2: Offset from tau where bias = -1 (typically positive, e.g., +5)
        a1: (|derivative|) at r1 (positive value)
        a2: (|derivative|) at r2 (positive value)
        tau: Target/equilibrium value
        size: Number of samples to generate
        phi: Regularization parameter (optional, default 0.5)
        xkm1: Initial state value (optional, default tau)
        rng: Random number generator (optional, default None)
        seed: Random seed (optional, default 0)

    Returns:
        Array of size samples from the bounded random walk process
    """
    # Compute original parameters
    alpha1, alpha2, beta = compute_brw_parameters_with_steepness(
        r1, r2, a1, a2, tau
    )

    if rng is None:
        rng = np.random.default_rng(seed=seed)

    # Set initial state if not provided
    if xkm1 is None:
        xkm1 = tau

    # Generate white noise
    e = rng.normal(size=size)
    p = np.zeros(size)

    # Simulate
    for i in range(size):
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


def sample_bounded_random_walk_with_noise(
    sd_e, r1, r2, a1, a2, tau, noise_sequence, phi=0.5, xkm1=None
):
    """Simulate bounded random walk with pre-generated noise sequence.

    Args:
        sd_e: Standard deviation of the stochastic noise
        r1: Offset from tau where bias = +1 (typically negative, e.g., -5)
        r2: Offset from tau where bias = -1 (typically positive, e.g., +5)
        a1: (|derivative|) at r1 (positive value)
        a2: (|derivative|) at r2 (positive value)
        tau: Target/equilibrium value
        noise_sequence: Pre-generated noise array
        phi: Regularization parameter (optional, default 0.5)
        xkm1: Initial state value (optional, default tau)

    Returns:
        Array of samples from the bounded random walk process
    """
    # Compute original parameters
    alpha1, alpha2, beta = compute_brw_parameters_with_steepness(
        r1, r2, a1, a2, tau
    )

    # Set initial state if not provided
    if xkm1 is None:
        xkm1 = tau

    size = len(noise_sequence)
    p = np.zeros(size)

    # Simulate
    for i in range(size):
        # Stochastic input
        alpha = sd_e * noise_sequence[i]

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


# Example usage and testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Common parameters
    sd_e = 1.0
    tau = 100.0
    r1 = -10.0  # When x is 5 units below tau, bias = +1 (pushes up)
    r2 = 10.0  # When x is 5 units above tau, bias = -1 (pushes down)
    size = 100000
    plot_size = 500

    # parameters - symmetric
    a1_low = 1.0
    a2_low = 1.0

    a1_high = 2.0
    a2_high = 2.0

    # Generate single noise sequence to use for both
    rng = np.random.default_rng(seed=42)
    noise_sequence = rng.normal(size=size)

    # Generate both samples with same noise
    samples_low = sample_bounded_random_walk_with_noise(
        sd_e, r1, r2, a1_low, a2_low, tau, noise_sequence
    )
    samples_high = sample_bounded_random_walk_with_noise(
        sd_e, r1, r2, a1_high, a2_high, tau, noise_sequence
    )

    # Compute parameters for both
    alpha1_low, alpha2_low, beta_low = compute_brw_parameters_with_steepness(
        r1, r2, a1_low, a2_low, tau
    )
    alpha1_high, alpha2_high, beta_high = (
        compute_brw_parameters_with_steepness(r1, r2, a1_high, a2_high, tau)
    )

    print(f"Common parameters:")
    print(f"  r1 = {r1}, r2 = {r2}, tau = {tau}")
    print(f"  sd_e = {sd_e}")

    print(f"\nLow (a1=a2={a1_low}):")
    print(
        f"  alpha1 = {alpha1_low:.4f}, alpha2 = {alpha2_low:.4f}, beta = {beta_low:.4f}"
    )

    print(f"\nHigh (a1=a2={a1_high}):")
    print(
        f"  alpha1 = {alpha1_high:.4f}, alpha2 = {alpha2_high:.4f}, beta = {beta_high:.4f}"
    )

    # Analyze the bias functions
    x_range = np.linspace(tau + 1.1 * r1, tau + 1.1 * r2, 1000)
    bias_values_low = np.array(
        [
            brw_reversion_bias(x, alpha1_low, alpha2_low, beta_low, tau)
            for x in x_range
        ]
    )
    bias_values_high = np.array(
        [
            brw_reversion_bias(x, alpha1_high, alpha2_high, beta_high, tau)
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
        tau,
        color="g",
        linestyle="--",
        linewidth=2,
        alpha=0.5,
        label=f"tau = {tau}",
    )
    ax1.axhline(tau + r1, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax1.axhline(tau + r2, color="gray", linestyle=":", linewidth=1, alpha=0.5)
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
    ax2.axvline(tau, color="g", linestyle="--", linewidth=2, alpha=0.5)
    ax2.axvline(tau + r1, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax2.axvline(tau + r2, color="gray", linestyle=":", linewidth=1, alpha=0.5)
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
    ax3.axvline(tau, color="g", linestyle="--", linewidth=2, alpha=0.5)
    ax3.axvline(tau + r1, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax3.axvline(tau + r2, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax3.set_xlabel("Value")
    ax3.set_ylabel("Density")
    ax3.set_title(f"Distribution (a={a1_high})")
    ax3.grid(True)

    # Bias function comparison
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(
        x_range,
        bias_values_low,
        "b-",
        linewidth=2,
        label=f"Low (a={a1_low})",
    )
    ax4.plot(
        x_range,
        bias_values_high,
        "r-",
        linewidth=2,
        label=f"High (a={a1_high})",
    )
    ax4.axhline(0, color="k", linestyle="-", linewidth=0.5)
    ax4.axhline(1, color="orange", linestyle="--", linewidth=1, alpha=0.5)
    ax4.axhline(-1, color="orange", linestyle="--", linewidth=1, alpha=0.5)
    ax4.axvline(tau, color="g", linestyle="--", linewidth=2, alpha=0.5)
    ax4.axvline(tau + r1, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax4.axvline(tau + r2, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax4.scatter([tau + r1, tau + r2], [1, -1], color="orange")
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
