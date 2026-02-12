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


def compute_brw_parameters_simple(r1, r2, tau):
    """Compute BRW parameters from intuitive offset parameters.
    
    This parameterization defines:
    - r1: offset from tau where a(tau + r1) = +1 (typically r1 < 0, below tau)
    - r2: offset from tau where a(tau + r2) = -1 (typically r2 > 0, above tau)
    
    For symmetric bounds: r1 = -r2
    
    Args:
        r1: Offset from tau where bias = +1 (negative for below tau)
        r2: Offset from tau where bias = -1 (positive for above tau)
        tau: Target/equilibrium value
        
    Returns:
        tuple: (alpha1, alpha2, beta)
    """
    # From the equations:
    # exp(β - α₁*r1) - exp(β + α₂*r1) = 1
    # exp(β - α₁*r2) - exp(β + α₂*r2) = -1
    
    # For simplicity and symmetry, assume α₁ = α₂ = α
    # Then:
    # exp(β - α*r1) - exp(β + α*r1) = 1
    # exp(β - α*r2) - exp(β + α*r2) = -1
    
    # Factor out:
    # exp(β)[exp(-α*r1) - exp(α*r1)] = 1
    # exp(β)[exp(-α*r2) - exp(α*r2)] = -1
    
    # Using sinh: exp(-x) - exp(x) = -2*sinh(x)
    # So: -2*exp(β)*sinh(α*r1) = 1  =>  exp(β)*sinh(α*r1) = -1/2
    #     -2*exp(β)*sinh(α*r2) = -1  =>  exp(β)*sinh(α*r2) = 1/2
    
    # This gives us:
    # sinh(α*r1) / sinh(α*r2) = -1
    # So: sinh(α*r1) = -sinh(α*r2)
    # Which means: α*r1 = -α*r2  (approximately, for symmetric case)
    
    # For the symmetric case where r1 = -r2:
    if np.abs(r1 + r2) < 1e-6:  # Symmetric case
        # sinh(α*r2) = 1/(2*exp(β))
        # We need to choose α and β
        # Let's pick α based on the scale of r2
        alpha = 2.0 / np.abs(r2)  # Rule of thumb: want significant change over distance r2
        
        # Then solve for β:
        # exp(β) = 1 / (2 * sinh(α * r2))
        beta = np.log(1.0 / (2.0 * np.sinh(alpha * r2)))
        
        alpha1 = alpha2 = alpha
        
    else:  # Asymmetric case - more complex
        # Use numerical approach or make approximation
        # Simple approximation: use average
        r_avg = (np.abs(r1) + np.abs(r2)) / 2.0
        alpha = 2.0 / r_avg
        
        # Use r2 for β calculation (could also use r1 or average)
        beta = np.log(1.0 / (2.0 * np.sinh(alpha * r2)))
        
        # Adjust alpha1 and alpha2 for asymmetry
        ratio = np.abs(r2 / r1)
        alpha1 = alpha * np.sqrt(ratio)
        alpha2 = alpha / np.sqrt(ratio)
    
    return alpha1, alpha2, beta


def sample_bounded_random_walk_simple(
    sd_e, r1, r2, tau, size, phi=0.5, xkm1=None, rng=None, seed=0
):
    """Simulate bounded random walk with simple intuitive parameters.
    
    Args:
        sd_e: Standard deviation of the stochastic noise
        r1: Offset from tau where bias = +1 (typically negative, e.g., -5)
        r2: Offset from tau where bias = -1 (typically positive, e.g., +5)
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
    alpha1, alpha2, beta = compute_brw_parameters_simple(r1, r2, tau)
    
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


# Example usage and testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Simple intuitive parameters
    sd_e = 1.0
    tau = 100.0
    r1 = -5.0  # When x is 5 units below tau, bias = +1 (pushes up)
    r2 = 5.0   # When x is 5 units above tau, bias = -1 (pushes down)
    size = 500
    
    # Compute the equivalent original parameters
    alpha1, alpha2, beta = compute_brw_parameters_simple(r1, r2, tau)
    
    print(f"Simple parameters:")
    print(f"  r1 = {r1} (bias = +1 at x = {tau + r1})")
    print(f"  r2 = {r2} (bias = -1 at x = {tau + r2})")
    print(f"  tau = {tau}")
    
    print(f"\nComputed original parameters:")
    print(f"  alpha1 = {alpha1:.4f}")
    print(f"  alpha2 = {alpha2:.4f}")
    print(f"  beta = {beta:.4f}")
    
    # Verify the parameterization
    bias_at_r1 = brw_reversion_bias(tau + r1, alpha1, alpha2, beta, tau)
    bias_at_r2 = brw_reversion_bias(tau + r2, alpha1, alpha2, beta, tau)
    print(f"\nVerification:")
    print(f"  a(tau + r1) = a({tau + r1}) = {bias_at_r1:.6f} (should be 1.0)")
    print(f"  a(tau + r2) = a({tau + r2}) = {bias_at_r2:.6f} (should be -1.0)")
    
    # Generate sample
    rng = np.random.default_rng(seed=42)
    samples = sample_bounded_random_walk_simple(sd_e, r1, r2, tau, size, rng=rng)
    
    # Analyze the bias function
    x_range = np.linspace(tau + 1.5*r1, tau + 1.5*r2, 1000)
    bias_values = np.array([brw_reversion_bias(x, alpha1, alpha2, beta, tau) 
                            for x in x_range])
    
    # Plot results
    fig = plt.figure(figsize=(9, 7))
    gs = fig.add_gridspec(3, 2)
    
    # Time series
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(samples, 'b-', linewidth=1, alpha=0.7)
    ax1.axhline(tau, color='g', linestyle='--', linewidth=2, alpha=0.5, label=f'tau = {tau}')
    ax1.axhline(tau + r1, color='r', linestyle=':', linewidth=1, 
                label=f'tau + r1 = {tau + r1} (bias = +1)')
    ax1.axhline(tau + r2, color='r', linestyle=':', linewidth=1, 
                label=f'tau + r2 = {tau + r2} (bias = -1)')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Value')
    ax1.set_title('Bounded Random Walk')
    ax1.legend()
    ax1.grid(True)
    
    # Histogram
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(samples, bins=50, density=True, alpha=0.7, edgecolor='black')
    ax2.axvline(tau, color='g', linestyle='--', linewidth=2, alpha=0.5)
    ax2.axvline(tau + r1, color='r', linestyle=':', linewidth=2)
    ax2.axvline(tau + r2, color='r', linestyle=':', linewidth=2)
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution of Samples')
    ax2.grid(True)
    
    # Bias function
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(x_range, bias_values, 'r-', linewidth=2)
    ax3.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax3.axhline(1, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    ax3.axhline(-1, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    ax3.axvline(tau, color='g', linestyle='--', linewidth=2, alpha=0.5)
    ax3.axvline(tau + r1, color='r', linestyle=':', linewidth=2, label='tau + r1')
    ax3.axvline(tau + r2, color='r', linestyle=':', linewidth=2, label='tau + r2')
    ax3.scatter([tau + r1, tau + r2], [1, -1], color='red')
    ax3.set_xlabel('x')
    ax3.set_ylabel('$a(x)$ - bias')
    ax3.set_title('Reversion Bias Function')
    ax3.legend()
    ax3.grid(True)
    
    # Sample ACF
    ax4 = fig.add_subplot(gs[2, :])
    from numpy import correlate
    lags = range(50)
    samples_centered = samples - np.mean(samples)
    acf = [correlate(samples_centered, np.roll(samples_centered, lag))[0] / 
           correlate(samples_centered, samples_centered)[0] 
           for lag in lags]
    ax4.stem(lags, acf, basefmt=' ')
    ax4.set_xlabel('Lag')
    ax4.set_ylabel('Autocorrelation')
    ax4.set_title('Sample Autocorrelation Function')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nSample statistics:")
    print(f"  Mean: {np.mean(samples):.4f}")
    print(f"  Std: {np.std(samples):.4f}")
    print(f"  Min: {np.min(samples):.4f}")
    print(f"  Max: {np.max(samples):.4f}")