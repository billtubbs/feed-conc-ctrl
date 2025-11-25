"""Analysis of numerical stability near v_dot_in ≈ v_dot_out

This script explores the numerical behavior of the analytical solution
when the flow rates are close but not equal.
"""

import numpy as np
import matplotlib.pyplot as plt

def analytical_solution_general(t, L0, m0, v_in, conc_in, v_out, A):
    """General analytical solution (v_in != v_out case)"""
    V0 = A * L0
    V_t = V0 + (v_in - v_out) * t

    # Exponent in the integrating factor
    exponent = v_out / (v_in - v_out)

    # Integrating factor: (V0 + (v_in - v_out)*t)^exponent
    L_t = L0 + (v_in - v_out) * t / A
    m_t = (
        conc_in * V_t
        + (V0**exponent * m0 - conc_in * V0**(v_in / (v_in - v_out)))
        * V_t**(-exponent)
    )

    return L_t, m_t


def analytical_solution_equal(t, L0, m0, v, conc_in, A):
    """Analytical solution for equal flows (v_in = v_out = v)"""
    L_t = L0  # Constant level
    V0 = A * L0
    m_equilibrium = V0 * conc_in
    m_t = m_equilibrium + (m0 - m_equilibrium) * np.exp(-v * t / V0)
    return L_t, m_t


def test_near_equal_flows():
    """Test numerical behavior when v_in is close to v_out"""
    # Parameters
    D = 5
    A = np.pi * D
    L0 = 5
    m0 = 25
    conc_in = 0.1
    t = 1.0

    # Base flow rate
    v_base = 1.0

    # Test different values of delta = v_in - v_out
    deltas = np.logspace(-10, 0, 50)  # From 1e-10 to 1.0

    results = []
    for delta in deltas:
        v_in = v_base + delta / 2
        v_out = v_base - delta / 2

        try:
            L_t, m_t = analytical_solution_general(t, L0, m0, v_in, conc_in, v_out, A)
            results.append({'delta': delta, 'L': L_t, 'm': m_t, 'error': False})
        except (OverflowError, FloatingPointError) as e:
            results.append({'delta': delta, 'L': np.nan, 'm': np.nan, 'error': True})

    # Compare with equal flow solution
    L_eq, m_eq = analytical_solution_equal(t, L0, m0, v_base, conc_in, A)

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Extract data
    deltas_arr = np.array([r['delta'] for r in results])
    m_vals = np.array([r['m'] for r in results])

    # Plot mass values
    ax1.semilogx(deltas_arr, m_vals, 'b.-', label='General solution')
    ax1.axhline(m_eq, color='r', linestyle='--', label=f'Equal flow limit = {m_eq:.4f}')
    ax1.set_xlabel('|v_in - v_out|')
    ax1.set_ylabel('Mass m(t)')
    ax1.set_title('Convergence of General Solution to Equal-Flow Case')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot relative error
    rel_error = np.abs((m_vals - m_eq) / m_eq)
    ax2.loglog(deltas_arr, rel_error, 'b.-')
    ax2.set_xlabel('|v_in - v_out|')
    ax2.set_ylabel('Relative Error')
    ax2.set_title('Numerical Error vs Flow Difference')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('numerical_stability_analysis.png', dpi=150)
    print("Plot saved as 'numerical_stability_analysis.png'")

    # Print key statistics
    print("\n" + "="*70)
    print("Numerical Stability Analysis")
    print("="*70)
    print(f"Equal flow solution: m = {m_eq:.6f}")
    print(f"\nFlow difference | Mass value | Rel. Error")
    print("-" * 50)
    for i, r in enumerate(results[::5]):  # Print every 5th result
        if not r['error']:
            err = abs((r['m'] - m_eq) / m_eq)
            print(f"{r['delta']:.2e}        | {r['m']:.6f}  | {err:.2e}")
        else:
            print(f"{r['delta']:.2e}        | ERROR      | ---")

    # Find threshold where relative error exceeds 1%
    good_indices = ~np.isnan(m_vals) & (rel_error < 0.01)
    if np.any(good_indices):
        min_safe_delta = deltas_arr[good_indices].min()
        print(f"\nMinimum |v_in - v_out| for <1% error: {min_safe_delta:.2e}")
    else:
        print("\nNo values achieved <1% relative error")

    # Check if CasADi's if_else threshold is appropriate
    print(f"\nCurrent if_else threshold in code: 1e-10")
    idx_threshold = np.argmin(np.abs(deltas_arr - 1e-10))
    if not np.isnan(m_vals[idx_threshold]):
        err_at_threshold = rel_error[idx_threshold]
        print(f"Relative error at threshold: {err_at_threshold:.2e}")
        if err_at_threshold < 1e-6:
            print("✓ Threshold appears appropriate")
        else:
            print(f"⚠ Consider increasing threshold to ~{deltas_arr[np.argmax(rel_error < 1e-6)]:.2e}")


def compare_exponentiation_methods():
    """Compare different ways to compute the integrating factor"""
    print("\n" + "="*70)
    print("Exponentiation Methods Comparison")
    print("="*70)

    # Test case where exponent is very large
    base = 15.708  # A * L
    v_out = 1.0
    deltas = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5])

    print(f"\nBase = {base}, v_out = {v_out}")
    print(f"\n{'Delta':<12} {'Exponent':<15} {'base^exp':<20} {'exp(exp*log(base))'}")
    print("-" * 70)

    for delta in deltas:
        exponent = v_out / delta

        # Method 1: Direct power (what's in the analytical solution)
        try:
            result1 = base ** exponent
        except OverflowError:
            result1 = np.inf

        # Method 2: exp(exponent * log(base))
        try:
            result2 = np.exp(exponent * np.log(base))
        except OverflowError:
            result2 = np.inf

        print(f"{delta:<12.2e} {exponent:<15.2e} {result1:<20.4e} {result2:.4e}")

    print("\nNote: Both methods are mathematically equivalent and face the same")
    print("      numerical issues when the exponent becomes very large.")


if __name__ == "__main__":
    test_near_equal_flows()
    compare_exponentiation_methods()
    plt.show()
