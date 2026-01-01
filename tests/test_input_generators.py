"""Tests for input_generators module.

Replicates MATLAB tests for sample_bounded_random_walk.m and
brw_reversion_bias.m
"""

import numpy as np
from feed_conc_ctrl.input_generators import (
    brw_reversion_bias,
    sample_bounded_random_walk,
)


def test_brw_reversion_bias():
    """Test brw_reversion_bias function."""
    # Example from paper
    beta = -15  # notation change beta = k from Nicolau paper
    alpha1 = 3
    alpha2 = 3
    tau = 100

    x = np.linspace(94, 106, 11)
    # Values from Python code Bounded-random-walk-demo.ipynb
    a_test = np.array(
        [
            2.00855369e01,
            5.48811636e-01,
            1.49955768e-02,
            4.09734751e-04,
            1.11871265e-05,
            0.00000000e00,
            -1.11871265e-05,
            -4.09734751e-04,
            -1.49955768e-02,
            -5.48811636e-01,
            -2.00855369e01,
        ]
    )
    a = brw_reversion_bias(x, alpha1, alpha2, beta, tau)
    assert np.max(np.abs(a - a_test)) < 1e-6


def test_sample_bounded_random_walk():
    """Test sample_bounded_random_walk with default arguments."""
    # Example from paper
    np.random.seed(0)
    beta = -15
    alpha1 = 3
    alpha2 = 3
    tau = 100
    sd_e = 0.4
    n = 2000

    p = sample_bounded_random_walk(sd_e, beta, alpha1, alpha2, n, tau)
    assert np.all(p < 106) and np.all(p > 94)

    # Expected values from Python (NumPy RNG differs from MATLAB)
    expected = np.array(
        [
            100.7056,
            100.8657,
            101.2572,
            102.1535,
            102.9003,
            102.5076,
            102.8871,
            102.8248,
            102.7820,
            102.9450,
        ]
    )
    np.testing.assert_array_almost_equal(p[:10], expected, decimal=4)


def test_sample_bounded_random_walk_with_phi_and_initial():
    """Test sample_bounded_random_walk with phi and initial state."""
    # Run again with phi argument and non-zero initial state
    np.random.seed(0)
    beta = -15
    alpha1 = 3
    alpha2 = 3
    tau = 100
    sd_e = 0.4
    n = 2000
    phi = 0.5
    xkm1 = 95

    p = sample_bounded_random_walk(
        sd_e, beta, alpha1, alpha2, n, tau, phi, xkm1
    )
    assert np.all(p < 106) and np.all(p > 94)

    # Expected values from Python (NumPy RNG differs from MATLAB)
    expected = np.array(
        [
            96.7056,
            96.8717,
            97.2668,
            98.1643,
            98.9114,
            98.5205,
            98.9005,
            98.8400,
            98.7987,
            98.9630,
        ]
    )
    np.testing.assert_array_almost_equal(p[:10], expected, decimal=4)
