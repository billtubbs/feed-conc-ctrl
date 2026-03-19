"""Zero-order hold (ZOH) discretization for measurement filter.

This module provides the ZOH discretization function for computing
discrete-time filter parameters used in the PID controller's
measurement filter.
"""

import math


def zoh_Fy(TfTs, Tx=1.0, exp=math.exp):
    """Compute filter parameters using zero-order hold discretization.

    Args:
        TfTs: Filter time constant as a multiple of nominal sample
              time
        Tx: Execution period (normalized, default: 1.0)
        exp: Exponential function (default: math.exp)

    Returns:
        tuple: Six state-space matrix coefficients
               (a11, a12, a21, a22, b1, b2)
    """
    # Help variables
    h1 = Tx / TfTs
    h2 = exp(-h1)
    h3 = h1 * h2
    h4 = h3 / TfTs

    # Filter parameters
    a11 = h2 + h3
    a12 = h2
    a21 = -h4
    a22 = h2 - h3
    b1 = 1 - h2 - h3
    b2 = h4

    return a11, a12, a21, a22, b1, b2
