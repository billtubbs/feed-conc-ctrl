"""Measurement filter for PID controller.

This module provides a second-order filter for filtering the process
measurement and computing its filtered derivative for use in the
D-term of the PID controller.
"""

from .zoh_pid import zoh_Fy


def filter_update(
    y, yf, dyf, a11, a12, a21, a22, b1, b2, TfTs, Tx=1.0, Tx_old=None
):
    """Pure function for measurement filter update.

    This is a standalone function implementing critically-damped
    second-order filter measurement filter with automatic
    re-discretization.

    Args:
        y: Process measurement
        yf: Previous filtered output
        dyf: Previous filtered derivative
        a11: Filter coefficient a11
        a12: Filter coefficient a12
        a21: Filter coefficient a21
        a22: Filter coefficient a22
        b1: Filter coefficient b1
        b2: Filter coefficient b2
        TfTs: Filter time constant as multiple of nominal sample time
        Tx: Execution period (normalized, default: 1.0)
        Tx_old: Previous execution period (None if first call)

    Returns:
        tuple: (yf, dyf, a11, a12, a21, a22, b1, b2, Tx_old)
               Filtered output, derivative, updated coefficients, and
               Tx_old
    """
    # Rediscretize to match execution period
    if Tx != Tx_old:
        a11, a12, a21, a22, b1, b2 = zoh_Fy(TfTs, Tx)

    # State update
    yf_prev = yf
    Tx_old = Tx
    yf = a11 * yf_prev + a12 * dyf + b1 * y
    dyf = a21 * yf_prev + a22 * dyf + b2 * y

    return yf, dyf, a11, a12, a21, a22, b1, b2, Tx_old


class MeasurementFilter:
    """Second-order measurement filter with automatic re-discretization.

    The filter provides both filtered output and filtered derivative, with
    automatic parameter adjustment when the execution period changes.
    """

    def __init__(self, TfTs=10.0):
        """Initialize the measurement filter.

        Args:
            TfTs: Filter time constant as a multiple of nominal sample
                  time (default: 10.0)
        """
        # Filter time constant parameter
        self.TfTs = TfTs

        # Filter parameters (initially None, computed on first call)
        self.a11 = None
        self.a12 = None
        self.a21 = None
        self.a22 = None
        self.b1 = None
        self.b2 = None

        # Filter state
        self.yf = 0.0
        self.dyf = 0.0
        self.Tx_old = None

    def __call__(self, y, Tx=1.0):
        """Apply the filter to a measurement.

        Args:
            y: Process measurement
            Tx: Execution period (normalized, default: 1.0)

        Returns:
            tuple: (yf, dyf) - filtered output and filtered derivative
        """
        # Call standalone filter update function
        result = filter_update(
            y,
            self.yf,
            self.dyf,
            self.a11,
            self.a12,
            self.a21,
            self.a22,
            self.b1,
            self.b2,
            self.TfTs,
            Tx,
            self.Tx_old,
        )
        (
            self.yf,
            self.dyf,
            self.a11,
            self.a12,
            self.a21,
            self.a22,
            self.b1,
            self.b2,
            self.Tx_old,
        ) = result

        return self.yf, self.dyf

    def reset(self):
        """Reset the filter state to zero."""
        self.yf = 0.0
        self.dyf = 0.0
        self.Tx_old = None
