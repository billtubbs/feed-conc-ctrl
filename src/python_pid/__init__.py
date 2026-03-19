"""Python implementation of PID controller reference.

This package provides a reference implementation of a PID controller
based on the work by Sundström et al. (2024), using the incremental
(velocity) form for intrinsic anti-windup and bumpless transfer.

Reference:
    E. Sundström, T. Hägglund, M. Bauer, J. Eker, K. Soltesz,
    Reference Implementation of the PID Controller,
    IFAC-PapersOnLine, Volume 58, Issue 7, 2024, Pages 370-375,
    https://doi.org/10.1016/j.ifacol.2024.08.090
"""

from .anti_windup import WindupMode, anti_windup
from .measurement_filter import MeasurementFilter, filter_update
from .pid import PIDController, pid_update
from .run import PIDRuntime
from .zoh_pid import zoh_Fy

__all__ = [
    "PIDController",
    "pid_update",
    "MeasurementFilter",
    "filter_update",
    "anti_windup",
    "WindupMode",
    "zoh_Fy",
    "PIDRuntime",
]

__version__ = "0.1.0"
