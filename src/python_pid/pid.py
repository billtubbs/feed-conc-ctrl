"""PID controller implementation.

This module provides a PID controller class using the incremental
(velocity) form, which provides intrinsic integrator anti-windup and
bumpless transfer behavior.
"""

from .anti_windup import anti_windup
from .measurement_filter import MeasurementFilter


def pid_update(
    r,
    yf,
    dyf,
    kp,
    ki,
    kd,
    umin,
    umax,
    u0,
    b,
    u_old,
    up_old,
    ud_old,
    uff_old,
    uff=0.0,
    uman=0.0,
    utrack=0.0,
    Tx=1.0,
    track=False,
    auto=True,
    windup=None,
):
    """PID control signal update.

    Pure function to implement the incremental PID algorithm.

    Args:
        r: Reference (setpoint) signal
        yf: Filtered measurement
        dyf: Filtered derivative of measurement
        kp: Proportional gain
        ki: Integral gain
        kd: Derivative gain
        umin: Minimum control signal
        umax: Maximum control signal
        u0: Bias term for P or PD control
        b: Setpoint weight for proportional term
        u_old: Previous control signal
        up_old: Previous proportional component
        ud_old: Previous derivative component
        uff_old: Previous feedforward signal
        uff: Feedforward control signal (default: 0.0)
        uman: Manual mode control signal (default: 0.0)
        utrack: Tracking signal for bumpless transfer (default: 0.0)
        Tx: Execution period normalized (default: 1.0)
        track: Tracking mode flag (default: False)
        auto: Automatic mode flag (default: True)
        windup: Windup status ("upper", "lower", "both", or None)

    Returns:
        tuple: (u, u_old, up_old, ud_old, uff_old, b)
               Control signal and updated state values
    """
    if auto:
        # Reset state if using P or PD control (ki == 0)
        if ki == 0:
            u_old = u0  # Bias term if P or PD control
            up_old = 0.0
            ud_old = 0.0
            uff_old = 0.0
            b = 1.0

        # Tracking mode for bumpless transfer
        if track:
            u_old = utrack
            up_old = 0.0
            ud_old = 0.0
            uff_old = 0.0

        # Control signal increments
        Dup = kp * (b * r - yf) - up_old
        Dui = ki * (r - yf) * Tx
        Dui = anti_windup(Dui, windup)
        Dud = (-kd * dyf - ud_old) / Tx
        Duff = uff - uff_old

        # Add control signal increment
        Du = Dup + Dui + Dud + Duff
        u = u_old + Du
    else:
        # Manual control signal
        u = uman

    # Saturate control signal
    u = max(min(u, umax), umin)

    # Compute new state values (to be used in next iteration)
    u_old = u  # Store saturated control signal
    up_old = kp * (b * r - yf)
    ud_old = -kd * dyf
    uff_old = uff

    return u, u_old, up_old, ud_old, uff_old, b


class PIDController:
    """PID controller using incremental (velocity) form.

    The incremental form computes control signal increments rather than
    absolute values, providing natural anti-windup and bumpless transfer
    capabilities.
    """

    def __init__(
        self,
        kp,
        ki,
        kd,
        TfTs=10.0,
        umin=-float("inf"),
        umax=float("inf"),
        u0=0.0,
        b=1.0,
    ):
        """Initialize the PID controller.

        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            TfTs: Filter time constant as multiple of nominal sample
                  time (default: 10.0)
            umin: Minimum control signal (default: -inf)
            umax: Maximum control signal (default: +inf)
            u0: Bias term for P or PD control (default: 0.0)
            b: Setpoint weight for proportional term (default: 1.0)
        """
        # Controller parameters
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.umin = umin
        self.umax = umax
        self.u0 = u0
        self.b = b

        # Signal states
        self.u_old = 0.0
        self.up_old = 0.0
        self.ud_old = 0.0
        self.uff_old = 0.0

        # Measurement filter
        self.filter = MeasurementFilter(TfTs=TfTs)

    def __call__(
        self,
        r,
        y,
        uff=0.0,
        uman=0.0,
        utrack=0.0,
        Tx=1.0,
        track=False,
        auto=True,
        windup=None,
    ):
        """Compute the PID control signal.

        Args:
            r: Reference (setpoint) signal
            y: Process measurement
            uff: Feedforward control signal (default: 0.0)
            uman: Manual mode control signal (default: 0.0)
            utrack: Tracking signal for bumpless transfer (default: 0.0)
            Tx: Execution period normalized (default: 1.0)
            track: Tracking mode flag (default: False)
            auto: Automatic mode flag (default: True)
            windup: Windup status ("upper", "lower", "both", or None)

        Returns:
            float: Control signal u
        """
        # Filter updates
        yf, dyf = self.filter(y, Tx)

        # Call standalone PID update function
        u, self.u_old, self.up_old, self.ud_old, self.uff_old, self.b = (
            pid_update(
                r,
                yf,
                dyf,
                self.kp,
                self.ki,
                self.kd,
                self.umin,
                self.umax,
                self.u0,
                self.b,
                self.u_old,
                self.up_old,
                self.ud_old,
                self.uff_old,
                uff,
                uman,
                utrack,
                Tx,
                track,
                auto,
                windup,
            )
        )

        return u

    def reset(self):
        """Reset the controller state."""
        self.u_old = 0.0
        self.up_old = 0.0
        self.ud_old = 0.0
        self.uff_old = 0.0
        self.filter.reset()
