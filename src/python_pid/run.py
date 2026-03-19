"""Runtime execution loop for PID controller.

This module demonstrates how to integrate the PID controller into a
control system with proper timing and signal handling.
"""

import time


class PIDRuntime:
    """Runtime environment for executing PID controller.

    This class manages the execution loop, timing, and signal I/O for
    the PID controller.
    """

    def __init__(self, controller, Ts=1.0):
        """Initialize the runtime environment.

        Args:
            controller: PIDController instance
            Ts: Sample time in seconds (default: 1.0)
        """
        self.Ts = Ts
        self.controller = controller

        # Runtime state
        self.stop = False
        self.auto = True
        self.track = False
        self.windup = None
        self.t_old = None

    def get_r(self):
        """Read reference (setpoint) signal from environment.

        This method should be overridden to read from actual hardware
        or runtime environment.

        Returns:
            float: Reference signal
        """
        return 0.0

    def get_y(self):
        """Read process measurement from environment.

        This method should be overridden to read from actual hardware
        or runtime environment.

        Returns:
            float: Process measurement
        """
        return 0.0

    def get_uff(self):
        """Read feedforward signal from environment.

        This method should be overridden to read from actual hardware
        or runtime environment.

        Returns:
            float: Feedforward control signal
        """
        return 0.0

    def get_utrack(self):
        """Read tracking signal from environment.

        This method should be overridden to read from actual hardware
        or runtime environment.

        Returns:
            float: Tracking signal
        """
        return 0.0

    def get_uman(self):
        """Read manual control signal from environment.

        This method should be overridden to read from actual hardware
        or runtime environment.

        Returns:
            float: Manual mode control signal
        """
        return 0.0

    def set_u(self, u):
        """Send control signal to environment.

        This method should be overridden to write to actual hardware
        or runtime environment.

        Args:
            u: Control signal to send
        """
        pass

    def run(self):
        """Execute the PID control loop.

        Runs continuously until stop is set to True. Maintains the
        specified sample time and handles timing variations.
        """
        self.t_old = time.time()

        while not self.stop:
            # Time when loop starts
            t0 = time.time()

            # Read signals from runtime or hardware
            r = self.get_r()
            y = self.get_y()
            uff = self.get_uff()
            utrack = self.get_utrack()
            uman = self.get_uman()

            # Compute time between two executions
            t = time.time()
            Tx = (t - self.t_old) / self.Ts

            # Invoke the PID update
            u = self.controller(
                r, y, uff, uman, utrack, Tx, self.track, self.auto, self.windup
            )

            # Send control signal
            self.set_u(u)

            # State update
            self.t_old = t

            # Sleep for remaining time (non-blocking)
            elapsed = time.time() - t0
            sleep_time = self.Ts - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
