import casadi as cas
import numpy as np
from cas_models.discrete_time.models import StateSpaceModelDT


def rk4_step(f, t, x, u, h, *params):
    """Single Runge-Kutta 4th order integration step.

    Args:
        f: Function with signature f(t, x, u, *params) -> dx/dt
        t: Current time
        x: Current state
        u: Current input
        h: Step size
        *params: Additional parameters to pass to f

    Returns:
        x_next: State at t + h
    """
    k1 = f(t, x, u, *params)
    k2 = f(t + h/2, x + h/2*k1, u, *params)
    k3 = f(t + h/2, x + h/2*k2, u, *params)
    k4 = f(t + h, x + h*k3, u, *params)

    x_next = x + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    return x_next


class MixingTankModel(StateSpaceModelDT):
    """
    State-space model for a mixing tank with variable inlet density
    and flow rate.

    This model uses an analytical solution for the discrete-time state
    transition of a tank with mass concentration.

    States:
        x[0]: Tank level, L [m]
        x[1]: Total mass of suspended mineral in tank, m [tons]

    Inputs:
        u[0]: Volumetric flowrate into tank, v_dot_in [m^3/hr]
        u[1]: Density of fluid entering tank, rho_in [tons/m^3]
        u[2]: Volumetric flowrate out of tank, v_dot_out [m^3/hr]

    Outputs:
        y[0]: Tank level, L [m]
        y[1]: Total mass of suspended mineral in tank, m [tons]
        y[2]: Density of outflow, conc_out [tons/m^3]

    Notes:
        The analytical solution uses different formulas depending on whether
        the inlet and outlet flow rates are approximately equal:
        - When |v_dot_in - v_dot_out| >= 1% of average flow: general formula
        - When |v_dot_in - v_dot_out| < 1% of average flow: equal-flow formula

        This switching is necessary because the general analytical solution
        involves exponents of the form base^(v_out/(v_in - v_out)), which
        causes numerical overflow when the flow difference is small.
    """

    def __init__(self, D, dt=1.0):
        """Initialize a mixing tank model.

        Args:
            D (float): Tank diameter [m]
            dt (float, optional): Time step [hr]. Default: 1.0
        """

        # Compute cross-sectional area from diameter
        # Note: Using A = π * D as in the notebook (not the standard π * D²/4)
        A = float(np.pi * D)

        # Define symbolic variables
        t = cas.SX.sym("t")
        L = cas.SX.sym("L")
        m = cas.SX.sym("m")
        v_dot_in = cas.SX.sym("v_dot_in")
        conc_in = cas.SX.sym("conc_in")
        v_dot_out = cas.SX.sym("v_dot_out")

        # State and input vectors
        xk = cas.vertcat(L, m)
        uk = cas.vertcat(v_dot_in, conc_in, v_dot_out)

        # Analytical solution for one-step state transition
        # Based on ivp_solution from the notebook

        # Level equation (same for both cases)
        L_next = L + dt * (v_dot_in - v_dot_out) / A

        # Mass equation - need to handle two cases:
        # Case 1: v_dot_in != v_dot_out (general case)
        m_next_general = conc_in * (A * L + dt * v_dot_in - dt * v_dot_out) + (
            (A * L) ** (v_dot_out / (v_dot_in - v_dot_out)) * m
            - conc_in
            * cas.exp(v_dot_in * cas.log(A * L) / (v_dot_in - v_dot_out))
        ) * cas.exp(
            -v_dot_out
            * cas.log(A * L + dt * v_dot_in - dt * v_dot_out)
            / (v_dot_in - v_dot_out)
        )

        # Case 2: v_dot_in == v_dot_out (equal flows, constant level)
        # dm/dt = v_dot * (conc_in - m/(L*A))
        # Solution: m(t) = L*A*conc_in + (m_0 - L*A*conc_in) * exp(-v_dot*t/(L*A))
        m_next_equal = (
            L * A * conc_in + (m - L * A * conc_in) * cas.exp(-v_dot_in * dt / (L * A))
        )

        # Use conditional to select appropriate formula
        # Threshold chosen based on numerical stability analysis:
        # - General formula overflows when |v_in - v_out| < ~0.001
        # - Use relative threshold: 1% of the average flow rate
        # - This avoids numerical issues while maintaining accuracy
        v_avg = (v_dot_in + v_dot_out) / 2
        eps_rel = 0.01  # 1% relative tolerance
        m_next = cas.if_else(
            cas.fabs(v_dot_in - v_dot_out) < eps_rel * v_avg,
            m_next_equal,
            m_next_general
        )

        xkp1 = cas.vertcat(L_next, m_next)

        # State transition function
        F = cas.Function(
            "F",
            [t, xk, uk],
            [xkp1],
            ["t", "xk", "uk"],
            ["xkp1"],
        )

        # Output function (states plus outflow concentration)
        # conc_out = m / (A * L) = mass / volume
        conc_out = m / (A * L)
        yk = cas.vertcat(L, m, conc_out)
        H = cas.Function(
            "H",
            [t, xk, uk],
            [yk],
            ["t", "xk", "uk"],
            ["yk"],
        )

        # Define dimensions
        n = 2  # Number of states
        nu = 3  # Number of inputs
        ny = 3  # Number of outputs

        # Initialize parent class
        super().__init__(
            F=F,
            H=H,
            n=n,
            nu=nu,
            ny=ny,
            dt=dt,
            params=None,
            name="MixingTankModel",
            input_names=["v_dot_in", "conc_in", "v_dot_out"],
            state_names=["L", "m"],
            output_names=["L", "m", "conc_out"],
        )

        # Store tank parameters
        self.D = D
        self.A = A
