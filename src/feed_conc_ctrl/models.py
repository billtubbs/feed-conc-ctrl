"""State-space models for mixing tank with concentration control."""

from itertools import chain
import casadi as cas
import numpy as np
from cas_models.continuous_time.models import StateSpaceModelCT
from cas_models.discrete_time.models import (
    StateSpaceModelDT,
    StateSpaceModelDTFromCTRK4,
)


class MixingTankModelCT(StateSpaceModelCT):
    """
    Continuous-time state-space model for a mixing tank with variable
    inlet density, inlet flow rate and exit flow rate.

    This model represents the dynamics of a surge tank with mass concentration
    where both the volume and concentration can change over time.

    States:
        x[0]: Tank level, L [m]
        x[1]: Total mass of suspended mineral in tank, m [tons]

    Inputs:
        u[0]: Volumetric flowrate into tank, v_dot_in [m^3/hr]
        u[1]: Density of fluid entering tank, conc_in [tons/m^3]
        u[2]: Volumetric flowrate out of tank, v_dot_out [m^3/hr]

    Outputs:
        y[0]: Tank level, L [m]
        y[1]: Total mass of suspended mineral in tank, m [tons]
        y[2]: Concentration/density of outflow, conc_out [tons/m^3]

    Differential Equations:
        dL/dt = (v_dot_in - v_dot_out) / A
        dm/dt = v_dot_in * conc_in - v_dot_out * m / (A * L)

    Notes:
        This is a continuous-time model. For discrete-time simulation or use
        in MPC, create a discrete-time version using:
        ```
        from cas_models.discrete_time.models import StateSpaceModelDTFromCTRK4
        model_dt = StateSpaceModelDTFromCTRK4(model_ct, dt)
        ```

        The RK4 integration approach provides:
        - Smooth derivatives (excellent for optimization with IPOPT/SNOPT)
        - No numerical overflow issues
        - No need for conditional if_else statements
        - Tunable accuracy via choice of dt
    """

    def __init__(self, D=None, A=None, name="MixingTankModel"):
        """Initialize a mixing tank model.

        Args:
            D (float, optional): Tank diameter [m]
            A (float, optional): Tank cross-sectional area [m^2]
            name (str, optional): Give the tank a name (default: "MixingTankModel").

        Note:
            Exactly one of D or A must be provided.
            If D is provided, A is calculated as A = π * D² / 4
        """

        # Validate inputs - exactly one must be provided
        if D is None and A is None:
            raise ValueError("Must provide either D (diameter) or A (area)")
        if D is not None and A is not None:
            raise ValueError("Cannot provide both D and A, choose one")

        # Compute area from diameter if D is provided
        if D is not None:
            # A = π * r² = π * (D/2)² = π * D² / 4
            A = float(np.pi * (D**2) / 4)
        else:
            # A was provided directly
            A = float(A)
            D = None  # Diameter is unknown

        # Define symbolic variables
        t = cas.SX.sym("t")
        x = cas.SX.sym("x", 2)
        u = cas.SX.sym("u", 3)

        # Differential equations (ODE right-hand side)
        dL_dt = (u[0] - u[2]) / A
        dm_dt = u[0] * u[1] - u[2] * x[1] / (x[0] * A)
        rhs = cas.vertcat(dL_dt, dm_dt)

        # State transition function (ODE)
        f = cas.Function("f", [t, x, u], [rhs], ["t", "x", "u"], ["rhs"])

        # Output function (states plus outflow concentration)
        # conc_out = m / (A * L) = mass / volume
        conc_out = x[1] / (A * x[0])
        y = cas.vertcat(x[0], x[1], conc_out)
        h = cas.Function("h", [t, x, u], [y], ["t", "x", "u"], ["y"])

        super().__init__(
            f=f,
            h=h,
            n=2,
            nu=3,
            ny=3,
            params=None,
            name=name,
            input_names=["v_dot_in", "conc_in", "v_dot_out"],
            state_names=["L", "m"],
            output_names=["L", "m", "conc_out"],
        )

        # Store tank parameters
        self.D = D
        self.A = A


class MixingTankModelDT(StateSpaceModelDTFromCTRK4):
    """
    Discrete-time state-space model for a mixing tank with variable
    inlet density and flow rate.

    This model uses RK4 integration of the continuous-time mixing tank
    dynamics. It provides smooth derivatives suitable for gradient-based
    optimization and is numerically stable at all flow rate combinations.

    States:
        x[0]: Tank level, L [m]
        x[1]: Total mass of suspended mineral in tank, m [tons]

    Inputs:
        u[0]: Volumetric flowrate into tank, v_dot_in [m^3/hr]
        u[1]: Density of fluid entering tank, conc_in [tons/m^3]
        u[2]: Volumetric flowrate out of tank, v_dot_out [m^3/hr]

    Outputs:
        y[0]: Tank level, L [m]
        y[1]: Total mass of suspended mineral in tank, m [tons]
        y[2]: Concentration/density of outflow, conc_out [tons/m^3]
    """

    def __init__(self, dt, D=None, A=None, name=None):
        """Initialize a discrete-time mixing tank model.

        Args:
            D (float, optional): Tank diameter [m]
            A (float, optional): Tank cross-sectional area [m^2]
            dt (float, optional): Time step [hr]. Default: 0.25

        Note:
            Exactly one of D or A must be provided.
        """
        # Create continuous-time model
        model_ct = MixingTankModelCT(D=D, A=A, name=name)

        # Initialize parent class with RK4 integration
        super().__init__(model_ct, dt)

        # Store tank parameters for easy access
        self.D = model_ct.D
        self.A = model_ct.A


class FlowMixerCT(StateSpaceModelCT):
    """
    Continuous-time state-space model for a flow mixer with n inlet streams
    and one outlet stream.

    States:
        None

    Inputs:
        u[0]: Volumetric flowrate into mixer from stream 1, v_dot_1 [m^3/hr]
        u[1]: Density of fluid entering mixer from stream 1, conc_1 [tons/m^3]
        ...
        u[n-2]: Volumetric flowrate into mixer from stream n-1, v_dot_n-1 [m^3/hr]
        u[n-1]: Density of fluid entering mixer from stream n-1, conc_n-1 [tons/m^3]

    Outputs:
        y[0]: Volumetric flowrate out of mixer, v_dot_out [m^3/hr]
        y[1]: Density of fluid exiting mixer, conc_out [tons/m^3]
    """

    def __init__(self, n_in=2, name="FlowMixerModel"):
        """Initialize a flow mixer model.

        Args:
            n_in (int, optional): Number of inlet streams (default: 2)
            name (str, optional): Give the mixer a name (default: "FlowMixerModel").
        """

        if n_in < 2:
            raise ValueError("n_in must be at least 2")

        input_names = list(
            chain.from_iterable(
                [f"v_dot_in_{i + 1}", f"conc_in_{i + 1}"] for i in range(n_in)
            )
        )
        output_names = ["v_dot_out", "conc_out"]

        # Dimensions
        n = 0  # No states
        nu = 2 * n_in
        ny = 2

        # Define symbolic variables
        t = cas.SX.sym("t")
        x = cas.SX.sym("x", n)
        u = cas.SX.sym("u", nu)

        # No dynamics
        rhs = cas.vertcat()  # Empty

        # State transition function (empty)
        f = cas.Function("f", [t, x, u], [rhs], ["t", "x", "u"], ["rhs"])

        # Sum of all inlet flow rates
        v_dot_out = cas.sum1(u[0::2])

        # Sum of flow rate * concentration
        conc_out_numerator = cas.sum1(u[0::2] * u[1::2])

        # Weighted average concentration
        conc_out = conc_out_numerator / v_dot_out

        # Output function
        y = cas.vertcat(v_dot_out, conc_out)
        h = cas.Function("h", [t, x, u], [y], ["t", "x", "u"], ["y"])

        super().__init__(
            f=f,
            h=h,
            n=n,
            nu=nu,
            ny=ny,
            params=None,
            name=name,
            input_names=input_names,
            state_names=[],
            output_names=output_names,
        )


class FlowMixerDT(StateSpaceModelDT):
    """
    Discrete-time state-space model for a flow mixer with n inlet streams
    and one outlet stream.

    States:
        None

    Inputs:
        uk[0]: Volumetric flowrate into mixer from stream 1, v_dot_1 [m^3/hr]
        uk[1]: Density of fluid entering mixer from stream 1, conc_1 [tons/m^3]
        ...
        uk[n-2]: Volumetric flowrate into mixer from stream n-1, v_dot_n-1 [m^3/hr]
        uk[n-1]: Density of fluid entering mixer from stream n-1, conc_n-1 [tons/m^3]

    Outputs:
        yk[0]: Volumetric flowrate out of mixer, v_dot_out [m^3/hr]
        yk[1]: Density of fluid exiting mixer, conc_out [tons/m^3]
    """

    def __init__(self, dt, n_in, name=None):
        """Initialize a discrete-time flow mixer model.

        Args:
            dt (float): Time step
            n_in (int): Number of inlet streams
            name (str, optional): Name for the mixer
        """
        # Create continuous-time model to get the functions
        model_ct = FlowMixerCT(n_in=n_in, name=name)

        # For a stateless system, discrete-time model is the same as continuous-time
        # Just need to wrap the functions with discrete-time naming conventions
        t = cas.SX.sym("t")
        xk = cas.SX.sym("xk", 0)  # Empty state vector
        uk = cas.SX.sym("uk", model_ct.nu)

        # State transition: empty state stays empty
        F = cas.Function("F", [t, xk, uk], [xk], ["t", "xk", "uk"], ["xkp1"])

        # Output function: same as continuous-time
        yk = model_ct.h(t, xk, uk)
        H = cas.Function("H", [t, xk, uk], [yk], ["t", "xk", "uk"], ["yk"])

        super().__init__(
            F=F,
            H=H,
            n=0,
            nu=model_ct.nu,
            ny=model_ct.ny,
            dt=dt,
            params=None,
            name=name or model_ct.name,
            input_names=model_ct.input_names,
            state_names=[],
            output_names=model_ct.output_names,
        )
