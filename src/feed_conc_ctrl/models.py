"""State-space models for mixing tank with concentration control."""

import casadi as cas
import numpy as np
from cas_models.continuous_time.models import StateSpaceModelCT
from cas_models.discrete_time.models import StateSpaceModelDTFromCTRK4


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

    def __init__(self, D):
        """Initialize a mixing tank model.

        Args:
            D (float): Tank diameter [m]
        """

        # Compute cross-sectional area from diameter
        # Note: Using A = π * D as in the notebook (not the standard π * D²/4)
        A = float(np.pi * D)

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

        # Initialize parent class
        super().__init__(
            f=f,
            h=h,
            n=2,
            nu=3,
            ny=3,
            params=None,
            name="MixingTankModel",
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

    def __init__(self, D, dt=0.25):
        """Initialize a discrete-time mixing tank model.

        Args:
            D (float): Tank diameter [m]
            dt (float, optional): Time step [hr]. Default: 0.25
        """
        # Create continuous-time model
        model_ct = MixingTankModelCT(D=D)

        # Initialize parent class with RK4 integration
        super().__init__(model_ct, dt)

        # Store tank parameters for easy access
        self.D = D
        self.A = model_ct.A
