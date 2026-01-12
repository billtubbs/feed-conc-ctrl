from pathlib import Path
from collections import defaultdict

import warnings
import numpy as np
import pandas as pd
import casadi as cas
import casadi.tools as castools

# Suppress do_mpc optional feature warnings
warnings.filterwarnings("ignore", category=UserWarning, module="do_mpc.sysid")
warnings.filterwarnings("ignore", category=UserWarning, module="do_mpc.opcua")
warnings.filterwarnings(
    "ignore", category=UserWarning, module="do_mpc.approximateMPC"
)

import do_mpc


def calc_mixing_tank_dynamics(L, m, v_dot_in, conc_in, v_dot_out, D):
    """Calculate rates-of-change of tank level and solids mass.
    L: level
    m: mass
    v_dot_in: inflow rate
    conc_in: inflow concentration
    v_dot_out: outflow rate
    D: tank diameter
    """
    A = cas.pi * (D**2) / 4
    dL_dt = (v_dot_in - v_dot_out) / A
    conc_out = m / (A * L)
    dm_dt = v_dot_in * conc_in - v_dot_out * conc_out
    return dL_dt, dm_dt


def calc_mixing_tank_conc(L, m, D):
    """Calculate mixing tank solids concentration.
    L: level
    m: mass
    D: tank diameter
    """
    A = cas.pi * (D**2) / 4
    conc_out = m / (A * L)
    return conc_out


def calc_mixer_outputs(v_dot_in_1, conc_in_1, v_dot_in_2, conc_in_2):
    """Calculate mixer outflow rate and solids concentration.
    v_dot_in_1: inflow rate 1
    conc_in_1: inflow concentration 1
    v_dot_in_2: inflow rate 2
    conc_in_2: inflow concentration 2
    """
    # Sum of all inlet flow rates
    v_dot_out = v_dot_in_1 + v_dot_in_2

    # Weighted average concentration: sum(v_i * c_i) / sum(v_i)
    conc_out = (v_dot_in_1 * conc_in_1 + v_dot_in_2 * conc_in_2) / v_dot_out

    return v_dot_out, conc_out


def construct_4_tank_system_model(D):
    """Construct Do-MPC model of mixing tank system

    D: Tank diameter (m)

    4-Tank System

                          ┌────────┐
                       ┌─►┤ Tank 2 │
                       │  │        │
                       │  │        ├──┐
                       │  └────────┘  │
         ┌────────┐    │              │     ┌────────┐
    ────►┤ Tank 1 │    │              ├────►┤ Tank 4 │
         │        │    │              │     │        │
         │        ├────┤  ┌────────┐  │     │        ├────►
         └────────┘    └─►┤ Tank 3 │  │     └────────┘
                          │        │  │
                          │        ├──┘
                          └────────┘
    """

    # Tank diameters
    D = cas.DM(D)
    if D.shape[0] == 1:
        D = cas.repmat(D, 4)
    tank_1_D = D[0]
    tank_2_D = D[1]
    tank_3_D = D[2]
    tank_4_D = D[3]

    # Control Model Design

    model_type = "continuous"
    model = do_mpc.model.Model(model_type)

    # Manipulated Variables (MVs)
    #  1. Tank 2 inflow rate
    #  2. Tank 3 inflow rate
    #  3. Mixer inflow 1 rate
    #  4. Mixer inflow 2 rate
    #  5. Tank 4 outflow rate

    tank_2_v_dot_in = model.set_variable(
        var_type="_u", var_name="tank_2_v_dot_in"
    )
    tank_3_v_dot_in = model.set_variable(
        var_type="_u", var_name="tank_3_v_dot_in"
    )
    mixer_v_dot_in_1 = model.set_variable(
        var_type="_u", var_name="mixer_v_dot_in_1"
    )
    mixer_v_dot_in_2 = model.set_variable(
        var_type="_u", var_name="mixer_v_dot_in_2"
    )
    tank_4_v_dot_out = model.set_variable(
        var_type="_u", var_name="tank_4_v_dot_out"
    )

    # States
    #  1. Tank 1 level
    #  2. Tank 1 mass
    #  3. Tank 2 level
    #  4. Tank 2 mass
    #  5. Tank 3 level
    #  6. Tank 3 mass
    #  7. Tank 4 level
    #  8. Tank 4 mass

    tank_1_L = model.set_variable(var_type="_x", var_name="tank_1_L")
    tank_1_m = model.set_variable(var_type="_x", var_name="tank_1_m")
    tank_2_L = model.set_variable(var_type="_x", var_name="tank_2_L")
    tank_2_m = model.set_variable(var_type="_x", var_name="tank_2_m")
    tank_3_L = model.set_variable(var_type="_x", var_name="tank_3_L")
    tank_3_m = model.set_variable(var_type="_x", var_name="tank_3_m")
    tank_4_L = model.set_variable(var_type="_x", var_name="tank_4_L")
    tank_4_m = model.set_variable(var_type="_x", var_name="tank_4_m")

    # Unmeasured Disturbances
    #  1. Tank 1 inflow rate
    #  2. Tank 1 inflow concentration

    tank_1_v_dot_in = model.set_variable(
        var_type="_x", var_name="tank_1_v_dot_in"
    )
    tank_1_conc_in = model.set_variable(
        var_type="_x", var_name="tank_1_conc_in"
    )

    # Output variables
    tank_1_conc_out = calc_mixing_tank_conc(tank_1_L, tank_1_m, tank_1_D)
    tank_2_conc_out = calc_mixing_tank_conc(tank_2_L, tank_2_m, tank_2_D)
    tank_3_conc_out = calc_mixing_tank_conc(tank_3_L, tank_3_m, tank_3_D)
    tank_4_conc_out = calc_mixing_tank_conc(tank_4_L, tank_4_m, tank_4_D)
    model.set_expression("tank_1_conc_out", tank_1_conc_out)
    model.set_expression("tank_2_conc_out", tank_2_conc_out)
    model.set_expression("tank_3_conc_out", tank_3_conc_out)
    model.set_expression("tank_4_conc_out", tank_4_conc_out)

    # Measured outputs
    #  1. Tank 1 level
    #  2. Tank 1 outflow concentration
    #  3. Tank 2 level
    #  4. Tank 2 outflow concentration
    #  5. Tank 3 level
    #  6. Tank 3 outflow concentration
    #  7. Tank 4 level
    #  8. Tank 4 outflow concentration

    model.set_meas(meas_name="tank_1_L_meas", expr=tank_1_L)
    model.set_meas(meas_name="tank_1_conc_out_meas", expr=tank_1_conc_out)
    model.set_meas(meas_name="tank_2_L_meas", expr=tank_2_L)
    model.set_meas(meas_name="tank_2_conc_out_meas", expr=tank_2_conc_out)
    model.set_meas(meas_name="tank_3_L_meas", expr=tank_3_L)
    model.set_meas(meas_name="tank_3_conc_out_meas", expr=tank_3_conc_out)
    model.set_meas(meas_name="tank_4_L_meas", expr=tank_4_L)
    model.set_meas(meas_name="tank_4_conc_out_meas", expr=tank_4_conc_out)

    # State dynamics
    tank_1_v_dot_out = tank_2_v_dot_in + tank_3_v_dot_in
    tank_1_dL_dt, tank_1_dm_dt = calc_mixing_tank_dynamics(
        tank_1_L,
        tank_1_m,
        tank_1_v_dot_in,
        tank_1_conc_in,
        tank_1_v_dot_out,
        tank_1_D,
    )
    model.set_rhs("tank_1_L", tank_1_dL_dt)
    model.set_rhs("tank_1_m", tank_1_dm_dt)

    tank_2_conc_in = tank_1_conc_out
    tank_2_v_dot_out = mixer_v_dot_in_1
    tank_2_dL_dt, tank_2_dm_dt = calc_mixing_tank_dynamics(
        tank_2_L,
        tank_2_m,
        tank_2_v_dot_in,
        tank_2_conc_in,
        tank_2_v_dot_out,
        tank_2_D,
    )
    model.set_rhs("tank_2_L", tank_2_dL_dt)
    model.set_rhs("tank_2_m", tank_2_dm_dt)

    tank_3_conc_in = tank_1_conc_out
    tank_3_v_dot_out = mixer_v_dot_in_2
    tank_3_dL_dt, tank_3_dm_dt = calc_mixing_tank_dynamics(
        tank_3_L,
        tank_3_m,
        tank_3_v_dot_in,
        tank_3_conc_in,
        tank_3_v_dot_out,
        tank_3_D,
    )
    model.set_rhs("tank_3_L", tank_3_dL_dt)
    model.set_rhs("tank_3_m", tank_3_dm_dt)

    mixer_conc_in_1 = tank_2_conc_out
    mixer_conc_in_2 = tank_3_conc_out
    mixer_v_dot_out, mixer_conc_out = calc_mixer_outputs(
        mixer_v_dot_in_1, mixer_conc_in_1, mixer_v_dot_in_2, mixer_conc_in_2
    )

    tank_4_conc_in = mixer_conc_out
    tank_4_v_dot_in = mixer_v_dot_out
    tank_4_dL_dt, tank_4_dm_dt = calc_mixing_tank_dynamics(
        tank_4_L,
        tank_4_m,
        tank_4_v_dot_in,
        tank_4_conc_in,
        tank_4_v_dot_out,
        tank_4_D,
    )
    model.set_rhs("tank_4_L", tank_4_dL_dt)
    model.set_rhs("tank_4_m", tank_4_dm_dt)

    # Dynamics for unmeasured disturbances (assume random walk model)
    model.set_rhs("tank_1_conc_in", cas.DM(0))
    model.set_rhs("tank_1_v_dot_in", cas.DM(0))

    # Time-varying parameters for setpoints
    # These will be used in the cost function
    model.set_variable(var_type="_tvp", var_name="tank_1_L_sp")
    model.set_variable(var_type="_tvp", var_name="tank_2_L_sp")
    model.set_variable(var_type="_tvp", var_name="tank_3_L_sp")
    model.set_variable(var_type="_tvp", var_name="tank_4_L_sp")
    model.set_variable(var_type="_tvp", var_name="tank_4_conc_out_sp")
    model.set_variable(var_type="_tvp", var_name="tank_4_v_dot_out_sp")

    model.setup()

    return model


def cost_function_tracking(controlled_variables, setpoints, weights):
    pred_errors = setpoints - controlled_variables
    cost = cas.sum1(weights * cas.sumsqr(pred_errors))
    return cost


def construct_mpc_controller(
    model,
    t_step,
    n_horizon,
    cv_weights,
    mv_weights,
    v_dot_bounds,
    tank_level_bounds,
    tank_4_conc_out_bounds,
    n_robust=0,
):
    """Create MPC controller for 4-tank mixing system."""
    mpc = do_mpc.controller.MPC(model)

    mpc_params = {
        "n_horizon": n_horizon,  # Prediction horizon (hours)
        "t_step": t_step,  # Time step (hours)
        "n_robust": n_robust,  # Robust horizon
        "store_full_solution": True,
    }
    mpc.set_param(**mpc_params)

    # Construct objective function for setpoint tracking
    controlled_variables = {
        "tank_1_L": model.x["tank_1_L"],
        "tank_2_L": model.x["tank_2_L"],
        "tank_3_L": model.x["tank_3_L"],
        "tank_4_L": model.x["tank_4_L"],
        "tank_4_conc_out": model.aux["tank_4_conc_out"],
        "tank_4_v_dot_out": model.u["tank_4_v_dot_out"],
    }

    # Map controlled variables to their TVP setpoints
    setpoint_tvps = {
        "tank_1_L": model.tvp["tank_1_L_sp"],
        "tank_2_L": model.tvp["tank_2_L_sp"],
        "tank_3_L": model.tvp["tank_3_L_sp"],
        "tank_4_L": model.tvp["tank_4_L_sp"],
        "tank_4_conc_out": model.tvp["tank_4_conc_out_sp"],
        "tank_4_v_dot_out": model.tvp["tank_4_v_dot_out_sp"],
    }

    # Sum-of-squared tracking errors using TVP setpoints
    lterm = cost_function_tracking(
        cas.vcat(controlled_variables.values()),
        cas.vcat([setpoint_tvps[name] for name in controlled_variables]),
        cas.DM([cv_weights[name] for name in controlled_variables]),
    )

    # Terminal cost
    mterm = cas.DM(0)

    mpc.set_objective(mterm=mterm, lterm=lterm)

    # Set weights in control action cost term
    mpc.set_rterm(**mv_weights)

    bounds = {
        "inputs": {
            "tank_2_v_dot_in": v_dot_bounds,
            "tank_3_v_dot_in": v_dot_bounds,
            "mixer_v_dot_in_1": v_dot_bounds,
            "mixer_v_dot_in_2": v_dot_bounds,
            "tank_4_v_dot_out": v_dot_bounds,
        },
        "states": {
            "tank_1_L": tank_level_bounds,
            "tank_2_L": tank_level_bounds,
            "tank_3_L": tank_level_bounds,
            "tank_4_L": tank_level_bounds,
        },
    }

    # Apply lower and upper bounds to states
    for state_name, b in bounds["states"].items():
        mpc.bounds["lower", "_x", state_name] = b["lower"]
        mpc.bounds["upper", "_x", state_name] = b["upper"]

    # Apply lower and upper bounds to inputs
    for input_name, b in bounds["inputs"].items():
        mpc.bounds["lower", "_u", input_name] = b["lower"]
        mpc.bounds["upper", "_u", input_name] = b["upper"]

    # Apply lower and upper bounds to tank 4 output concentration
    mpc.set_nl_cons(
        "tank_4_conc_out_ub",
        -model.aux["tank_4_conc_out"],
        ub=-tank_4_conc_out_bounds["lower"],
    )
    mpc.set_nl_cons(
        "tank_4_conc_out_lb",
        model.aux["tank_4_conc_out"],
        ub=tank_4_conc_out_bounds["upper"],
    )

    return mpc


def generate_random_steps_beta(
    nT: int,
    step_length: int,
    step_separation: int = 0,
    y_base=0.0,
    y_min=0.0,
    y_max=1.0,
    a=10,
    b=10,
    y_range=(0.5, 3.5),
    rng=None,
    seed=10,
):
    if rng is None:
        rng = np.random.default_rng(seed)
    step_sequence = np.full(nT, y_base)
    n_steps = nT // (step_length + step_separation)
    step_levels = y_min + (y_max - y_min) * rng.beta(a, b, size=n_steps)
    for i in range(n_steps):
        start_idx = i * (step_length + step_separation)
        end_idx = start_idx + step_length
        step_sequence[start_idx:end_idx] = step_levels[i]

    return step_sequence


def create_setpoint_tvp_function(
    mpc_or_sim,
    setpoints,
    forecast_data=None,
    t_step=1.0,
    n_horizon=None,
):
    """Create TVP function for MPC setpoints.

    Parameters
    ----------
    mpc_or_sim : do_mpc.controller.MPC or do_mpc.simulator.Simulator
        The MPC controller or simulator object (NOT the model)
    setpoints : dict
        Dictionary mapping variable names to either:
        - float/int: constant setpoint value
        - str: key to lookup in forecast_data
    forecast_data : dict, optional
        Dictionary mapping forecast keys to arrays of values.
        Each array should contain discrete-time forecast values.
    t_step : float
        Time step for simulation/control (hours)
    n_horizon : int, optional
        Prediction horizon length. If None, inferred from mpc_or_sim

    Returns
    -------
    tvp_function : callable
        Function that returns TVP values at time t using ZOH interpolation
    """

    # Initialize TVP template from mpc or simulator object
    tvp_template = mpc_or_sim.get_tvp_template()

    # Get horizon length
    if n_horizon is None:
        if hasattr(mpc_or_sim, "settings"):
            n_horizon = mpc_or_sim.settings.n_horizon
        else:
            n_horizon = 1  # For simulator, just one step

    # Process setpoints configuration
    setpoint_config = {}
    for var_name, value in setpoints.items():
        tvp_name = f"{var_name}_sp"
        if isinstance(value, str):
            # String indicates a reference to forecast_data
            if forecast_data is None or value not in forecast_data:
                raise ValueError(
                    f"Setpoint for {var_name} references '{value}' but "
                    f"forecast_data is missing this key"
                )
            setpoint_config[tvp_name] = {
                "type": "forecast",
                "key": value,
                "data": np.array(forecast_data[value]),
            }
        else:
            # Numeric value indicates constant setpoint
            setpoint_config[tvp_name] = {
                "type": "constant",
                "value": float(value),
            }

    def tvp_function(t_now):
        """Return TVP values at time t_now using zero-order-hold.

        For forecast data, uses ZOH interpolation:
        - Finds the discrete time index: k = floor(t_now / t_step)
        - Returns the value at index k (holds until next sample)
        """
        for tvp_name, config in setpoint_config.items():
            # Fill in values for each step in the prediction horizon
            for k in range(n_horizon):
                if config["type"] == "constant":
                    tvp_template["_tvp", k, tvp_name, 0] = config["value"]
                else:  # forecast
                    # Calculate the time index for this prediction step
                    t_pred = t_now + k * t_step
                    idx = int(np.floor(t_pred / t_step))
                    data = config["data"]

                    # Clamp to valid range (hold last value if beyond forecast)
                    idx = np.clip(idx, 0, len(data) - 1)
                    tvp_template["_tvp", k, tvp_name, 0] = data[idx]

        return tvp_template

    return tvp_function


def create_simulator_tvp_function(simulator):
    """Create a simple TVP function for the simulator.

    For simulation, we don't need setpoint forecasts - the simulator
    just runs the plant model. Returns a function that provides
    zero/dummy values for all TVPs.

    Parameters
    ----------
    simulator : do_mpc.simulator.Simulator
        The simulator object
    """
    tvp_template = simulator.get_tvp_template()

    # Get list of TVP variable names
    tvp_names = list(simulator.model._tvp.keys())
    # Remove 'default' if it exists
    if "default" in tvp_names:
        tvp_names.remove("default")

    def tvp_function(t_now):
        # Simulator uses flat indexing: just variable name
        for tvp_name in tvp_names:
            tvp_template[tvp_name] = 0.0
        return tvp_template

    return tvp_function


def get_measurements(simulator, v0: np.ndarray = None):
    """Get output measurements from simulator"""

    # This code mirrors some of the code in Simulator.make_step
    if v0 is None:
        v0 = simulator.model._v(0)
    else:
        input_types = (np.ndarray, castools.DM, castools.structure3.DMStruct)
        assert isinstance(v0, input_types), (
            f"v0 is wrong input type. You have: {type(v0)}. Must be of type "
            f"{input_types}"
        )
        assert v0.shape == simulator.model._v.shape, (
            f"v0 has incorrect shape. You have: {v0.shape}, expected: "
            f"{simulator.model._v.shape}"
        )

    # Note: assumes no direct transmission so u0 here is a placeholder only
    u0 = cas.DM.zeros(simulator.model.n_u)

    x0 = simulator.x0
    z0 = simulator.z0
    x0_unscaled = x0 * simulator._x_scaling.cat
    z0_unscaled = z0 * simulator._z_scaling.cat
    t0 = simulator.t0
    tvp0 = simulator.tvp_fun(t0)
    p0 = simulator.p_fun(t0)

    # Call measurement function
    y0 = simulator.model._meas_fun(x0_unscaled, u0, z0_unscaled, tvp0, p0, v0)

    # Update data object
    simulator.data.update(_x=x0.cat)
    simulator.data.update(_u=u0)
    simulator.data.update(_z=z0_unscaled)
    simulator.data.update(_tvp=tvp0)
    simulator.data.update(_p=p0)
    simulator.data.update(_y=y0)

    return y0


def simulator_step(simulator, u0: np.ndarray = None, w0: np.ndarray = None):
    # This code mirrors some of the code in Simulator.make_step
    if w0 is None:
        w0 = simulator.model._w(0)
    else:
        input_types = (np.ndarray, castools.DM, castools.structure3.DMStruct)
        assert isinstance(w0, input_types), (
            f"w0 is wrong input type. You have: {type(w0)}. Must be of type "
            f"{input_types}"
        )
        assert w0.shape == simulator.model._w.shape, (
            f"w0 has incorrect shape. You have: {w0.shape}, expected: "
            f"{simulator.model._w.shape}"
        )

    tvp0 = simulator.tvp_fun(simulator._t0)
    p0 = simulator.p_fun(simulator._t0)
    t0 = simulator._t0
    x0 = simulator._x0

    simulator.sim_x_num["_x"] = x0.cat / simulator._x_scaling
    simulator.sim_p_num["_u"] = u0
    simulator.sim_p_num["_p"] = p0
    simulator.sim_p_num["_tvp"] = tvp0
    simulator.sim_p_num["_w"] = w0

    # Make sure that simulate() is computed with the curret tvp and p values
    aux0 = simulator.sim_aux_expression_fun(
        simulator.sim_x_num, simulator.sim_z_num, simulator.sim_p_num
    )

    x_next, z_next = simulator.simulate()

    x_next_unscaled = x_next * simulator._x_scaling.cat
    z_next_unscaled = z_next * simulator._z_scaling.cat

    # Update data object
    simulator.data.update(_aux=aux0)
    simulator.data.update(_time=t0)

    simulator._x0.master = x_next_unscaled
    simulator._z0.master = z_next_unscaled
    simulator._u0.master = u0
    simulator._t0 = simulator._t0 + simulator._settings.t_step

    simulator.flags["first_step"] = False

    return x_next, z_next


def run_simulation():
    seed = 42
    rng = np.random.default_rng(seed)

    # Tank dimensions (same for all tanks)
    D = 3.0  # diameter (m)
    tank_height = 10.0  # height (m)
    tank_level_bounds = {"lower": 1.0, "upper": tank_height}

    # Design basis
    feed_rate_nominal = 100.0  # m^3/h
    feed_conc_nominal = 0.5  # solids density (w/w)

    # Construct system model
    model = construct_4_tank_system_model(D=D)

    # Simulation time
    n_steps = 100
    t_step = 1.0
    time = np.arange(n_steps) * t_step

    # Create time-varying forecasts
    conc_setpoint_forecast = np.full(n_steps, feed_conc_nominal)
    # conc_setpoint_forecast[25:50] = feed_conc_nominal * 1.2
    # conc_setpoint_forecast[50:75] = feed_conc_nominal * 0.8

    # Example: Step changes in flow rate setpoint
    flow_setpoint_forecast = np.full(n_steps, feed_rate_nominal)
    # flow_setpoint_forecast[25:50] = feed_rate_nominal * 1.2
    # flow_setpoint_forecast[50:75] = feed_rate_nominal * 0.8

    # Define setpoints with mix of constant and time-varying
    setpoints = {
        "tank_1_L": tank_height * 0.8,  # Constant
        "tank_2_L": tank_height * 0.7,  # Constant
        "tank_3_L": tank_height * 0.6,  # Constant
        "tank_4_L": tank_height * 0.5,  # Constant
        "tank_4_conc_out": "conc_sp",  # Time-varying
        "tank_4_v_dot_out": "flow_sp",  # Time-varying
    }

    forecast_data = {
        "conc_sp": conc_setpoint_forecast,
        "flow_sp": flow_setpoint_forecast,
    }

    cv_weights = {
        "tank_1_L": 10.0,
        "tank_2_L": 10.0,
        "tank_3_L": 10.0,
        "tank_4_L": 10.0,
        "tank_4_conc_out": 1.0,
        "tank_4_v_dot_out": 1.0,
    }
    mv_weights = {
        "tank_2_v_dot_in": 0.1,
        "tank_3_v_dot_in": 0.1,
        "mixer_v_dot_in_1": 0.1,
        "mixer_v_dot_in_2": 0.1,
        "tank_4_v_dot_out": 0.1,
    }
    v_dot_bounds = {"lower": 0.0, "upper": 200.0}
    tank_level_bounds = {"lower": tank_height * 0.1, "upper": tank_height}
    tank_4_conc_out_bounds = {"lower": 0.0, "upper": 1.0}

    # Length of prediction horizon
    n_horizon = 50

    # Create MPC controller
    mpc = construct_mpc_controller(
        model,
        t_step,
        n_horizon,
        cv_weights,
        mv_weights,
        v_dot_bounds,
        tank_level_bounds,
        tank_4_conc_out_bounds,
    )

    # Create and set MPC TVP function
    mpc_tvp_fun = create_setpoint_tvp_function(
        mpc, setpoints, forecast_data, t_step=t_step, n_horizon=n_horizon
    )
    mpc.set_tvp_fun(mpc_tvp_fun)

    # Finalize MPC setup
    mpc.setup()

    # Create simulator
    simulator = do_mpc.simulator.Simulator(model)
    simulator.set_param(t_step=t_step)

    # Set TVP functions
    simulator_tvp_fun = create_simulator_tvp_function(simulator)
    simulator.set_tvp_fun(simulator_tvp_fun)

    simulator.setup()

    tank_level = sum(tank_level_bounds.values()) / 2
    tank_level = tank_height * 0.15

    x0_init = {
        "tank_1_L": tank_level,
        "tank_1_m": np.pi * D**2 / 4 * tank_level * feed_conc_nominal,
        "tank_2_L": tank_level,
        "tank_2_m": np.pi * D**2 / 4 * tank_level * feed_conc_nominal * 1.5,
        "tank_3_L": tank_level,
        "tank_3_m": np.pi * D**2 / 4 * tank_level * feed_conc_nominal * 0.5,
        "tank_4_L": tank_level,
        "tank_4_m": np.pi * D**2 / 4 * tank_level * feed_conc_nominal,
        "tank_1_v_dot_in": feed_rate_nominal,
        "tank_1_conc_in": feed_conc_nominal,
    }

    assert list(x0_init.keys()) == model.x.keys()
    # Convert to CasADi array
    x0_init = cas.DM(x0_init.values())

    # Set initial state for all components
    mpc.x0 = x0_init
    simulator.x0 = x0_init

    # Prepare lists of names of inputs and measured outputs
    mv_names = list(model.u.keys())
    measured_output_names = list(model.y.keys())
    # TODO: This is only necessary because model has a 'default' name
    mv_names.remove("default")
    measured_output_names.remove("default")
    output_names = [
        name.removesuffix("_meas") for name in measured_output_names
    ]

    # Generate input data

    # Disturbance inputs:
    step_length = 5
    tank_1_v_dot_in = generate_random_steps_beta(
        n_steps,
        step_length,
        y_base=feed_rate_nominal,
        y_min=0.9 * feed_rate_nominal,
        y_max=1.1 * feed_rate_nominal,
        seed=10,
    )
    tank_1_v_dot_in = np.full(tank_1_v_dot_in.shape, feed_rate_nominal)
    tank_1_conc_in = generate_random_steps_beta(
        n_steps,
        step_length,
        y_base=feed_conc_nominal,
        y_min=0.0 * feed_conc_nominal,
        y_max=2.0 * feed_conc_nominal,
        seed=10,
    )
    # tank_1_conc_in = np.full(tank_1_conc_in.shape, feed_conc_nominal)

    print("\nRunning closed-loop simulation...")
    print(f"Simulation steps: {n_steps}")
    print(f"Time step: {mpc.settings.t_step} hours")

    # Dictionary to store simulation results
    sim_results = defaultdict(list)
    time_values = []

    # Measurement noise std. dev.
    sigma_m = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    V = rng.normal(scale=sigma_m, size=(n_steps, model.n_y))

    # Process noise std. dev (TODO: Why is model.n_w == 0?)
    sigma_w = np.array([])
    W = rng.normal(scale=sigma_w, size=(n_steps, model.n_w))

    x0 = x0_init
    for k in range(n_steps):
        # Set disturbance inputs in simulator
        simulator.x0["tank_1_v_dot_in"] = tank_1_v_dot_in[k]
        simulator.x0["tank_1_conc_in"] = tank_1_conc_in[k]

        # Generate simulated measurements
        # Note: simulator has no direct transmission from u(k) to y(k).
        v0 = cas.DM(V[k, :])
        y0 = get_measurements(simulator, v0=np.zeros((model.n_y, 1)))
        y0_m = y0 + v0

        # For controller testing, use true state from simulator
        x0 = simulator.x0
        # x0 = estimator.update(y0_m)

        if k == 0:
            # Set initial guess for MPC
            mpc.x0 = x0
            mpc.u0 = cas.vertcat(
                feed_rate_nominal / 2,
                feed_rate_nominal / 2,
                feed_rate_nominal / 2,
                feed_rate_nominal / 2,
                feed_rate_nominal,
            )
            mpc.set_initial_guess()

        # Compute control action
        mpc.make_step(x0)

        # Save current inputs, states and outputs
        u0 = {name: float(mpc.u0[name]) for name in mv_names}
        x0 = {name: float(value) for name, value in dict(mpc.x0).items()}
        y0 = {name: float(y0[i]) for i, name in enumerate(output_names)}
        y0_m = {
            name: float(y0_m[i])
            for i, name in enumerate(measured_output_names)
        }

        # Extract current setpoints from MPC's TVP function
        t_now = float(simulator.t0)
        tvp_current = mpc.tvp_fun(t_now)

        # Get all TVP setpoint names dynamically
        tvp_names = [key for key in model._tvp.keys() if key != "default"]

        # Extract setpoints at current time (k=0 is current time in prediction horizon)
        y0_sp = {
            tvp_name: float(tvp_current["_tvp", 0, tvp_name, 0])
            for tvp_name in tvp_names
        }

        time_values.append(float(simulator.t0))
        sim_results["inputs"].append(u0)
        sim_results["states"].append(x0)
        sim_results["true_outputs"].append(y0)
        sim_results["measured_outputs"].append(y0_m)
        sim_results["time_varying_params"].append(y0_sp)

        # Simulate system
        w0 = cas.DM(W[k, :])
        x_next, z_next = simulator_step(simulator, u0=mpc.u0, w0=w0)

    print("Simulation complete!")

    # Compile results into Pandas DataFrame
    sim_results = pd.concat(
        {
            name: pd.DataFrame(data, index=pd.Index(time_values, name="t"))
            for name, data in sim_results.items()
        },
        axis=1,
    )

    return sim_results


if __name__ == "__main__":
    sim_name = Path(__file__).stem
    results_dir = Path("results") / sim_name
    plot_dir = results_dir / "plots"
    results_dir.mkdir(exist_ok=True, parents=True)
    plot_dir.mkdir(exist_ok=True)

    sim_results = run_simulation()

    filename = "sim_results.csv"
    sim_results.to_csv(results_dir / filename)
