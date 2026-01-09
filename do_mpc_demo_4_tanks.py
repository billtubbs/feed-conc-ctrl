from pathlib import Path
from collections import defaultdict

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import casadi as cas
import casadi.tools as castools
from feed_conc_ctrl.plot_utils import make_tsplots

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
    dm_dt = v_dot_in * conc_in - v_dot_out * m / (L * A)
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


def construct_4_tank_system_model(D=3.0):
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

    model.setup()

    return model


def cost_function_tracking(controlled_variables, setpoints, weights):
    pred_errors = setpoints - controlled_variables
    cost = cas.sum1(weights * cas.sumsqr(pred_errors))
    return cost


def construct_mpc_controller(model):
    # Create MPC controller
    mpc = do_mpc.controller.MPC(model)

    mpc_params = {
        "n_horizon": 50,  # Prediction horizon (hours)
        "t_step": 1.0,  # Time step (hours)
        "n_robust": 0,  # No robust horizon for now
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
    setpoints = {
        "tank_1_L": 2.0,
        "tank_2_L": 2.0,
        "tank_3_L": 2.0,
        "tank_4_L": 2.0,
        "tank_4_conc_out": 0.5,
        "tank_4_v_dot_out": 1.0,
    }
    cv_weights = {
        "tank_1_L": 0.1,
        "tank_2_L": 0.1,
        "tank_3_L": 0.1,
        "tank_4_L": 0.1,
        "tank_4_conc_out": 10.0,
        "tank_4_v_dot_out": 5.0,
    }

    # Sum-of-squared tracking errors
    lterm = cost_function_tracking(
        cas.vcat(controlled_variables.values()),
        cas.DM([setpoints[name] for name in controlled_variables]),
        cas.DM([cv_weights[name] for name in controlled_variables]),
    )
    mterm = cas.DM(0)  # Terminal cost
    mpc.set_objective(mterm=mterm, lterm=lterm)

    # Set weights in control action cost term
    mv_weights = {
        "tank_2_v_dot_in": 1.0,
        "tank_3_v_dot_in": 1.0,
        "mixer_v_dot_in_1": 1.0,
        "mixer_v_dot_in_2": 1.0,
        "tank_4_v_dot_out": 5.0,
    }
    mpc.set_rterm(**mv_weights)

    # Input constraints
    mpc.bounds["lower", "_u", "tank_2_v_dot_in"] = 0.0
    mpc.bounds["upper", "_u", "tank_2_v_dot_in"] = 10.0
    mpc.bounds["lower", "_u", "tank_3_v_dot_in"] = 0.0
    mpc.bounds["upper", "_u", "tank_3_v_dot_in"] = 10.0
    mpc.bounds["lower", "_u", "mixer_v_dot_in_1"] = 0.0
    mpc.bounds["upper", "_u", "mixer_v_dot_in_1"] = 10.0
    mpc.bounds["lower", "_u", "mixer_v_dot_in_2"] = 0.0
    mpc.bounds["upper", "_u", "mixer_v_dot_in_2"] = 10.0
    mpc.bounds["lower", "_u", "tank_4_v_dot_out"] = 0.0
    mpc.bounds["upper", "_u", "tank_4_v_dot_out"] = 2.0

    # State constraints
    mpc.bounds["lower", "_x", "tank_1_L"] = 0.1
    mpc.bounds["upper", "_x", "tank_1_L"] = 4.0
    mpc.bounds["lower", "_x", "tank_2_L"] = 0.1
    mpc.bounds["upper", "_x", "tank_2_L"] = 4.0
    mpc.bounds["lower", "_x", "tank_3_L"] = 0.1
    mpc.bounds["upper", "_x", "tank_3_L"] = 4.0
    mpc.bounds["lower", "_x", "tank_4_L"] = 0.1
    mpc.bounds["upper", "_x", "tank_4_L"] = 4.0

    # Output constraints
    mpc.set_nl_cons("tank_4_conc_out_lb", model.aux["tank_4_conc_out"], ub=3.0)
    mpc.set_nl_cons(
        "tank_4_conc_out_ub", -model.aux["tank_4_conc_out"], ub=-0.0
    )

    mpc.setup()

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


def compile_sim_results(simulator, y0_init):
    """Put all simulation results into a Pandas DataFrame"""

    # Simulator.make_step stores y(k+1) in the '_y' field, not y(k).
    # Therefore shift the values forward one time step and insert
    # the initial condition at t = 0.
    data_y = simulator.data["_y"].copy()
    data_y = np.roll(data_y, 1, axis=0)
    data_y[0, :] = np.array(y0_init).flatten()
    sim_results = pd.concat(
        {
            "time": pd.DataFrame(simulator.data["_time"]),
            "manipulated_inputs": pd.DataFrame(
                simulator.data["_u"],
                columns=pd.Index(simulator.model._u.keys()).drop("default"),
            ),
            "states": pd.DataFrame(
                simulator.data["_x"],
                columns=pd.Index(simulator.model._x.keys()),
            ),
            "outputs": pd.DataFrame(
                data_y,
                columns=pd.Index(simulator.model._y.keys()).drop("default"),
            ),
        },
        axis=1,
    )
    return sim_results


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
    D = 4  # diameter (m)
    # Basis of design
    # tank_height = 9  # height (m)
    # feed_rate_nominal = 50  # t/h
    # feed_density_nominal = 0.5  # solids density (w/w)

    # Construct system model
    model = construct_4_tank_system_model(D=D)

    # Create MPC controller
    mpc = construct_mpc_controller(model)

    # Create simulator
    simulator = do_mpc.simulator.Simulator(model)

    t_step = mpc.settings.t_step
    simulator.set_param(t_step=t_step)
    simulator.setup()

    x0_init = {
        "tank_1_L": 1.0,
        "tank_1_m": np.pi * (D / 2) ** 2 * 2.0 * 1.0,
        "tank_2_L": 1.0,
        "tank_2_m": np.pi * (D / 2) ** 2 * 2.0 * 1.0,
        "tank_3_L": 1.0,
        "tank_3_m": np.pi * (D / 2) ** 2 * 2.0 * 1.0,
        "tank_4_L": 1.0,
        "tank_4_m": np.pi * (D / 2) ** 2 * 2.0 * 1.0,
        "tank_1_v_dot_in": 4.0,
        "tank_1_conc_in": 2.0,
    }

    assert list(x0_init.keys()) == model.x.keys()
    # Convert to CasADi array
    x0_init = cas.DM(x0_init.values())

    # Set initial state for all components
    mpc.x0 = x0_init
    simulator.x0 = x0_init

    # Set initial guess for MPC
    mpc.set_initial_guess()

    # Generate input data
    n_steps = 100

    # Disturbance inputs:
    step_length = 5
    tank_1_v_dot_in = generate_random_steps_beta(
        n_steps,
        step_length,
        y_base=0.0,
        y_min=0.75,
        y_max=1.25,
        seed=10,
    )
    tank_1_conc_in = generate_random_steps_beta(
        n_steps,
        step_length,
        y_base=0.0,
        y_min=0.2,
        y_max=0.8,
        seed=10,
    )

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
        # Generate simulated measurements
        # Note: simulator has no direct transmission from u(k) to y(k).
        v0 = cas.DM(V[k, :])
        y0_m = get_measurements(simulator, v0=v0)

        # For controller testing, use true state from simulator
        x0 = simulator.x0
        x0 = cas.vcat(x0[x0.keys()])
        # x0 = estimator.update(y0_m)

        # Compute control action
        u0 = mpc.make_step(x0)

        # Set disturbance inputs in simulator
        simulator.x0["tank_1_v_dot_in"] = tank_1_v_dot_in[k]
        simulator.x0["tank_1_conc_in"] = tank_1_conc_in[k]

        # Save current inputs, states and outputs
        time_values.append(simulator.t0)
        sim_results["Y_m"].append(np.array(y0_m).reshape(-1))
        sim_results["X"].append(np.array(x0).reshape(-1))
        sim_results["U"].append(np.array(u0).reshape(-1))

        # Simulate system
        w0 = cas.DM(W[k, :])
        x_next, z_next = simulator_step(simulator, u0=u0, w0=w0)

    print("Simulation complete!")

    # Compile results into Pandas DataFrame
    for name, data in sim_results.items():
        sim_results[name] = pd.DataFrame(
            np.stack(data), index=pd.Index(time_values, name="t")
        )
    sim_results = pd.concat(sim_results, axis=1)

    return sim_results


if __name__ == "__main__":
    sim_name = Path(__file__).stem
    results_dir = Path("results") / sim_name
    plot_dir = results_dir / "plots"
    results_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)

    sim_results = run_simulation()

    filename = "sim_results.csv"
    sim_results.to_csv(results_dir / filename)
