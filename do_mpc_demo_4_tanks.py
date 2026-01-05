from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import casadi as cas
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


def cost_function(controlled_variables, setpoints, weights):
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
        "tank_4_v_dot_out": 5.0,
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
    lterm = cost_function(
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


results_dir = Path("results")
plot_dir = results_dir / "plots"
results_dir.mkdir(exist_ok=True)
plot_dir.mkdir(exist_ok=True)


# Construct system model
model = construct_4_tank_system_model()

# Create MPC controller
mpc = construct_mpc_controller(model)
