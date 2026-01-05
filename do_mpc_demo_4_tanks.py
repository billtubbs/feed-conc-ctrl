from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import casadi as cas
import do_mpc

from feed_conc_ctrl.plot_utils import make_tsplots

# Suppress do_mpc optional feature warnings
warnings.filterwarnings("ignore", category=UserWarning, module="do_mpc.sysid")
warnings.filterwarnings("ignore", category=UserWarning, module="do_mpc.opcua")
warnings.filterwarnings(
    "ignore", category=UserWarning, module="do_mpc.approximateMPC"
)


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

    # Measured output variables
    #  1. Tank 1 level
    #  2. Tank 1 outflow concentration
    #  3. Tank 2 level
    #  4. Tank 2 outflow concentration
    #  5. Tank 3 level
    #  6. Tank 3 outflow concentration
    #  7. Tank 4 level
    #  8. Tank 4 outflow concentration

    tank_1_conc_out = calc_mixing_tank_conc(tank_1_L, tank_1_m, tank_1_D)
    tank_2_conc_out = calc_mixing_tank_conc(tank_2_L, tank_2_m, tank_2_D)
    tank_3_conc_out = calc_mixing_tank_conc(tank_3_L, tank_3_m, tank_3_D)
    tank_4_conc_out = calc_mixing_tank_conc(tank_4_L, tank_4_m, tank_4_D)

    tank_1_L_meas = model.set_meas(meas_name="tank_1_L", expr=tank_1_L)
    tank_1_conc_out_meas = model.set_meas(
        meas_name="tank_1_conc_out", expr=tank_1_conc_out
    )
    tank_2_L_meas = model.set_meas(meas_name="tank_2_L", expr=tank_2_L)
    tank_2_conc_out_meas = model.set_meas(
        meas_name="tank_2_conc_out", expr=tank_2_conc_out
    )
    tank_3_L_meas = model.set_meas(meas_name="tank_3_L", expr=tank_3_L)
    tank_3_conc_out_meas = model.set_meas(
        meas_name="tank_3_conc_out", expr=tank_3_conc_out
    )
    tank_4_L_meas = model.set_meas(meas_name="tank_4_L", expr=tank_4_L)
    tank_4_conc_out_meas = model.set_meas(
        meas_name="tank_4_conc_out", expr=tank_4_conc_out
    )

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


results_dir = Path("results")
plot_dir = results_dir / "plots"
results_dir.mkdir(exist_ok=True)
plot_dir.mkdir(exist_ok=True)


# Construct system model
model = construct_4_tank_system_model()
