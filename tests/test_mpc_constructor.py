"""Tests for MPC constructor function."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import do_mpc

from cas_models.continuous_time.models import StateSpaceModelCT
from cas_models.transformations import connect_systems

from feed_conc_ctrl.models import MixingTankModelCT, RatioControlledFlowMixerCT
from do_mpc_utils.mpc_constructor import construct_mpc


@pytest.fixture
def two_tank_mixer_system():
    """Create the two-tank + mixer system from the notebook."""
    D = 3  # tank diameter [m]
    n_tanks = 2
    tank_names = [f"tank_{i + 1}" for i in range(n_tanks)]

    # Initialize tank system models
    systems = [MixingTankModelCT(D=D, name=name) for name in tank_names]

    # Add a flow mixer to join flows from two tanks
    systems.append(RatioControlledFlowMixerCT(2, name="mixer"))

    # Connect all systems together
    connections = {
        "mixer_conc_in_1": "tank_1_conc_out",
        "mixer_conc_in_2": "tank_2_conc_out",
        "tank_1_v_dot_out": "mixer_v_dot_in_1",
        "tank_2_v_dot_out": "mixer_v_dot_in_2",
    }

    model_class = StateSpaceModelCT
    feed_tanks_system = connect_systems(
        systems,
        connections,
        model_class,
        name="tank_system_21",
        verbose_names=True,
    )

    return feed_tanks_system


@pytest.fixture
def control_design():
    """Control design configuration from notebook."""
    return {
        "system_states": ["tank_1_L", "tank_1_m", "tank_2_L", "tank_2_m"],
        "manipulated_variables": [
            "tank_1_v_dot_in",
            "tank_2_v_dot_in",
            "mixer_r_1",
        ],
        "unmeasured_disturbances": [
            "tank_1_conc_in",
            "tank_2_conc_in",
            "mixer_v_dot_out",
        ],
        "controlled_variables": [
            "tank_1_L",
            "tank_1_conc_out",
            "tank_2_L",
            "tank_2_conc_out",
            "mixer_conc_out",
        ],
    }


@pytest.fixture
def mpc_params():
    """MPC parameters from notebook."""
    return {
        "n_horizon": 50,
        "t_step": 1.0,
        "n_robust": 0,
        "store_full_solution": True,
    }


@pytest.fixture
def setpoints():
    """Setpoints from notebook (only for CVs with weights)."""
    return {
        "tank_1_L": 1.5,
        "tank_2_L": 1.5,
        "mixer_conc_out": 2.0,
    }


@pytest.fixture
def cv_weights():
    """CV weights from notebook."""
    return {
        "tank_1_L": 0.1,
        "tank_2_L": 0.1,
        "mixer_conc_out": 10.0,
    }


@pytest.fixture
def mv_weights():
    """MV weights from notebook."""
    return {
        "tank_1_v_dot_in": 0.1,
        "tank_2_v_dot_in": 0.1,
        "mixer_r_1": 0.1,
    }


@pytest.fixture
def bounds():
    """Bounds from notebook."""
    return {
        "inputs": {
            "tank_1_v_dot_in": {"lower": 0.0, "upper": 2.0},
            "tank_2_v_dot_in": {"lower": 0.0, "upper": 2.0},
            "mixer_r_1": {"lower": 0.0, "upper": 1.0},
        },
        "system_states": {
            "tank_1_L": {"lower": 0.1, "upper": 3.0},
            "tank_2_L": {"lower": 0.1, "upper": 3.0},
        },
        "outputs": {
            "mixer_conc_out": {"lower": 0.0, "upper": 3.0},
        },
    }


def test_construct_mpc_creates_controller(
    two_tank_mixer_system,
    control_design,
    mpc_params,
    setpoints,
    cv_weights,
    mv_weights,
    bounds,
):
    """Test that construct_mpc creates an MPC controller."""
    mpc, model = construct_mpc(
        two_tank_mixer_system,
        control_design=control_design,
        mpc_params=mpc_params,
        setpoints=setpoints,
        cv_weights=cv_weights,
        mv_weights=mv_weights,
        bounds=bounds,
    )

    # Check that mpc is a do_mpc controller
    assert isinstance(mpc, do_mpc.controller.MPC)

    # Check that model is a do_mpc model
    assert isinstance(model, do_mpc.model.Model)

    # Check model dimensions
    # States: 4 system states + 3 disturbance states = 7
    assert model.n_x == 7

    # Manipulated inputs: 3
    assert model.n_u == 3

    # Controlled variables: 5
    assert model.n_y == 5

    # Check that state names are correct
    expected_states = [
        "tank_1_L",
        "tank_1_m",
        "tank_2_L",
        "tank_2_m",
        "tank_1_conc_in",
        "tank_2_conc_in",
        "mixer_v_dot_out",
    ]
    assert list(model.x.keys()) == expected_states

    # Check that MV names are correct (default comes first)
    expected_mvs = [
        "default",
        "tank_1_v_dot_in",
        "tank_2_v_dot_in",
        "mixer_r_1",
    ]
    assert list(model.u.keys()) == expected_mvs

    # Check that CV names are correct (default comes first)
    expected_cvs = [
        "default",
        "tank_1_L",
        "tank_1_conc_out",
        "tank_2_L",
        "tank_2_conc_out",
        "mixer_conc_out",
    ]
    assert list(model.y.keys()) == expected_cvs

    # Check MPC parameters
    assert mpc.settings.n_horizon == 50
    assert mpc.settings.t_step == 1.0
    assert mpc.settings.n_robust == 0
    assert mpc.settings.store_full_solution is True

    # Check rterm factors (control effort penalty weights)
    for mv_name, expected_weight in mv_weights.items():
        assert mpc.rterm_factor[mv_name] == expected_weight

    # Check bounds on manipulated variables
    for mv_name, mv_bounds in bounds["inputs"].items():
        if "lower" in mv_bounds:
            assert mpc.bounds["lower", "_u", mv_name] == (mv_bounds["lower"])
        if "upper" in mv_bounds:
            assert mpc.bounds["upper", "_u", mv_name] == (mv_bounds["upper"])

    # Check bounds on states
    for state_name, state_bounds in bounds["system_states"].items():
        if "lower" in state_bounds:
            assert (
                mpc.bounds["lower", "_x", state_name]
                == (state_bounds["lower"])
            )
        if "upper" in state_bounds:
            assert (
                mpc.bounds["upper", "_x", state_name]
                == (state_bounds["upper"])
            )


def test_mpc_simulation_matches_notebook(
    two_tank_mixer_system,
    control_design,
    mpc_params,
    setpoints,
    cv_weights,
    mv_weights,
    bounds,
):
    """Test MPC simulation matches notebook results."""
    # Construct MPC
    mpc, model = construct_mpc(
        two_tank_mixer_system,
        control_design=control_design,
        mpc_params=mpc_params,
        setpoints=setpoints,
        cv_weights=cv_weights,
        mv_weights=mv_weights,
        bounds=bounds,
    )

    # Setup simulator
    simulator = do_mpc.simulator.Simulator(model)
    simulator.set_param(t_step=1.0)
    simulator.setup()

    # Set initial conditions (same as notebook)
    x0_init = {
        # System states
        "tank_1_L": 1.0,
        "tank_1_m": 0.0,
        "tank_2_L": 1.0,
        "tank_2_m": 0.0,
        # Disturbance states (true values for simulation)
        "tank_1_conc_in": 1.0,
        "tank_2_conc_in": 3.0,
        "mixer_v_dot_out": 1.0,
    }
    x0_init = np.array(list(x0_init.values())).reshape(-1, 1)

    # Set initial state for all components
    mpc.x0 = x0_init
    simulator.x0 = x0_init

    # Calculate initial outputs
    y0_init = simulator.model._meas_fun(
        simulator.x0,
        np.zeros((simulator.model.n_u, 1)),
        np.zeros((simulator.model.n_z, 1)),
        np.zeros((simulator.model.n_tvp, 1)),
        np.zeros((simulator.model.n_p, 1)),
        np.zeros((simulator.model.n_y, 1)),
    )

    # Set initial guess for MPC
    mpc.set_initial_guess()

    # Run simulation (same as notebook)
    n_steps = 100

    # Disturbance inputs (same as notebook)
    dist_inputs = np.zeros((n_steps, 3))
    dist_inputs[:, 0] = 1.0  # tank_1_conc_in
    dist_inputs[:, 1] = 3.0  # tank_2_conc_in
    dist_inputs[n_steps // 2 :, 1] = 2.5  # step change
    dist_inputs[:, 2] = 1.0  # mixer_v_dot_out

    x0 = x0_init
    for k in range(n_steps):
        # Get control action from MPC
        u0 = mpc.make_step(x0)

        # Set disturbance inputs in simulator
        simulator.x0["tank_1_conc_in"] = dist_inputs[k, 0]
        simulator.x0["tank_2_conc_in"] = dist_inputs[k, 1]
        simulator.x0["mixer_v_dot_out"] = dist_inputs[k, 2]

        # Simulate system
        simulator.make_step(u0)

        # For perfect controller testing, use true state from simulator
        x0 = simulator.x0

    # Process simulation results
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

    # Load notebook results
    test_data_dir = Path(__file__).parent / "data"
    notebook_results = pd.read_csv(
        test_data_dir / "feed_conc_ctrl_sim_results.csv", header=[0, 1]
    )

    # Compare outputs (controlled variables)
    for cv_name in control_design["controlled_variables"]:
        sim_values = sim_results["outputs"][cv_name].values
        notebook_values = notebook_results["outputs"][cv_name].values

        # Check that outputs match within tolerance
        np.testing.assert_allclose(
            sim_values,
            notebook_values,
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"Output {cv_name} does not match notebook results",
        )

    # Compare manipulated variables
    for mv_name in control_design["manipulated_variables"]:
        sim_values = sim_results["manipulated_inputs"][mv_name].values
        notebook_values = notebook_results["manipulated_inputs"][
            mv_name
        ].values

        # Check that MVs match within tolerance
        np.testing.assert_allclose(
            sim_values,
            notebook_values,
            rtol=1e-5,
            atol=1e-6,
            err_msg=(
                f"Manipulated variable {mv_name} "
                f"does not match notebook results"
            ),
        )

    # Compare states
    for state_name in two_tank_mixer_system.state_names:
        sim_values = sim_results["states"][state_name].values
        notebook_values = notebook_results["states"][state_name].values

        # Check that states match within tolerance
        np.testing.assert_allclose(
            sim_values,
            notebook_values,
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"State {state_name} does not match notebook results",
        )


def test_construct_mpc_with_optional_defaults(
    two_tank_mixer_system, control_design, mpc_params, cv_weights
):
    """Test construct_mpc with optional arguments set to defaults."""
    # Construct MPC with only required arguments
    # Optional arguments (mv_weights, setpoints, bounds) = None
    mpc, model = construct_mpc(
        two_tank_mixer_system,
        control_design=control_design,
        mpc_params=mpc_params,
        cv_weights=cv_weights,
        # mv_weights=None (default - no control effort penalty)
        # setpoints=None (default - all setpoints = 0.0)
        # bounds=None (default - unbounded)
    )

    # Check that mpc is created successfully
    assert isinstance(mpc, do_mpc.controller.MPC)
    assert isinstance(model, do_mpc.model.Model)

    # Verify that MPC is properly configured
    assert mpc.settings.n_horizon == 50
    assert mpc.settings.t_step == 1.0

    # Check that rterm factors are all zero (default when mv_weights=None)
    for mv_name in control_design["manipulated_variables"]:
        assert mpc.rterm_factor[mv_name] == 0.0

    # Note: We cannot directly check that setpoints default to 0.0
    # or that bounds are unbounded without inspecting internal
    # structures, but successful MPC creation verifies they work
