"""Unit tests for FlowMixerCT and FlowMixerDT classes"""

import pytest
import numpy as np
import casadi as cas
from feed_conc_ctrl.models import FlowMixerCT, FlowMixerDT, RatioControlledFlowMixerCT
from cas_models.continuous_time.models import StateSpaceModelCT
from cas_models.discrete_time.models import StateSpaceModelDT


class TestFlowMixerCT:
    """Test suite for FlowMixerCT class"""

    def test_initialization_default(self):
        """Test that FlowMixerCT initializes correctly with default parameters"""
        n_in = 2
        model = FlowMixerCT(n_in)

        assert isinstance(model, StateSpaceModelCT)

        # Check model dimensions (stateless system)
        assert model.n == 0, "Should have 0 states (stateless system)"
        assert model.nu == 4, "Should have 4 inputs (2 inlets x 2 properties)"
        assert model.ny == 2, "Should have 2 outputs (flow and concentration)"

        # Check default name
        assert model.name == "FlowMixerModel"

        # Check input/output names
        assert model.input_names == [
            "v_dot_in_1",
            "conc_in_1",
            "v_dot_in_2",
            "conc_in_2",
        ]
        assert model.output_names == ["v_dot_out", "conc_out"]
        assert model.state_names == []

        # Check functions exist
        assert hasattr(model, "f")
        assert hasattr(model, "h")

    def test_initialization_with_n_in(self):
        """Test that FlowMixerCT initializes correctly with custom number of inlets"""
        n_in = 3
        model = FlowMixerCT(n_in)

        # Check model dimensions
        assert model.n == 0
        assert model.nu == 2 * n_in  # 6 inputs
        assert model.ny == 2

        # Check input names
        expected_inputs = [
            "v_dot_in_1",
            "conc_in_1",
            "v_dot_in_2",
            "conc_in_2",
            "v_dot_in_3",
            "conc_in_3",
        ]
        assert model.input_names == expected_inputs

    def test_initialization_with_name(self):
        """Test that FlowMixerCT initializes with custom name"""
        custom_name = "MyMixer"
        n_in = 2
        model = FlowMixerCT(n_in, name=custom_name)

        assert model.name == custom_name

    def test_initialization_validation(self):
        """Test initialization validation - n_in must be at least 2"""
        with pytest.raises(ValueError, match="n_in must be at least 2"):
            n_in = 1
            FlowMixerCT(n_in)

        with pytest.raises(ValueError, match="n_in must be at least 2"):
            n_in = 0
            FlowMixerCT(n_in)

    def test_ode_function_stateless(self):
        """Test the ODE function for stateless system (should return empty)"""
        n_in = 2
        model = FlowMixerCT(n_in)

        t = 0.0
        x = cas.vertcat()  # Empty state vector
        u = cas.vertcat(1.0, 0.5, 2.0, 0.75)  # Arbitrary inputs

        # Compute derivatives (should be empty)
        rhs = model.f(t, x, u)
        rhs_array = np.array(rhs).flatten()

        assert rhs_array.shape == (0,), (
            "Stateless system should have no derivatives"
        )

    def test_output_function_two_inlets(self):
        """Test the output function with 2 inlets"""
        n_in = 2
        model = FlowMixerCT(n_in)

        # Test inputs
        v_dot_1 = 1.0
        conc_1 = 0.5
        v_dot_2 = 2.0
        conc_2 = 0.75

        t = 0.0
        x = cas.vertcat()  # Empty state
        u = cas.vertcat(v_dot_1, conc_1, v_dot_2, conc_2)

        # Compute outputs
        y = model.h(t, x, u)
        y_array = np.array(y).flatten()

        # Expected outputs
        v_dot_out_expected = v_dot_1 + v_dot_2  # 3.0
        conc_out_expected = (
            v_dot_1 * conc_1 + v_dot_2 * conc_2
        ) / v_dot_out_expected
        # = (1.0*0.5 + 2.0*0.75) / 3.0 = 2.0 / 3.0 ≈ 0.6667

        assert y_array.shape == (2,)
        assert np.allclose(y_array[0], v_dot_out_expected, rtol=1e-10)
        assert np.allclose(y_array[1], conc_out_expected, rtol=1e-10)

    def test_output_function_three_inlets(self):
        """Test the output function with 3 inlets"""
        n_in = 3
        model = FlowMixerCT(n_in)

        # Test inputs
        v_dot_1 = 1.0
        conc_1 = 0.5
        v_dot_2 = 2.0
        conc_2 = 0.75
        v_dot_3 = 1.5
        conc_3 = 0.6

        t = 0.0
        x = cas.vertcat()
        u = cas.vertcat(v_dot_1, conc_1, v_dot_2, conc_2, v_dot_3, conc_3)

        # Compute outputs
        y = model.h(t, x, u)
        y_array = np.array(y).flatten()

        # Expected outputs
        v_dot_out_expected = v_dot_1 + v_dot_2 + v_dot_3  # 4.5
        mass_flow_total = (
            v_dot_1 * conc_1 + v_dot_2 * conc_2 + v_dot_3 * conc_3
        )
        # = 1.0*0.5 + 2.0*0.75 + 1.5*0.6 = 0.5 + 1.5 + 0.9 = 2.9
        conc_out_expected = mass_flow_total / v_dot_out_expected
        # = 2.9 / 4.5 ≈ 0.6444

        assert np.allclose(y_array[0], v_dot_out_expected, rtol=1e-10)
        assert np.allclose(y_array[1], conc_out_expected, rtol=1e-10)

    def test_output_function_equal_concentrations(self):
        """Test that equal inlet concentrations give same outlet concentration"""
        n_in = 2
        model = FlowMixerCT(n_in)

        conc = 0.75  # Same concentration for both inlets
        v_dot_1 = 1.0
        v_dot_2 = 3.0

        t = 0.0
        x = cas.vertcat()
        u = cas.vertcat(v_dot_1, conc, v_dot_2, conc)

        # Compute outputs
        y = model.h(t, x, u)
        y_array = np.array(y).flatten()

        # When all concentrations are equal, output should be same concentration
        assert np.allclose(y_array[1], conc, rtol=1e-10)

    def test_output_function_zero_flow(self):
        """Test mixer with zero flow in one inlet"""
        n_in = 2
        model = FlowMixerCT(n_in)

        v_dot_1 = 0.0  # No flow from inlet 1
        conc_1 = 0.5
        v_dot_2 = 2.0
        conc_2 = 0.75

        t = 0.0
        x = cas.vertcat()
        u = cas.vertcat(v_dot_1, conc_1, v_dot_2, conc_2)

        # Compute outputs
        y = model.h(t, x, u)
        y_array = np.array(y).flatten()

        # With zero flow from inlet 1, output should match inlet 2
        assert np.allclose(y_array[0], v_dot_2, rtol=1e-10)
        assert np.allclose(y_array[1], conc_2, rtol=1e-10)


class TestFlowMixerDT:
    """Test suite for FlowMixerDT class"""

    def test_initialization_default(self):
        """Test that FlowMixerDT initializes correctly"""
        dt = 1.0
        n_in = 2
        model = FlowMixerDT(dt, n_in)

        assert isinstance(model, StateSpaceModelDT)

        # Check model dimensions (stateless system)
        assert model.n == 0, "Should have 0 states (stateless system)"
        assert model.nu == 4, "Should have 4 inputs (2 inlets x 2 properties)"
        assert model.ny == 2, "Should have 2 outputs (flow and concentration)"
        assert model.dt == dt

        # Check input/output names
        assert model.input_names == [
            "v_dot_in_1",
            "conc_in_1",
            "v_dot_in_2",
            "conc_in_2",
        ]
        assert model.output_names == ["v_dot_out", "conc_out"]
        assert model.state_names == []

        # Check functions exist
        assert hasattr(model, "F")  # Discrete-time state transition
        assert hasattr(model, "H")  # Discrete-time output

    def test_initialization_with_name(self):
        """Test that FlowMixerDT initializes with custom name"""
        dt = 1.0
        n_in = 2
        custom_name = "MyMixer"
        model = FlowMixerDT(dt, n_in, name=custom_name)

        assert model.name == custom_name

    def test_initialization_validation(self):
        """Test initialization validation - n_in must be at least 2"""
        dt = 1.0

        with pytest.raises(ValueError, match="n_in must be at least 2"):
            n_in = 1
            FlowMixerDT(dt, n_in)

        with pytest.raises(ValueError, match="n_in must be at least 2"):
            n_in = 0
            FlowMixerDT(dt, n_in)

    def test_state_transition_stateless(self):
        """Test state transition for stateless system (should do nothing)"""
        dt = 1.0
        n_in = 2
        model = FlowMixerDT(dt, n_in)

        t = 0.0
        xk = []  # Empty state
        uk = [1.0, 0.5, 2.0, 0.75]  # Arbitrary inputs

        # Compute next state (should be empty)
        xkp1 = model.F(t, xk, uk)
        xkp1_array = np.array(xkp1).flatten()

        assert xkp1_array.shape == (0,), (
            "Stateless system state should remain empty"
        )

    def test_output_function_two_inlets(self):
        """Test the output function with 2 inlets"""
        dt = 1.0
        n_in = 2
        model = FlowMixerDT(dt, n_in)

        # Test inputs
        v_dot_1 = 1.0
        conc_1 = 0.5
        v_dot_2 = 2.0
        conc_2 = 0.75

        t = 0.0
        xk = []  # Empty state
        uk = [v_dot_1, conc_1, v_dot_2, conc_2]

        # Compute outputs
        yk = model.H(t, xk, uk)
        yk_array = np.array(yk).flatten()

        # Expected outputs
        v_dot_out_expected = v_dot_1 + v_dot_2  # 3.0
        conc_out_expected = (
            v_dot_1 * conc_1 + v_dot_2 * conc_2
        ) / v_dot_out_expected

        assert yk_array.shape == (2,)
        assert np.allclose(yk_array[0], v_dot_out_expected, rtol=1e-10)
        assert np.allclose(yk_array[1], conc_out_expected, rtol=1e-10)

    def test_output_function_three_inlets(self):
        """Test the output function with 3 inlets"""
        dt = 1.0
        n_in = 3
        model = FlowMixerDT(dt, n_in)

        # Test inputs
        v_dot_1 = 1.0
        conc_1 = 0.5
        v_dot_2 = 2.0
        conc_2 = 0.75
        v_dot_3 = 1.5
        conc_3 = 0.6

        t = 0.0
        xk = []
        uk = [v_dot_1, conc_1, v_dot_2, conc_2, v_dot_3, conc_3]

        # Compute outputs
        yk = model.H(t, xk, uk)
        yk_array = np.array(yk).flatten()

        # Expected outputs
        v_dot_out_expected = v_dot_1 + v_dot_2 + v_dot_3  # 4.5
        mass_flow_total = (
            v_dot_1 * conc_1 + v_dot_2 * conc_2 + v_dot_3 * conc_3
        )
        conc_out_expected = mass_flow_total / v_dot_out_expected

        assert np.allclose(yk_array[0], v_dot_out_expected, rtol=1e-10)
        assert np.allclose(yk_array[1], conc_out_expected, rtol=1e-10)

    def test_output_function_equal_concentrations(self):
        """Test that equal inlet concentrations give same outlet concentration"""
        dt = 1.0
        n_in = 2
        model = FlowMixerDT(dt, n_in)

        conc = 0.75  # Same concentration for both inlets
        v_dot_1 = 1.0
        v_dot_2 = 3.0

        t = 0.0
        xk = []
        uk = [v_dot_1, conc, v_dot_2, conc]

        # Compute outputs
        yk = model.H(t, xk, uk)
        yk_array = np.array(yk).flatten()

        # When all concentrations are equal, output should be same concentration
        assert np.allclose(yk_array[1], conc, rtol=1e-10)

    def test_output_function_zero_flow(self):
        """Test mixer with zero flow in one inlet"""
        dt = 1.0
        n_in = 2
        model = FlowMixerDT(dt, n_in)

        v_dot_1 = 0.0  # No flow from inlet 1
        conc_1 = 0.5
        v_dot_2 = 2.0
        conc_2 = 0.75

        t = 0.0
        xk = []
        uk = [v_dot_1, conc_1, v_dot_2, conc_2]

        # Compute outputs
        yk = model.H(t, xk, uk)
        yk_array = np.array(yk).flatten()

        # With zero flow from inlet 1, output should match inlet 2
        assert np.allclose(yk_array[0], v_dot_2, rtol=1e-10)
        assert np.allclose(yk_array[1], conc_2, rtol=1e-10)

    def test_dt_independence(self):
        """Test that outputs are independent of dt (stateless system)"""
        # Create two mixers with different time steps
        dt = 0.5
        n_in = 2
        model_dt1 = FlowMixerDT(dt, n_in)
        dt = 2.0
        model_dt2 = FlowMixerDT(dt, n_in)

        # Same inputs
        t = 0.0
        xk = []
        uk = [1.5, 0.6, 2.5, 0.8]

        # Compute outputs
        yk1 = model_dt1.H(t, xk, uk)
        yk2 = model_dt2.H(t, xk, uk)

        yk1_array = np.array(yk1).flatten()
        yk2_array = np.array(yk2).flatten()

        # Outputs should be identical regardless of dt (stateless system)
        assert np.allclose(yk1_array, yk2_array, rtol=1e-10)

    def test_consistency_with_continuous_time(self):
        """Test that FlowMixerDT outputs match FlowMixerCT (stateless)"""
        n_in = 2
        model_ct = FlowMixerCT(n_in)
        dt = 1.0
        model_dt = FlowMixerDT(dt, n_in)

        # Test inputs
        t = 0.0
        x = []
        u = [1.2, 0.55, 1.8, 0.65]

        # Compute outputs
        y_ct = model_ct.h(t, x, u)
        y_dt = model_dt.H(t, x, u)

        y_ct_array = np.array(y_ct).flatten()
        y_dt_array = np.array(y_dt).flatten()

        # For stateless system, CT and DT outputs should be identical
        assert np.allclose(y_ct_array, y_dt_array, rtol=1e-10)


class TestRatioControlledFlowMixerCT:
    """Test suite for RatioControlledFlowMixerCT class"""

    def test_initialization_default(self):
        """Test that RatioControlledFlowMixerCT initializes correctly with default parameters"""
        n_in = 2
        model = RatioControlledFlowMixerCT(n_in)

        assert isinstance(model, StateSpaceModelCT)

        # Check model dimensions (stateless system)
        assert model.n == 0, "Should have 0 states (stateless system)"
        assert model.nu == 4, "Should have 4 inputs (2 inlets x 2 properties)"
        assert model.ny == 3, "Should have 3 outputs (2 inflows + concentration)"

        # Check default name
        assert model.name == "FlowMixerModel"

        # Check input/output names
        assert model.input_names == [
            "r_1",
            "conc_in_1",
            "v_dot_out",
            "conc_in_2",
        ]
        assert model.output_names == ["v_dot_in_1", "v_dot_in_2", "conc_out"]
        assert model.state_names == []

        # Check functions exist
        assert hasattr(model, "f")
        assert hasattr(model, "h")

    def test_initialization_with_n_in(self):
        """Test that RatioControlledFlowMixerCT initializes correctly with custom number of inlets"""
        n_in = 3
        model = RatioControlledFlowMixerCT(n_in)

        # Check model dimensions
        assert model.n == 0
        assert model.nu == 2 * n_in  # 6 inputs
        assert model.ny == n_in + 1  # 4 outputs (3 inflows + concentration)

        # Check input names
        expected_inputs = [
            "r_1",
            "conc_in_1",
            "r_2",
            "conc_in_2",
            "v_dot_out",
            "conc_in_3",
        ]
        assert model.input_names == expected_inputs

        # Check output names
        expected_outputs = ["v_dot_in_1", "v_dot_in_2", "v_dot_in_3", "conc_out"]
        assert model.output_names == expected_outputs

    def test_initialization_with_name(self):
        """Test that RatioControlledFlowMixerCT initializes with custom name"""
        custom_name = "MyRatioMixer"
        n_in = 2
        model = RatioControlledFlowMixerCT(n_in, name=custom_name)

        assert model.name == custom_name

    def test_initialization_validation(self):
        """Test initialization validation - n_in must be at least 2"""
        with pytest.raises(ValueError, match="n_in must be at least 2"):
            n_in = 1
            RatioControlledFlowMixerCT(n_in)

        with pytest.raises(ValueError, match="n_in must be at least 2"):
            n_in = 0
            RatioControlledFlowMixerCT(n_in)

    def test_str_representation_two_inlets(self):
        """Test string representation of model with 2 inlets"""
        mixer = RatioControlledFlowMixerCT(2)
        assert str(mixer) == (
            "RatioControlledFlowMixerCT("
            "f=Function(f:(t,x[0],u[4])->(rhs[0]) SXFunction), "
            "h=Function(h:(t,x[0],u[4])->(y[3]) SXFunction), "
            "n=0, nu=4, ny=3, params={}, name='FlowMixerModel', "
            "input_names=['r_1', 'conc_in_1', 'v_dot_out', 'conc_in_2'], "
            "state_names=[], "
            "output_names=['v_dot_in_1', 'v_dot_in_2', 'conc_out'])"
        )

    def test_str_representation_three_inlets(self):
        """Test string representation of model with 3 inlets"""
        mixer = RatioControlledFlowMixerCT(3)
        assert str(mixer) == (
            "RatioControlledFlowMixerCT("
            "f=Function(f:(t,x[0],u[6])->(rhs[0]) SXFunction), "
            "h=Function(h:(t,x[0],u[6])->(y[4]) SXFunction), "
            "n=0, nu=6, ny=4, params={}, name='FlowMixerModel', "
            "input_names=['r_1', 'conc_in_1', 'r_2', 'conc_in_2', 'v_dot_out', 'conc_in_3'], "
            "state_names=[], "
            "output_names=['v_dot_in_1', 'v_dot_in_2', 'v_dot_in_3', 'conc_out'])"
        )

    def test_ode_function_stateless(self):
        """Test the ODE function for stateless system (should return empty)"""
        n_in = 2
        model = RatioControlledFlowMixerCT(n_in)

        t = 0.0
        x = cas.vertcat()  # Empty state vector
        u = cas.vertcat(0.4, 0.5, 3.0, 0.75)  # Arbitrary inputs

        # Compute derivatives (should be empty)
        rhs = model.f(t, x, u)
        rhs_array = np.array(rhs).flatten()

        assert rhs_array.shape == (0,), (
            "Stateless system should have no derivatives"
        )

    def test_output_function_two_inlets(self):
        """Test the output function with 2 inlets"""
        n_in = 2
        model = RatioControlledFlowMixerCT(n_in)

        # Test inputs
        r_1 = 0.4  # 40% of total flow from inlet 1
        conc_1 = 0.5
        v_dot_out = 3.0
        conc_2 = 0.75

        t = 0.0
        x = cas.vertcat()  # Empty state
        u = cas.vertcat(r_1, conc_1, v_dot_out, conc_2)

        # Compute outputs
        y = model.h(t, x, u)
        y_array = np.array(y).flatten()

        # Expected outputs
        # r_2 = 1 - r_1 = 0.6
        v_dot_in_1_expected = r_1 * v_dot_out  # 0.4 * 3.0 = 1.2
        v_dot_in_2_expected = (1 - r_1) * v_dot_out  # 0.6 * 3.0 = 1.8
        conc_out_expected = (
            v_dot_in_1_expected * conc_1 + v_dot_in_2_expected * conc_2
        ) / v_dot_out
        # = (1.2*0.5 + 1.8*0.75) / 3.0 = (0.6 + 1.35) / 3.0 = 0.65

        assert y_array.shape == (3,)
        assert np.allclose(y_array[0], v_dot_in_1_expected, rtol=1e-10)
        assert np.allclose(y_array[1], v_dot_in_2_expected, rtol=1e-10)
        assert np.allclose(y_array[2], conc_out_expected, rtol=1e-10)

    def test_output_function_three_inlets(self):
        """Test the output function with 3 inlets"""
        n_in = 3
        model = RatioControlledFlowMixerCT(n_in)

        # Test inputs
        r_1 = 0.2  # 20% from inlet 1
        conc_1 = 0.5
        r_2 = 0.3  # 30% from inlet 2
        conc_2 = 0.75
        v_dot_out = 5.0
        conc_3 = 0.6

        t = 0.0
        x = cas.vertcat()
        u = cas.vertcat(r_1, conc_1, r_2, conc_2, v_dot_out, conc_3)

        # Compute outputs
        y = model.h(t, x, u)
        y_array = np.array(y).flatten()

        # Expected outputs
        # r_3 = 1 - r_1 - r_2 = 0.5
        v_dot_in_1_expected = r_1 * v_dot_out  # 0.2 * 5.0 = 1.0
        v_dot_in_2_expected = r_2 * v_dot_out  # 0.3 * 5.0 = 1.5
        v_dot_in_3_expected = (1 - r_1 - r_2) * v_dot_out  # 0.5 * 5.0 = 2.5
        mass_flow_total = (
            v_dot_in_1_expected * conc_1 + v_dot_in_2_expected * conc_2 + v_dot_in_3_expected * conc_3
        )
        # = 1.0*0.5 + 1.5*0.75 + 2.5*0.6 = 0.5 + 1.125 + 1.5 = 3.125
        conc_out_expected = mass_flow_total / v_dot_out
        # = 3.125 / 5.0 = 0.625

        assert y_array.shape == (4,)
        assert np.allclose(y_array[0], v_dot_in_1_expected, rtol=1e-10)
        assert np.allclose(y_array[1], v_dot_in_2_expected, rtol=1e-10)
        assert np.allclose(y_array[2], v_dot_in_3_expected, rtol=1e-10)
        assert np.allclose(y_array[3], conc_out_expected, rtol=1e-10)

    def test_output_function_equal_concentrations(self):
        """Test that equal inlet concentrations give same outlet concentration"""
        n_in = 2
        model = RatioControlledFlowMixerCT(n_in)

        conc = 0.75  # Same concentration for both inlets
        r_1 = 0.3
        v_dot_out = 4.0

        t = 0.0
        x = cas.vertcat()
        u = cas.vertcat(r_1, conc, v_dot_out, conc)

        # Compute outputs
        y = model.h(t, x, u)
        y_array = np.array(y).flatten()

        # When all concentrations are equal, output should be same concentration
        assert np.allclose(y_array[2], conc, rtol=1e-10)

    def test_output_function_zero_ratio(self):
        """Test mixer with zero ratio (no flow from inlet 1)"""
        n_in = 2
        model = RatioControlledFlowMixerCT(n_in)

        r_1 = 0.0  # No flow from inlet 1
        conc_1 = 0.5
        v_dot_out = 2.0
        conc_2 = 0.75

        t = 0.0
        x = cas.vertcat()
        u = cas.vertcat(r_1, conc_1, v_dot_out, conc_2)

        # Compute outputs
        y = model.h(t, x, u)
        y_array = np.array(y).flatten()

        # With r_1 = 0, all flow comes from inlet 2
        assert np.allclose(y_array[0], 0.0, rtol=1e-10)  # v_dot_in_1 = 0
        assert np.allclose(y_array[1], v_dot_out, rtol=1e-10)  # v_dot_in_2 = v_dot_out
        assert np.allclose(y_array[2], conc_2, rtol=1e-10)  # conc_out = conc_2

    def test_output_function_equal_ratios(self):
        """Test mixer with equal flow ratios"""
        n_in = 2
        model = RatioControlledFlowMixerCT(n_in)

        r_1 = 0.5  # 50/50 split
        conc_1 = 0.4
        v_dot_out = 6.0
        conc_2 = 0.8

        t = 0.0
        x = cas.vertcat()
        u = cas.vertcat(r_1, conc_1, v_dot_out, conc_2)

        # Compute outputs
        y = model.h(t, x, u)
        y_array = np.array(y).flatten()

        # Expected: both inlets contribute equally
        v_dot_in_expected = v_dot_out / 2.0  # 3.0 each
        conc_out_expected = (conc_1 + conc_2) / 2.0  # 0.6

        assert np.allclose(y_array[0], v_dot_in_expected, rtol=1e-10)
        assert np.allclose(y_array[1], v_dot_in_expected, rtol=1e-10)
        assert np.allclose(y_array[2], conc_out_expected, rtol=1e-10)

    def test_ratios_sum_to_one(self):
        """Test that the implicit calculation ensures ratios sum to 1"""
        n_in = 3
        model = RatioControlledFlowMixerCT(n_in)

        # Set r_1 and r_2; r_3 should be calculated as 1 - r_1 - r_2
        r_1 = 0.25
        r_2 = 0.35
        # r_3 should be 0.4
        v_dot_out = 10.0
        conc_1 = 0.5
        conc_2 = 0.6
        conc_3 = 0.7

        t = 0.0
        x = cas.vertcat()
        u = cas.vertcat(r_1, conc_1, r_2, conc_2, v_dot_out, conc_3)

        # Compute outputs
        y = model.h(t, x, u)
        y_array = np.array(y).flatten()

        # Verify flows sum to outlet flow
        total_inflow = y_array[0] + y_array[1] + y_array[2]
        assert np.allclose(total_inflow, v_dot_out, rtol=1e-10)

        # Verify individual flows match ratios
        assert np.allclose(y_array[0], r_1 * v_dot_out, rtol=1e-10)
        assert np.allclose(y_array[1], r_2 * v_dot_out, rtol=1e-10)
        assert np.allclose(y_array[2], (1 - r_1 - r_2) * v_dot_out, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
