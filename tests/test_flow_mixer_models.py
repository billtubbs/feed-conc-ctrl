"""Unit tests for FlowMixerCT and FlowMixerDT classes"""

import pytest
import numpy as np
import casadi as cas
from feed_conc_ctrl.models import FlowMixerCT, FlowMixerDT
from cas_models.continuous_time.models import StateSpaceModelCT
from cas_models.discrete_time.models import StateSpaceModelDT


class TestFlowMixerCT:
    """Test suite for FlowMixerCT class"""

    def test_initialization_default(self):
        """Test that FlowMixerCT initializes correctly with default parameters"""
        model = FlowMixerCT()

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
        model = FlowMixerCT(n_in=n_in)

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
        model = FlowMixerCT(name=custom_name)

        assert model.name == custom_name

    def test_initialization_validation(self):
        """Test initialization validation - n_in must be at least 2"""
        with pytest.raises(ValueError, match="n_in must be at least 2"):
            FlowMixerCT(n_in=1)

        with pytest.raises(ValueError, match="n_in must be at least 2"):
            FlowMixerCT(n_in=0)

    def test_ode_function_stateless(self):
        """Test the ODE function for stateless system (should return empty)"""
        model = FlowMixerCT(n_in=2)

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
        model = FlowMixerCT(n_in=2)

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
        model = FlowMixerCT(n_in=3)

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
        model = FlowMixerCT(n_in=2)

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
        model = FlowMixerCT(n_in=2)

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
        model = FlowMixerDT(dt, n_in=n_in)

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
        custom_name = "MyMixer"
        model = FlowMixerDT(dt, n_in=2, name=custom_name)

        assert model.name == custom_name

    def test_initialization_validation(self):
        """Test initialization validation - n_in must be at least 2"""
        dt = 1.0

        with pytest.raises(ValueError, match="n_in must be at least 2"):
            FlowMixerDT(dt, n_in=1)

        with pytest.raises(ValueError, match="n_in must be at least 2"):
            FlowMixerDT(dt, n_in=0)

    def test_state_transition_stateless(self):
        """Test state transition for stateless system (should do nothing)"""
        dt = 1.0
        model = FlowMixerDT(dt, n_in=2)

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
        model = FlowMixerDT(dt, n_in=2)

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
        model = FlowMixerDT(dt, n_in=3)

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
        model = FlowMixerDT(dt, n_in=2)

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
        model = FlowMixerDT(dt, n_in=2)

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
        model_dt1 = FlowMixerDT(dt=0.5, n_in=2)
        model_dt2 = FlowMixerDT(dt=2.0, n_in=2)

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
        dt = 1.0
        n_in = 2
        model_ct = FlowMixerCT(n_in=n_in)
        model_dt = FlowMixerDT(dt, n_in=n_in)

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
