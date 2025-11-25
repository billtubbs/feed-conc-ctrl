"""Unit tests for MixingTankModel class"""

import pytest
import numpy as np
import casadi as cas
from feed_conc_ctrl.models import MixingTankModel


class TestMixingTankModel:
    """Test suite for MixingTankModel class"""

    def test_initialization(self):
        """Test that MixingTankModel initializes correctly"""
        D = 5  # tank diameter [m]
        dt = 0.25  # time step [hr]

        model = MixingTankModel(D=D, dt=dt)

        # Check dimensions
        assert model.n == 2, "Should have 2 states"
        assert model.nu == 3, "Should have 3 inputs"
        assert model.ny == 3, "Should have 3 outputs"

        # Check time step
        assert model.dt == dt

        # Check tank parameters
        assert model.D == D
        assert model.A == np.pi * D

        # Check names
        assert model.state_names == ["L", "m"]
        assert model.input_names == ["v_dot_in", "conc_in", "v_dot_out"]
        assert model.output_names == ["L", "m", "conc_out"]

    def test_state_transition_equal_flows(self):
        """Test 1 from notebook - equal in/out flows with same concentration"""
        D = 5  # tank diameter [m]
        A = np.pi * D
        dt = 0.25

        model = MixingTankModel(D=D, dt=dt)

        # Initial conditions
        L = 5
        conc = 0.2
        m = A * L * conc

        # Inputs - equal flows with same concentration
        v_dot_in = 1
        v_dot_out = 1

        # Function arguments
        t = 0.0
        xk = cas.vertcat(L, m)
        uk = cas.vertcat(v_dot_in, conc, v_dot_out)

        # Check state transition
        xkp1 = model.F(t, xk, uk)
        xkp1_array = np.array(xkp1).flatten()

        # With equal flows and same concentration, states should remain constant
        assert np.allclose(xkp1_array[0], L, rtol=1e-5)
        assert np.allclose(xkp1_array[1], m, rtol=1e-5)

    def test_state_transition_equal_flows_different_conc(self):
        """Test equal flows but different concentrations - mass should change"""
        D = 5  # tank diameter [m]
        A = np.pi * D
        dt = 0.25

        model = MixingTankModel(D=D, dt=dt)

        # Initial conditions
        L = 5
        m = 0  # Start with no mass

        # Inputs - equal flows but inlet has concentration
        v_dot_in = 1
        conc_in = 0.5
        v_dot_out = 1

        # Function arguments
        t = 0.0
        xk = cas.vertcat(L, m)
        uk = cas.vertcat(v_dot_in, conc_in, v_dot_out)

        # Check state transition
        xkp1 = model.F(t, xk, uk)
        xkp1_array = np.array(xkp1).flatten()

        # Level should stay constant
        assert np.allclose(xkp1_array[0], L, rtol=1e-5)
        # Mass should increase toward equilibrium: m_eq = L*A*conc_in
        # After one time step: m = L*A*conc_in * (1 - exp(-v_dot*dt/(L*A)))
        m_expected = L * A * conc_in * (1 - np.exp(-v_dot_in * dt / (L * A)))
        assert np.allclose(xkp1_array[1], m_expected, rtol=1e-5)

    def test_equal_flows_convergence_from_notebook_test4(self):
        """Test 4 from notebook - equal flows convergence to equilibrium"""
        D = 5  # tank diameter [m]
        A = np.pi * D
        dt = 0.25

        model = MixingTankModel(D=D, dt=dt)

        # Initial conditions - from Test 4 in notebook
        L = 5
        m = 0
        v_dot_in = 1
        conc_in = 0.5
        v_dot_out = 1

        # Simulate 1000 steps like in the notebook
        t = 0.0
        xk = cas.vertcat(L, m)
        uk = cas.vertcat(v_dot_in, conc_in, v_dot_out)
        nT = 1000

        for _ in range(nT):
            xk = model.F(t, xk, uk)
            t += dt

        # Check final state matches notebook
        xk_array = np.array(xk).flatten()
        m_final = A * L * conc_in

        # Level should remain constant
        assert np.allclose(xk_array[0], L, rtol=1e-5)
        # Final mass from notebook assertion
        assert np.allclose(xk_array[1], 37.64189612141962, rtol=1e-5)
        # Should be close to equilibrium
        assert np.allclose(xk_array[1], m_final, rtol=0.05)

    def test_state_transition_from_notebook_test6(self):
        """Test 6 from notebook - step response with different flows"""
        D = 5  # tank diameter [m]
        A = np.pi * D
        dt = 0.25

        model = MixingTankModel(D=D, dt=dt)

        # Initial conditions (from Test 6 in notebook)
        L = 5
        m = 25

        # Inputs
        v_dot_in = 1.5
        conc_in = 0.1
        v_dot_out = 1

        # Function arguments
        t = 0.0
        xk = cas.vertcat(L, m)
        uk = cas.vertcat(v_dot_in, conc_in, v_dot_out)

        # Check state transition
        xkp1 = model.F(t, xk, uk)
        xkp1_array = np.array(xkp1).flatten()

        # Expected values from notebook: [5.007957747154594, 24.958052546626035]
        assert np.allclose(xkp1_array[0], 5.007957747154594, rtol=1e-5)
        assert np.allclose(xkp1_array[1], 24.958052546626035, rtol=1e-5)

    def test_state_transition_increasing_volume(self):
        """Test 2 from notebook - increasing volume"""
        D = 5  # tank diameter [m]
        A = np.pi * D
        dt = 0.25

        model = MixingTankModel(D=D, dt=dt)

        # Initial conditions
        L = 5
        conc = 0.2
        m = A * L * conc

        # Inputs - higher inflow than outflow
        v_dot_in = 2
        v_dot_out = 1

        # Function arguments
        t = 0.0
        xk = cas.vertcat(L, m)
        uk = cas.vertcat(v_dot_in, conc, v_dot_out)

        # Check state transition
        xkp1 = model.F(t, xk, uk)
        xkp1_array = np.array(xkp1).flatten()

        # Expected values
        L2 = L + (v_dot_in - v_dot_out) * dt / A
        m2 = (A * L + (v_dot_in - v_dot_out) * dt) * conc

        assert np.allclose(xkp1_array, [L2, m2], rtol=1e-5)

    def test_output_function(self):
        """Test output function returns 3 outputs including concentration"""
        D = 5  # tank diameter [m]
        A = np.pi * D
        dt = 0.25

        model = MixingTankModel(D=D, dt=dt)

        # Initial conditions
        L = 5
        conc = 0.2
        m = A * L * conc

        # Inputs
        v_dot_in = 1
        v_dot_out = 1

        # Function arguments
        t = 0.0
        xk = cas.vertcat(L, m)
        uk = cas.vertcat(v_dot_in, conc, v_dot_out)

        # Check output function
        yk = model.H(t, xk, uk)
        yk_array = np.array(yk).flatten()

        # Expected outputs
        expected_L = L
        expected_m = m
        expected_conc_out = m / (A * L)

        assert yk_array.shape == (3,), "Should have 3 outputs"
        assert np.allclose(yk_array[0], expected_L, rtol=1e-5)
        assert np.allclose(yk_array[1], expected_m, rtol=1e-5)
        assert np.allclose(yk_array[2], expected_conc_out, rtol=1e-5)
        # Check that concentration matches input concentration
        assert np.allclose(yk_array[2], conc, rtol=1e-5)

    def test_concentration_calculation(self):
        """Test that outflow concentration is correctly calculated"""
        D = 5  # tank diameter [m]
        A = np.pi * D
        dt = 0.25

        model = MixingTankModel(D=D, dt=dt)

        # Test with different concentration
        L = 8
        conc = 0.5
        m = A * L * conc

        v_dot_in = 1
        v_dot_out = 1

        t = 0.0
        xk = cas.vertcat(L, m)
        uk = cas.vertcat(v_dot_in, conc, v_dot_out)

        # Get outputs
        yk = model.H(t, xk, uk)
        conc_out = float(yk[2])

        # Check concentration
        expected_conc = m / (A * L)
        assert np.allclose(conc_out, expected_conc, rtol=1e-5)
        assert np.allclose(conc_out, conc, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
