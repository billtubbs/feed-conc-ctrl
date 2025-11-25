"""Unit tests for MixingTankModelDT (discrete-time) class"""

import pytest
import numpy as np
import casadi as cas
from feed_conc_ctrl.models import MixingTankModelDT
from cas_models.discrete_time.models import StateSpaceModelDT


class TestMixingTankModelDT:
    """Test suite for MixingTankModelDT class"""

    def test_initialization_with_D(self):
        """Test that MixingTankModelDT initializes correctly with diameter"""
        D = 5  # tank diameter [m]
        dt = 0.25

        model = MixingTankModelDT(D=D, dt=dt)

        assert isinstance(model, StateSpaceModelDT)

        # Check dimensions
        assert model.n == 2
        assert model.nu == 3
        assert model.ny == 3
        assert model.dt == dt

        # Check tank parameters
        assert model.D == D
        assert model.A == np.pi * D**2 / 4

        # Check it has discrete-time functions
        assert hasattr(model, 'F')
        assert hasattr(model, 'H')

    def test_initialization_with_A(self):
        """Test that MixingTankModelDT initializes correctly with area"""
        A = 19.634954084936208  # tank area [m^2] for D=5
        dt = 0.25

        model = MixingTankModelDT(A=A, dt=dt)

        assert isinstance(model, StateSpaceModelDT)

        # Check dimensions
        assert model.n == 2
        assert model.nu == 3
        assert model.ny == 3
        assert model.dt == dt

        # Check tank parameters
        assert model.D is None  # Unknown when A is provided
        assert model.A == A

        # Check it has discrete-time functions
        assert hasattr(model, 'F')
        assert hasattr(model, 'H')

    def test_initialization_validation(self):
        """Test initialization validation - must provide D or A, not both"""
        # Test providing neither
        with pytest.raises(ValueError, match="Must provide either D .* or A .*"):
            MixingTankModelDT(dt=0.25)

        # Test providing both
        with pytest.raises(ValueError, match="Cannot provide both D and A"):
            MixingTankModelDT(D=5, A=15.7, dt=0.25)

    def test_equal_flows_same_concentration(self):
        """Test 1 from notebook - equal flows, same concentration, no change"""
        D = 5  # tank diameter [m]
        A = np.pi * D**2 / 4
        dt = 0.25

        model = MixingTankModelDT(D=D, dt=dt)

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

        # Check output function
        yk = model.H(t, xk, uk)
        yk_array = np.array(yk).flatten()
        assert yk_array.shape == (3,), "Should have 3 outputs"
        assert np.allclose(yk_array[0], L, rtol=1e-5)
        assert np.allclose(yk_array[1], m, rtol=1e-5)
        assert np.allclose(yk_array[2], conc, rtol=1e-5)

    def test_increasing_volume(self):
        """Test 2 from notebook - increasing volume"""
        D = 5  # tank diameter [m]
        A = np.pi * D**2 / 4
        dt = 0.25

        model = MixingTankModelDT(D=D, dt=dt)

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

        assert np.allclose(xkp1_array[0], L2, rtol=1e-5)
        assert np.allclose(xkp1_array[1], m2, rtol=1e-5)

    def test_equal_flows_convergence_to_equilibrium(self):
        """Test 4 from notebook - equal flows, convergence to equilibrium"""
        D = 5  # tank diameter [m]
        A = np.pi * D**2 / 4
        dt = 0.25

        model = MixingTankModelDT(D=D, dt=dt)

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
        # Final mass from simulation
        assert np.allclose(xk_array[1], 45.24104157675106, rtol=1e-5)
        # Should be close to equilibrium (within 10%)
        assert np.allclose(xk_array[1], m_final, rtol=0.1)

    def test_equal_flows_washout_to_zero(self):
        """Test 5 from notebook - equal flows, washing out to zero concentration"""
        D = 5  # tank diameter [m]
        dt = 0.25

        model = MixingTankModelDT(D=D, dt=dt)

        # Initial conditions - from Test 5 in notebook
        L = 5
        m = 39.2699
        v_dot_in = 1
        conc_in = 0
        v_dot_out = 1

        # Simulate 1000 steps
        t = 0.0
        xk = cas.vertcat(L, m)
        uk = cas.vertcat(v_dot_in, conc_in, v_dot_out)
        nT = 1000

        for _ in range(nT):
            xk = model.F(t, xk, uk)
            t += dt

        # Check final state matches notebook
        xk_array = np.array(xk).flatten()

        # Level should remain constant
        assert np.allclose(xk_array[0], L, rtol=1e-5)
        # Final mass from simulation (washing out to near zero)
        assert np.allclose(xk_array[1], 3.0770742683043175, rtol=1e-5)

    def test_unequal_flows_from_notebook_test6(self):
        """Test 6 from notebook - unequal flows, step response"""
        D = 5  # tank diameter [m]
        dt = 0.25

        model = MixingTankModelDT(D=D, dt=dt)

        # Initial conditions (from Test 6 in notebook)
        L = 5
        m = 25

        # Inputs
        v_dot_in = 1.5
        conc_in = 0.1
        v_dot_out = 1

        # Simulate 1000 steps
        t = 0.0
        xk = cas.vertcat(L, m)
        uk = cas.vertcat(v_dot_in, conc_in, v_dot_out)
        nT = 1000

        for _ in range(nT):
            xk = model.F(t, xk, uk)
            t += dt

        # Check final state matches notebook
        xk_array = np.array(xk).flatten()

        # Expected values from simulation
        assert np.allclose(xk_array[0], 11.366197723675526, rtol=1e-5)
        assert np.allclose(xk_array[1], 25.255488883902494, rtol=1e-5)

    def test_output_concentration_calculation(self):
        """Test that output concentration is correctly calculated"""
        D = 5  # tank diameter [m]
        A = np.pi * D**2 / 4
        dt = 0.25

        model = MixingTankModelDT(D=D, dt=dt)

        # Test with specific concentration
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
        yk_array = np.array(yk).flatten()

        # Check all three outputs
        assert np.allclose(yk_array[0], L, rtol=1e-5)
        assert np.allclose(yk_array[1], m, rtol=1e-5)

        # Check concentration calculation
        expected_conc = m / (A * L)
        assert np.allclose(yk_array[2], expected_conc, rtol=1e-5)
        assert np.allclose(yk_array[2], conc, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
