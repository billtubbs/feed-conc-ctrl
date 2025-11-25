"""Unit tests for MixingTankModelCT (continuous-time) class"""

import pytest
import numpy as np
import casadi as cas
from feed_conc_ctrl.models import MixingTankModelCT
from cas_models.continuous_time.models import StateSpaceModelCT


class TestMixingTankModelCT:
    """Test suite for MixingTankModelCT class"""

    def test_initialization(self):
        """Test that MixingTankModelCT initializes correctly"""
        D = 5  # tank diameter [m]

        model = MixingTankModelCT(D=D)

        assert isinstance(model, StateSpaceModelCT)

        # Check model dimensions
        assert model.n == 2, "Should have 2 states"
        assert model.nu == 3, "Should have 3 inputs"
        assert model.ny == 3, "Should have 3 outputs"

        # Check tank parameters
        assert model.D == D
        assert model.A == np.pi * D

        # Check names
        assert model.state_names == ["L", "m"]
        assert model.input_names == ["v_dot_in", "conc_in", "v_dot_out"]
        assert model.output_names == ["L", "m", "conc_out"]

        # Check functions exist
        assert hasattr(model, 'f')
        assert hasattr(model, 'h')

    def test_ode_function(self):
        """Test the ODE right-hand side function"""
        D = 5
        A = np.pi * D
        model = MixingTankModelCT(D=D)

        # Test case: equal flows
        L = 5.0
        m = 20.0
        v_dot_in = 1.0
        conc_in = 0.3
        v_dot_out = 1.0

        t = 0.0
        x = cas.vertcat(L, m)
        u = cas.vertcat(v_dot_in, conc_in, v_dot_out)

        # Compute derivatives
        rhs = model.f(t, x, u)
        rhs_array = np.array(rhs).flatten()

        # Expected derivatives
        dL_dt = (v_dot_in - v_dot_out) / A
        dm_dt = v_dot_in * conc_in - v_dot_out * m / (A * L)

        assert np.allclose(rhs_array[0], dL_dt, rtol=1e-10)
        assert np.allclose(rhs_array[1], dm_dt, rtol=1e-10)

    def test_output_function(self):
        """Test the output function"""
        D = 5
        A = np.pi * D
        model = MixingTankModelCT(D=D)

        L = 8.0
        m = 50.0
        conc_expected = m / (A * L)

        t = 0.0
        x = cas.vertcat(L, m)
        u = cas.vertcat(1.0, 0.2, 1.0)

        # Compute outputs
        y = model.h(t, x, u)
        y_array = np.array(y).flatten()

        assert y_array.shape == (3,)
        assert np.allclose(y_array[0], L, rtol=1e-10)
        assert np.allclose(y_array[1], m, rtol=1e-10)
        assert np.allclose(y_array[2], conc_expected, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
