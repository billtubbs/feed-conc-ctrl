import numpy as np
import pytest
from bounded_random_walk import sample_bounded_random_walk


# Shared test parameters
SD_E = 1.0
R1 = -5.0
R2 = 5.0
A1 = 0.5
A2 = 0.5
SIZE = 100
PHI = 0.5
N_WALKS = 4

SD_E_ARR = np.full(N_WALKS, SD_E)
R1_ARR = np.full(N_WALKS, R1)
R2_ARR = np.full(N_WALKS, R2)
A1_ARR = np.full(N_WALKS, A1)
A2_ARR = np.full(N_WALKS, A2)
XKM1_ARR = np.array([0.0, 1.0, -1.0, 2.0])


class TestSampleBRWScalarAll:
    """Case 1: all arguments are scalars — one random walk."""

    def test_output_shape(self):
        p = sample_bounded_random_walk(SD_E, R1, R2, A1, A2, SIZE)
        assert p.shape == (SIZE,)
        assert p.dtype == np.float64

    def test_custom_initial_state(self):
        p = sample_bounded_random_walk(SD_E, R1, R2, A1, A2, SIZE, xkm1=2.0)
        assert p.shape == (SIZE,)

    def test_reproducible_with_seed(self):
        p1 = sample_bounded_random_walk(SD_E, R1, R2, A1, A2, SIZE, seed=42)
        p2 = sample_bounded_random_walk(SD_E, R1, R2, A1, A2, SIZE, seed=42)
        np.testing.assert_array_equal(p1, p2)

    def test_output_within_bounds(self):
        p = sample_bounded_random_walk(SD_E, R1, R2, A1, A2, SIZE, seed=0)
        assert np.all(p > R1 * 5)
        assert np.all(p < R2 * 5)


class TestSampleBRWScalarSdE:
    """Case 2: sd_e is scalar, BRW params are arrays — multiple walks, same std."""

    def test_output_shape(self):
        p = sample_bounded_random_walk(SD_E, R1_ARR, R2_ARR, A1_ARR, A2_ARR, SIZE)
        assert p.shape == (SIZE, N_WALKS)
        assert p.dtype == np.float64

    def test_custom_initial_state(self):
        p = sample_bounded_random_walk(
            SD_E, R1_ARR, R2_ARR, A1_ARR, A2_ARR, SIZE, xkm1=XKM1_ARR
        )
        assert p.shape == (SIZE, N_WALKS)

    def test_reproducible_with_seed(self):
        p1 = sample_bounded_random_walk(
            SD_E, R1_ARR, R2_ARR, A1_ARR, A2_ARR, SIZE, seed=42
        )
        p2 = sample_bounded_random_walk(
            SD_E, R1_ARR, R2_ARR, A1_ARR, A2_ARR, SIZE, seed=42
        )
        np.testing.assert_array_equal(p1, p2)

    def test_output_within_bounds(self):
        p = sample_bounded_random_walk(
            SD_E, R1_ARR, R2_ARR, A1_ARR, A2_ARR, SIZE, seed=0
        )
        assert np.all(p > R1 * 5)
        assert np.all(p < R2 * 5)

    def test_walks_are_independent(self):
        p = sample_bounded_random_walk(
            SD_E, R1_ARR, R2_ARR, A1_ARR, A2_ARR, SIZE, seed=0
        )
        # Independent walks should not be identical
        for i in range(N_WALKS - 1):
            assert not np.array_equal(p[:, i], p[:, i + 1])


class TestSampleBRWArrayAll:
    """Case 3: all arguments are arrays — multiple independent walks."""

    def test_output_shape(self):
        p = sample_bounded_random_walk(
            SD_E_ARR, R1_ARR, R2_ARR, A1_ARR, A2_ARR, SIZE
        )
        assert p.shape == (SIZE, N_WALKS)
        assert p.dtype == np.float64

    def test_custom_initial_state(self):
        p = sample_bounded_random_walk(
            SD_E_ARR, R1_ARR, R2_ARR, A1_ARR, A2_ARR, SIZE, xkm1=XKM1_ARR
        )
        assert p.shape == (SIZE, N_WALKS)

    def test_reproducible_with_seed(self):
        p1 = sample_bounded_random_walk(
            SD_E_ARR, R1_ARR, R2_ARR, A1_ARR, A2_ARR, SIZE, seed=42
        )
        p2 = sample_bounded_random_walk(
            SD_E_ARR, R1_ARR, R2_ARR, A1_ARR, A2_ARR, SIZE, seed=42
        )
        np.testing.assert_array_equal(p1, p2)

    def test_output_within_bounds(self):
        p = sample_bounded_random_walk(
            SD_E_ARR, R1_ARR, R2_ARR, A1_ARR, A2_ARR, SIZE, seed=0
        )
        assert np.all(p > R1 * 5)
        assert np.all(p < R2 * 5)

    def test_walks_are_independent(self):
        p = sample_bounded_random_walk(
            SD_E_ARR, R1_ARR, R2_ARR, A1_ARR, A2_ARR, SIZE, seed=0
        )
        for i in range(N_WALKS - 1):
            assert not np.array_equal(p[:, i], p[:, i + 1])


class TestSampleBRWValidation:
    """Test input validation."""

    def test_mismatched_brw_param_shapes(self):
        with pytest.raises(ValueError, match="r1, r2, a1, a2 must all have the same shape"):
            sample_bounded_random_walk(SD_E, R1_ARR, R2, A1_ARR, A2_ARR, SIZE)

    def test_mismatched_sd_e_shape(self):
        wrong_sd_e = np.full(N_WALKS + 1, SD_E)
        with pytest.raises(ValueError, match="sd_e must be a scalar or have shape"):
            sample_bounded_random_walk(wrong_sd_e, R1_ARR, R2_ARR, A1_ARR, A2_ARR, SIZE)

    def test_mismatched_xkm1_shape(self):
        wrong_xkm1 = np.zeros(N_WALKS + 1)
        with pytest.raises(ValueError, match="xkm1 must have shape"):
            sample_bounded_random_walk(
                SD_E, R1_ARR, R2_ARR, A1_ARR, A2_ARR, SIZE, xkm1=wrong_xkm1
            )
