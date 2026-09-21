from .main import decomp, solve, permut, det, inv
import numpy as np
import pytest

TEST_CASES = [
    pytest.param(
        [
            [1.0, 2.0, -2.0, 6.0],
            [-3.0, -5.0, 14.0, 13.0],
            [1.0, 2.0, -2.0, -2.0],
            [-2.0, -4.0, 5.0, 10.0]
        ],
        [24.0, 41.0, 0.0, 20.0],
        id="system_1"
    ),
    pytest.param(
        [
            [2.0, 7.0, -8.0, 6.0],
            [4.0, 4.0, 0.0, -7.0],
            [-1.0, -3.0, 6.0, 3.0],
            [9.0, -7.0, -2.0, -8.0]
        ],
        [-39.0, 41.0, 4.0, 113.0],
        id="system_2"
    ),
    pytest.param(
        [
            [9.0, -5.0, -6.0, 3.0],
            [1.0, -7.0, 1.0, 0.0],
            [3.0, -4.0, 9.0, 0.0],
            [6.0, -1.0, 9.0, 8.0]
        ],
        [-8.0, 38.0, 47.0, -8.0],
        id="system_3"
    )
]

@pytest.mark.parametrize("A, b", TEST_CASES)
def test_task(A, b):
    L, U, P, p = decomp(A)
    b_perm = permut(b, P)
    calculated_x = solve(L, U, b_perm)

    expected_x = np.linalg.solve(A, b)
    np.testing.assert_allclose(calculated_x, expected_x, rtol=1e-7, atol=1e-7)

    calculated_det = det(U, p)
    expected_det = np.linalg.det(A)
    np.testing.assert_allclose(calculated_det, expected_det, rtol=1e-7, atol=1e-7)

    calculated_inv = inv(A)
    expected_inv = np.linalg.inv(A)
    np.testing.assert_allclose(calculated_inv, expected_inv, rtol=1e-7, atol=1e-7)
