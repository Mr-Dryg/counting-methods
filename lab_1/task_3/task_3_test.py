import numpy as np
import pytest
from .main import simple_iter, seidel

TEST_CASES = [
    pytest.param(
        [
            [19.0, -4.0, -9.0, -1.0],
            [-2.0, 20.0, -2.0, -7.0],
            [6.0, -5.0, -25.0, 9.0],
            [0.0, -3.0, -9.0, 12.0]
        ],
        [100.0, -5.0, 34.0, 69.0],
        id="system_1"
    ),
    pytest.param(
        [
            [24.0, 2.0, 4.0, -9.0],
            [-6.0, -27.0, -8.0, -6.0],
            [-4.0, 8.0, 19.0, 6.0],
            [4.0, 5.0, -3.0, -13.0]
        ],
        [-9.0, -76.0, -79.0, -70.0],
        id="system_2"
    ),
    pytest.param(
        [
            [-23.0, -7.0, 5.0, 2.0],
            [-7.0, -21.0, 4.0, 9.0],
            [9.0, 5.0, -31.0, -8.0],
            [0.0, 1.0, -2.0, 10.0]
        ],
        [-26.0, -55.0, -58.0, -24.0],
        id="system_3"
    ),
    pytest.param(
        [
            [26.0, -9.0, -8.0, 8.0],
            [9.0, -21.0, -2.0, 8.0],
            [-3.0, 2.0, -18.0, 8.0],
            [1.0, -6.0, -1.0, 11.0]
        ],
        [20.0, -164.0, 140.0, -81.0],
        id="system_4"
    )
]


@pytest.mark.parametrize("A, b", TEST_CASES)
def test_simple_iter(A, b):
    eps = 1e-5
    history = simple_iter(A, b, eps)
    calculated_x = history[-1][1]
    
    expected_x = np.linalg.solve(A, b)
    np.testing.assert_allclose(calculated_x, expected_x, rtol=eps*10, atol=eps*10)


@pytest.mark.parametrize("A, b", TEST_CASES)
def test_seidel(A, b):
    eps = 1e-5
    history = seidel(A, b, eps)
    calculated_x = history[-1][1]
    
    expected_x = np.linalg.solve(A, b)
    np.testing.assert_allclose(calculated_x, expected_x, rtol=eps*10, atol=eps*10)
