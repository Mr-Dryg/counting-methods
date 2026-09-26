import numpy as np
import pytest
from .main import eigenvalue

TEST_CASES = [
    pytest.param(
        [
            [3.0, -7.0, -1.0],
            [-9.0, -8.0, 7.0],
            [5.0, 2.0, 2.0]
        ],
        id="matrix_1"
    ),
    pytest.param(
        [
            [-6.0, -4.0, 0.0],
            [-7.0, 6.0, -7.0],
            [-2.0, -6.0, -7.0]
        ],
        id="matrix_2"
    ),
    pytest.param(
        [
            [7.0, 6.0, -3.0],
            [5.0, -6.0, 8.0],
            [-7.0, 4.0, -5.0]
        ],
        id="matrix_11"
    ),
    pytest.param(
        [
            [5.0, -1.0, -2.0],
            [-4.0, 3.0, -3.0],
            [-2.0, -1.0, 1.0]
        ],
        id="matrix_12"
    ),
    pytest.param(
        [
            [7.0, 4.0, 2.0],
            [-8.0, -6.0, 5.0],
            [9.0, 0.0, -8.0]
        ],
        id="matrix_21"
    ),
    pytest.param(
        [
            [-1.0, 8.0, 5.0],
            [8.0, -4.0, 4.0],
            [2.0, 9.0, -2.0]
        ],
        id="matrix_22"
    )
]


@pytest.mark.parametrize("A_list", TEST_CASES)
def test_eigenvalue(A_list):
    A = np.array(A_list, dtype=np.float64)
    eps = 1e-6

    calculated_lambdas, _ = eigenvalue(A, eps)

    expected_lambdas = np.linalg.eigvals(A)

    calculated_sorted = np.sort_complex(calculated_lambdas)
    expected_sorted = np.sort_complex(expected_lambdas)

    np.testing.assert_allclose(
        calculated_sorted, 
        expected_sorted, 
        rtol=eps, 
        atol=eps
    )
