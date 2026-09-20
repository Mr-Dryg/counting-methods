from main import dtma
import numpy as np
import pytest

TEST_CASES = [
    pytest.param(
        [
            [-11.0, -9.0, 0.0, 0.0, 0.0],
            [5.0, -15.0, -2.0, 0.0, 0.0],
            [0.0, -8.0, 11.0, -3.0, 0.0],
            [0.0, 0.0, 6.0, -15.0, 4.0],
            [0.0, 0.0, 0.0, 3.0, 6.0]
        ],
        [-122.0, -48.0, -14.0, -50.0, 42.0],
        id="system_1"
    ),
    pytest.param(
        [
            [10.0, 5.0, 0.0, 0.0, 0.0],
            [3.0, 10.0, -2.0, 0.0, 0.0],
            [0.0, 2.0, -9.0, -5.0, 0.0],
            [0.0, 0.0, 5.0, 16.0, -4.0],
            [0.0, 0.0, 0.0, -8.0, 16.0]
        ],
        [-120.0, -91.0, 5.0, -74.0, -56.0],
        id="system_2"
    ),
    pytest.param(
        [
            [13.0, -5.0, 0.0, 0.0, 0.0],
            [-4.0, 9.0, -5.0, 0.0, 0.0],
            [0.0, -1.0, -12.0, -6.0, 0.0],
            [0.0, 0.0, 6.0, 20.0, -5.0],
            [0.0, 0.0, 0.0, 4.0, 5.0]
        ],
        [-66.0, -47.0, -43.0, -74.0, 14.0],
        id="system_3"
    )
]

@pytest.mark.parametrize("A, b", TEST_CASES)
def test_task(A, b):
    calculated_x = dtma(A, b)

    expected_x = np.linalg.solve(A, b)
    np.testing.assert_allclose(calculated_x, expected_x, rtol=1e-7, atol=1e-7)
