import numpy as np
import pytest
from .main import rotation_method

TEST_CASES = [
    pytest.param(
        [
            [-7.0, 4.0, 5.0],
            [4.0, -6.0, -9.0],
            [5.0, -9.0, -8.0]
        ],
        id="matrix_1"
    ),
    pytest.param(
        [
            [-9.0, 7.0, 5.0],
            [7.0, 8.0, 9.0],
            [5.0, 9.0, 8.0]
        ],
        id="matrix_2"
    ),
    pytest.param(
        [
            [-3.0, -2.0, -4.0],
            [-2.0, -3.0, -7.0],
            [-4.0, -7.0, -3.0]
        ],
        id="matrix_11"
    ),
    pytest.param(
        [
            [7.0, 3.0, -1.0],
            [3.0, -7.0, -8.0],
            [-1.0, -8.0, -2.0]
        ],
        id="matrix_12"
    ),
    pytest.param(
        [
            [-4.0, 1.0, 7.0],
            [1.0, 8.0, -5.0],
            [7.0, -5.0, 1.0]
        ],
        id="matrix_21"
    ),
    pytest.param(
        [
            [2.0, -9.0, 4.0],
            [-9.0, 0.0, 9.0],
            [4.0, 9.0, 6.0]
        ],
        id="matrix_22"
    )
]


@pytest.mark.parametrize("A_list", TEST_CASES)
def test_rotation_method(A_list):
    eps = 1e-3
    A_original = np.array(A_list, dtype=np.float64)

    result = rotation_method(A_original.copy(), eps)
    assert result is not None, "Функция вернула None (проверка check_sim не прошла)"

    calculated_vectors_list, calculated_eigenvalues_list, iters = result

    calc_eigenvalues = np.array(calculated_eigenvalues_list)
    calc_vectors = np.column_stack(calculated_vectors_list)
    
    sort_indices = np.argsort(calc_eigenvalues)
    calc_eigenvalues = calc_eigenvalues[sort_indices]
    calc_vectors = calc_vectors[:, sort_indices]

    expected_eigenvalues, expected_vectors = np.linalg.eigh(A_original)

    np.testing.assert_allclose(calc_eigenvalues, expected_eigenvalues, rtol=eps*10, atol=eps*10)

    for i in range(len(calc_eigenvalues)):
        v_calc = calc_vectors[:, i]
        v_exp = expected_vectors[:, i]

        # Способ А: Проверка через главное уравнение Av = lambda * v
        np.testing.assert_allclose(A_original @ v_calc, calc_eigenvalues[i] * v_calc, atol=eps*10)

        # Способ Б: Прямое сравнение с векторами numpy с коррекцией знака направление (+/-)
        if np.dot(v_calc, v_exp) < 0:
            v_calc = -v_calc

        np.testing.assert_allclose(v_calc, v_exp, atol=eps*10)

    identity_matrix = np.eye(A_original.shape[0])
    np.testing.assert_allclose(calc_vectors.T @ calc_vectors, identity_matrix, atol=1e-5)
