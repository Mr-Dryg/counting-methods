from .main import decomp, solve, permut, det, inv
import numpy as np

def test_decomp_and_reconstruction():
    A = [
        [1.0, 2.0, 3.0],
        [0.0, 1.0, 4.0],
        [5.0, 6.0, 1.0]
    ]
    
    L, U, P, _ = decomp(A)
    
    # Проверка треугольной формы
    assert np.allclose(L, np.tril(L))
    assert np.allclose(U, np.triu(U))
    assert np.allclose(np.diag(L), 1.0)
    
    # Проверка главного свойства LUP разложения: P * A = L * U
    A_perm = permut(A, P)
    LU_product = np.dot(L, U)
    np.testing.assert_allclose(A_perm, LU_product, rtol=1e-7, atol=1e-7)


def test_solve_system():
    A = [
        [10.0, 1.0, 1.0],
        [2.0, 10.0, 1.0],
        [2.0, 2.0, 10.0]
    ]
    b = [12.0, 13.0, 14.0]
    
    L, U, P, _ = decomp(A)
    b_perm = permut(b, P)
    x = solve(L, U, b_perm)
    
    expected_x = [1.0, 1.0, 1.0]
    np.testing.assert_allclose(x, expected_x, rtol=1e-7, atol=1e-7)


def test_determinant():
    A = [
        [1.0, 2.0, 3.0],
        [0.0, 1.0, 4.0],
        [5.0, 6.0, 1.0]
    ]
    
    _, U, P, p_count = decomp(A)
    calculated_det = det(U, p_count)
    
    expected_det = np.linalg.det(A)
    np.testing.assert_allclose(calculated_det, expected_det, rtol=1e-7, atol=1e-7)


def test_inverse_matrix():
    A = [
        [1.0, 2.0, 3.0],
        [0.0, 1.0, 4.0],
        [5.0, 6.0, 1.0]
    ]
    
    calculated_inv = inv(A)
    expected_inv = np.linalg.inv(A)
    
    np.testing.assert_allclose(calculated_inv, expected_inv, rtol=1e-7, atol=1e-7)


def test_matrix_with_zero_on_diagonal():
    # Матрица требует обязательной перестановки строк, так как A[0][0] == 0
    A = [
        [0.0, 1.0],
        [2.0, 3.0]
    ]
    b = [1.0, 5.0]
    
    L, U, P, p_count = decomp(A)
    b_perm = permut(b, P)
    x = solve(L, U, b_perm)
    
    expected_x = [1.0, 1.0]
    np.testing.assert_allclose(x, expected_x, rtol=1e-7, atol=1e-7)
    
    calculated_det = det(U, p_count)
    np.testing.assert_allclose(calculated_det, np.linalg.det(A), rtol=1e-7, atol=1e-7)
