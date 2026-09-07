from main import decomp, solve
import numpy as np

def test_basic_decomp():
    A = [
        [10, 1, 1],
        [2, 10, 1],
        [2, 2, 10]
    ]
    
    L, U = decomp(A)
    
    expected_L = [
        [1, 0, 0],
        [0.2, 1, 0],
        [0.2, 0.18, 1]
    ]
    expected_U = [
        [10, 1, 1],
        [0, 9.8, 0.8],
        [0,  0,  9.65]
    ]
    
    np.testing.assert_allclose(L, expected_L, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(U, expected_U, rtol=1e-2, atol=1e-2)


def test_decomp_reconstruction_property():
    A = [
        [4.0, 3.0],
        [6.0, 3.0]
    ]
    
    L, U = decomp(A)
    A_reconstructed = np.dot(L, U)
    
    np.testing.assert_allclose(A_reconstructed, A, rtol=1e-7, atol=1e-7)

def test_basic_solve():
    A = [
        [10, 1, 1],
        [2, 10, 1],
        [2, 2, 10]
    ]
    b = [12, 13, 14]
    
    L, U = decomp(A)
    x = solve(L, U, b)
    
    expected_x = [1, 1, 1]
    
    np.testing.assert_allclose(x, expected_x, rtol=1e-7, atol=1e-7)