import numpy as np
from numpy.typing import NDArray
from math import sin, cos, atan, pi, sqrt

def check_sim(mat: NDArray, eps: float) -> bool:
    if mat.shape[0] != mat.shape[1]:
        return False
    return np.allclose(mat, mat.T, atol=eps)

def rotation_method(
    A: NDArray[np.float64],
    eps: float
) -> tuple[list[NDArray[np.float64]], list[float], list[tuple[int, float]]]:
    if not check_sim(A, eps):
        return
    iters: list[tuple[int, float]] = []
    k = 0
    U = np.eye(A.shape[0])
    while True:
        k += 1
        a = 0
        p, q = 0, 0
        for i in range(A.shape[0]):
            for j in range(i + 1, A.shape[1]):
                if abs(A[i, j]) > a:
                    a = abs(A[i, j])
                    p, q = i, j
        d = A[p][p] - A[q][q]
        fi = atan(2 * A[p][q] / d) / 2 if d != 0 else pi / 4
        U_k = np.eye(A.shape[0])
        U_k[p][p] = U_k[q][q] = cos(fi)
        U_k[p][q] = -sin(fi)
        U_k[q][p] = sin(fi)
        U = U @ U_k
        A = U_k.T @ A @ U_k
        sum = 0
        for i in range(A.shape[0]):
            for j in range(i + 1, A.shape[1]):
                sum += A[i][j] * A[i][j]
        sum = sqrt(sum)
        iters.append((k, sum))
        if sum < eps:
            break
    return (
        [U[:, j] for j in range(A.shape[0])],
        [A[i, i] for i in range(A.shape[0])],
        iters
    )

if __name__ == "__main__":
    # n = int(input("n = "))
    # print("A:")
    # A = np.array([list(map(float, input().split(" "))) for _ in range(n)])
    # A = np.array([
    #     [-1, 3, 4],
    #     [3, 5, -1],
    #     [4, -1, -5]
    # ])
    A = np.array([
        [4, 2, 1],
        [2, 5, 3],
        [1, 3, 6]
    ])
    # eps = float(input("eps = "))
    eps = 0.3
    res = rotation_method(A, eps)
    print("U:")
    for u in res[0]:
        print(u)
    print("x:", res[1])
    print(f"{'k':4} | {'eps_i':6}")
    for i, eps_i in res[2]:
        print(f"{i:4} | {eps_i:6.4f}")