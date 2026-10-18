import numpy as np
from numpy.typing import NDArray

def sign(x: float) -> int:
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

def QR(A: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    H = np.eye(A.shape[0])
    for k in range(A.shape[0] - 1):
        v = A[:, k:k+1].copy()
        v[:k] = 0
        v[k] += sign(v[k]) * np.linalg.norm(v[k:])
        d = v.T @ v
        H_k = np.eye(A.shape[0])
        H_k -= 2 * v @ v.T / d if d != 0 else 0
        H = H_k @ H
        A = H_k @ A
    return H.T, A

def eigenvalue_k(A: NDArray[np.float64], eps: float) -> NDArray[np.complex128]:
    n = A.shape[0]
    j = 0
    res = []
    while j < n:
        if j == n - 1 or abs(A[j+1, j]) <= eps:
            res.append(A[j, j])
            j += 1
        else:
            block = A[j:j+2, j:j+2]
            tr = np.trace(block)
            det = np.linalg.det(block)
            D = tr ** 2 - 4 * det
            if D < 0:
                res.append((tr + 1j * np.sqrt(-D)) / 2)
                res.append((tr - 1j * np.sqrt(-D)) / 2)
            else:
                res.append((tr + np.sqrt(D)) / 2)
                res.append((tr - np.sqrt(D)) / 2)
            j += 2
    return np.array(res)

def eigenvalue(A_k: NDArray[np.float64], eps: float) -> tuple[NDArray[np.complex128], list[int, float, float]]:
    k = 0
    iters = []
    res_k = eigenvalue_k(A_k, eps)
    while True:
        k += 1
        Q, R = QR(A_k)
        A_k_next = R @ Q
        res_k_next = eigenvalue_k(A_k_next, eps)
        # eps_1 = all([np.sqrt(np.sum(A_k_next[m+1:, m] ** 2)) <= eps for m in range(A.shape[0])])
        eps_1 = np.linalg.norm(A_k_next - A_k)
        eps_2 = np.linalg.norm(res_k_next - res_k)
        iters.append((k, eps_1, eps_2))
        A_k = A_k_next
        res_k = res_k_next
        if (eps_1 <= eps or eps_2 <= eps):
            break
    return res_k, iters

if __name__ == "__main__":
    A = np.array([
        [1, 3, 1],
        [1, 1, 4],
        [4, 3, 1]
    ])
    eps = 0.01
    res, iters = eigenvalue(A, eps)
    print("lambda: ", ", ".join(res.astype(str)))
    print(f"{'k':4} | {'eps_1':6} | {'eps_2':6}")
    for i, eps_1, eps_2 in iters:
        print(f"{i:4} | {eps_1:6.4f} | {eps_2:6.4f}")
