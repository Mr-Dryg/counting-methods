from typing import Callable
import numpy as np
from numpy.typing import NDArray
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

sys.path.append(os.path.dirname(parent_dir))

from lab_1.task_1 import lu_solution

def newton(f: Callable[..., NDArray], J: Callable[..., NDArray], x_0: NDArray, eps: float) -> list[tuple[int, NDArray, NDArray, float]]:
    iters = []
    x_k = x_prev = x_0
    k = 1
    while True:
        fx = f(*x_k)
        dx = lu_solution(J(*x_k), -fx)
        x_k, x_prev = x_k + dx, x_k
        eps_k = max(*[abs(x_k[i] - x_prev[i]) for i in range(x_k.shape[0])])
        iters.append((k, x_k, fx, eps_k))
        if eps_k <= eps:
            return iters
        k += 1

def simple(fi: Callable[..., NDArray], x_0: NDArray, q: float, eps: float) -> list[tuple[int, NDArray, NDArray, float]]:
    iters = []
    x_k = x_0
    k = 0
    while True:
        fi_x = fi(*x_k)
        eps_k = max(*[abs(fi_x[i] - x_k[i]) for i in range(x_k.shape[0])])
        iters.append((k, x_k, fi_x, eps_k))
        if eps_k <= eps:
            iters.append((k+1, fi_x, np.array([0.0, 0.0]), 0.0))
            return iters
        x_k = fi_x
        k += 1

if __name__ == "__main__":
    eps = 1e-4

    # f = lambda x1, x2: np.array([
    #     0.1*x1**2 + x1 + 0.2*x2**2 - 0.3,
    #     0.2*x1**2 + x2 - 0.1*x1*x2 - 0.7
    # ])
    # J = lambda x1, x2: np.array([
    #     [0.2*x1 + 1, 0.4*x2],
    #     [0.4*x1 - 0.1*x2, 1 - 0.1*x1]
    # ])
    # x_0 = np.array([0.25, 0.75])

    f = lambda x1, x2: np.array([
        np.exp(x1 * x2) + x1 - 6,
        x1**2 - 6 * x2 - 1
    ])
    J = lambda x1, x2: np.array([
        [x2 * np.exp(x1 * x2) + 1, x1 * np.exp(x1 * x2)],
        [2 * x1, -6]
    ])
    x_0 = np.array([2, 0.5])

    iters = newton(f, J, x_0, eps)
    print(f"\n{'k':4} | {'x_k':25} | {'f(x_k)':30} | {'eps_k':10}")    
    for k, x_k, fi_x, eps_k in iters:
        x_str = np.array2string(x_k, precision=6, separator=', ')
        f_str = np.array2string(fi_x, precision=6, separator=', ')
        print(f"{k:4} | {x_str:25} | {f_str:30} | {eps_k:10.4e}")

    # fi = lambda x1, x2: np.array([
    #     0.3 - 0.1*x1**2 - 0.2*x2**2,
    #     0.7 - 0.2*x1**2 + 0.1*x1*x2
    # ])
    # q = 0.5
    # x_0 = np.array([0.25, 0.75])

    print()

    fi = lambda x1, x2: np.array([
        np.sqrt(1 + 6 * x2),
        np.log(6 - x1) / x1
    ])
    x_0 = np.array([2.2, 0.6])

    def get_q_value(x1, x2):
        J_fi = np.array([
            [0, 3 / np.sqrt(1 + 6 * x2)],
            [-(x1 / (6 - x1) + np.log(6 - x1)) / x1**2, 0]
        ])

        # Обычная бесконечная норма (максимальная сумма модулей по строкам)
        return np.linalg.norm(J_fi, ord=np.inf)

    q = get_q_value(*x_0)
    print(f"Константа сходимости q в точке {x_0} равна: {q:.4f}")

    iters = simple(fi, x_0, q, eps)
    print(f"\n{'k':4} | {'x_k':25} | {'fi(x_k)':30} | {'eps_k':10}")
    for k, x_k, fi_x, eps_k in iters:
        x_str = np.array2string(x_k, precision=6, separator=', ')
        f_str = np.array2string(fi_x, precision=6, separator=', ')
        print(f"{k:4} | {x_str:25} | {f_str:30} | {eps_k:10.4e}")