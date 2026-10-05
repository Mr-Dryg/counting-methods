import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from math import log10, ceil

import task_1

def calc_betta_alpha(A: list[list[float]], b: list[float]) -> tuple[list[float], list[list[float]]]:
    betta = []
    alpha: list[list[float]] = []
    for i in range(len(A)):
        betta.append(b[i] / A[i][i])
        alpha.append([])
        for j in range(len(A)):
            alpha[i].append(-A[i][j] / A[i][i] if i != j else 0)
    return betta, alpha

def norm(a: list) -> float:
    if isinstance(a[0], list):
        return max(sum(abs(x) for x in row) for row in a)
    else:
        return max(abs(x) for x in a)

def dot(alpha: list[list[float]], x: list) -> list:
    if isinstance(x[0], list):
        res = []
        for i in range(len(alpha)):
            res.append([sum(alpha[i][k] * x[k][j] for k in range(len(x))) for j in range(len(x[0]))])
        return res
    else:
        return [sum(a_i * x_i for a_i, x_i in zip(row, x)) for row in alpha]

def simple_iter(A: list[list[float]], b: list[float], eps: float) -> list[tuple[int, list[float], float]]:
    """Return: [(i, x, eps_i)]"""
    n = len(A)
    betta, alpha = calc_betta_alpha(A, b)
    norm_alpha = norm(alpha)
    k = (log10(eps) + log10(1 - norm_alpha) - log10(norm(betta))) / log10(norm_alpha)
    print("K = ", ceil(k))
    kf = norm_alpha / (1 - norm_alpha) if norm_alpha < 1 else 1
    k = 0
    x_prev = []
    x = betta
    res = []
    while True:
        k += 1
        x_prev = x
        x = [bet + ax for bet, ax in zip(betta, dot(alpha, x))]
        eps_i = kf * norm([x[i] - x_prev[i] for i in range(n)])
        res.append((k, x, eps_i))
        if eps_i <= eps:
            return res

def calc_B_C(alpha: list[list[float]]) -> tuple[list[list[float]]]:
    B = [[0] * len(alpha) for _ in range(len(alpha))]
    C = [[0] * len(alpha) for _ in range(len(alpha))]
    for i in range(len(alpha)):
        for j in range(len(alpha)):
            if i > j:
                B[i][j] = alpha[i][j]
            else:
                C[i][j] = alpha[i][j]
    return B, C

def seidel(A: list[list[float]], b: list[float], eps: float) -> list[tuple[int, list[float], float]]:
    """Return: [(i, x, eps_i)]"""
    n = len(A)
    betta, alpha = calc_betta_alpha(A, b)
    B, C = calc_B_C(alpha)
    I_minus_B_inv = task_1.inv([[ii - bi for ii, bi in zip(I_row, B_row)] for I_row, B_row in zip(task_1.get_I(n), B)])
    betta = dot(I_minus_B_inv, betta)
    alpha = dot(I_minus_B_inv, C)
    norm_alpha = norm(alpha)
    k = (log10(eps) + log10(1 - norm_alpha) - log10(norm(betta))) / log10(norm(C))
    print("K (Зейдель) = ", ceil(k))
    kf = norm(C) / (1 - norm_alpha) if norm_alpha < 1 else 1
    k = 0
    x_prev = []
    x = betta
    res = []
    while True:
        k += 1
        x_prev = x
        x = [bet + ax for bet, ax in zip(betta, dot(alpha, x))]
        eps_i = kf * norm([x[i] - x_prev[i] for i in range(n)])
        res.append((k, x, eps_i))
        if eps_i <= eps:
            return res

if __name__ == "__main__":
    # n = int(input("n = "))
    n = 3
    # print("A:")
    # A = [list(map(float, input().split(" "))) for _ in range(n)]
    A = [
        [10, 1, 1],
        [2, 10, 1],
        [2, 2, 10]
    ]
    # print("b:")
    # b = list(map(float, input().split(" ")))
    b = [12, 13, 14]
    # eps = float(input("eps = "))
    eps = 0.01

    print("simple iter:")
    iters = simple_iter(A, b, eps)
    print(f"{'k':4} | {'x':{8*n-2}} | {'eps_i':6}")
    for i, x_i, eps_i in iters:
        x_str = ", ".join([f"{num:.4f}" for num in x_i])
        print(f"{i:4} | {x_str:{8*n-2}} | {eps_i:6.4f}")

    print("seidel:")
    iters = seidel(A, b, eps)
    print(f"{'k':4} | {'x':{8*n-2}} | {'eps_i':6}")
    for i, x_i, eps_i in iters:
        x_str = ", ".join([f"{num:.4f}" for num in x_i])
        print(f"{i:4} | {x_str:{8*n-2}} | {eps_i:6.4f}")
