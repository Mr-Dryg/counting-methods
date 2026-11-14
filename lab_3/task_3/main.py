import matplotlib.pyplot as plt
import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.dirname(parent_dir))
from lab_1.task_1 import lu_solution

def F(X, Y, n):
    n += 1
    A = np.zeros((n, n))
    last_X = np.ones(len(X))
    for j in range(n):
        A[0][j] = sum(last_X)
        last_X *= X
    for k in range(1, n):
        A[k][:-1] = A[k-1][1:]
        A[k][-1] = sum(last_X)
        last_X *= X

    b = np.zeros(n)
    last_X = np.ones(len(X))
    for k in range(n):
        b[k] = Y @ last_X
        last_X *= X

    a = lu_solution(A, b)

    Fn = lambda x: sum(a[i] * x**i for i in range(n))

    print(f"F_{n-1}(x) = {a[0]:.4f} + {a[1]:.4f}x{''.join(f" + {a[i]:.4f}x^{i}" for i in range(2, n))}")

    print(f"Ф = {sum((Fn(x) - y)**2 for x, y in zip(X, Y))}")

    return Fn

if __name__ == "__main__":
    # X = np.array([0.0, 1.7, 3.4, 5.1, 6.8, 8.5])
    # Y = np.array([0.0, 1.3038, 1.8439, 2.2583, 2.6077, 2.9155])

    X = [-1.7, -1.2, -0.7, -0.2, 0.3, 0.8]
    Y = [0.52796, 0.43372, 0.24333, 0.03275, 0.12149, 1.4243]

    F1 = F(X, Y, 1)
    F2 = F(X, Y, 2)

    x_plot = np.linspace(X[0], X[-1], 200)

    plt.figure(figsize=(8, 5))
    plt.scatter(X, Y, color="black", zorder=3, label="Табличные данные")
    plt.plot(x_plot, F1(x_plot), label="Многочлен 1-й степени")
    plt.plot(x_plot, F2(x_plot), label="Многочлен 2-й степени")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.show()