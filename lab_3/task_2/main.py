import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.dirname(parent_dir))
from lab_1.task_1 import lu_solution

if __name__ == "__main__":
    # X = [0.0, 1.0, 2.0, 3.0, 4.0]
    # f = [0.0, 1.8415, 2.9093, 3.1411, 3.2432]
    # dest_X = 1.5

    X = [-1.2, -0.7, -0.2, 0.3, 0.8]
    f = [0.43372, 0.24333, 0.32749e-01, 0.12149, 1.4243]
    dest_X = -0.5

    h = [None] + [X[i] - X[i-1] for i in range(1, len(X))]

    n_eq = len(X) - 2
    A = np.zeros((n_eq, n_eq))
    b = []

    A[0][0] = 2 * (h[1] + h[2])
    A[0][1] = h[2]
    b.append(3 * ((f[2] - f[1]) / h[2] - (f[1] - f[0]) / h[1]))

    for i in range(1, n_eq-1):
        A[i][i-1] = h[i+1]
        A[i][i] = 2 * (h[i+1] + h[i+2])
        A[i][i+1] = h[i+2]
        b.append(3 * ((f[i+2] - f[i+1]) / h[i+2] - (f[i+1] - f[i]) / h[i+1]))

    A[-1][-2] = h[-2]
    A[-1][-1] = 2 * (h[-2] + h[-1])
    b.append(3 * ((f[-1] - f[-2]) / h[-1] - (f[-2] - f[-3]) / h[-2]))

    b = np.array(b)

    c = [0.0] * len(X)
    c[2:] = lu_solution(A, b)

    a = [None] + [f[i-1] for i in range(1, len(X))]
    b = [None] + [(f[i] - f[i-1]) / h[i] - 1/3 * h[i] * (c[i+1] + 2*c[i]) for i in range(1, len(X) - 1)]
    d = [None] + [(c[i+1] - c[i]) / (3 * h[i]) for i in range(1, len(X) - 1)]

    b.append((f[-1] - f[-2]) / h[-1] - 2/3 * h[-1] * c[-1])
    d.append(-c[-1] / (3 * h[-1]))

    f_x = None
    for i in range(1, len(X)):
        if X[i-1] <= dest_X <= X[i]:
            dx = dest_X - X[i-1]
            f_x = a[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3
            dx_str = f"(x - {X[i-1]})"
            print(f"f(x) = {a[i]:.4f} + {b[i]:.4f}{dx_str} + {c[i]:.4f}{dx_str}^2 + {d[i]:.4f}{dx_str}^3, {X[i-1]} <= x <= {X[i]}, i = {i}")
            break

    print(f"f({dest_X}) = {f_x}")
