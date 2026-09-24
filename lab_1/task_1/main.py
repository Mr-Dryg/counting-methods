import copy

def get_I(n: int) -> list[list[float]]:
    return [[0] * i + [1] + [0] * (n-i-1) for i in range(n)]

def decomp(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]], list[int], int]:
    n = len(A)
    L = get_I(n)

    U = copy.deepcopy(A)
    P = list(range(n))
    p = 0

    for j in range(n-1):
        max_row = j
        max_val = abs(U[j][j])
        for i in range(j + 1, n):
            if abs(U[i][j]) > max_val:
                max_val = abs(U[i][j])
                max_row = i

        if max_row != j:
            U[j], U[max_row] = U[max_row], U[j]
            L[j][:j], L[max_row][:j] = L[max_row][:j], L[j][:j]
            P[j], P[max_row] = P[max_row], P[j]
            p += 1

        for i in range(j+1, n):
            mu = U[i][j] / U[j][j]
            L[i][j] = mu
            # i-ая строка -= (i-1)-ая строка
            for k in range(j, n):
                U[i][k] -= mu * U[j][k]

    return L, U, P, p

def solve(L: list[list[float]], U: list[list[float]], b: list[float]) -> list[float]:
    n = len(L)

    # Lz = b
    z = [0] * n
    z[0] = b[0]
    for i in range(1, n):
        z[i] = b[i] - sum([L[i][j] * z[j] for j in range(i)])

    # Ux = z
    x = [0] * n
    x[n-1] = z[n-1] / U[n-1][n-1]
    for i in range(n-2, -1, -1):
        x[i] = (z[i] - sum([U[i][j] * x[j] for j in range(i + 1, n)])) / U[i][i]

    return x

def permut(A: list[list[float]] | list[float], P: list[int]) -> list[list[float]] | list[float]:
    if len(A) > 0 and isinstance(A[0], list):
        return [copy.deepcopy(A[P[i]]) for i in range(len(P))]
    return [A[P[i]] for i in range(len(P))]

def det(U: list[list[float]], p: int) -> float:
    u = 1
    for i in range(len(U)):
        u *= U[i][i]
    return (-1)**p * u

def inv(A: list[list[float]]) -> list[list[float]]:
    n = len(A)
    L, U, P, _ = decomp(A)
    I = get_I(n)
    I = permut(I, P)
    X = [[0] * n for _ in range(n)]
    for j in range(n):
        b = [I[i][j] for i in range(n)]
        x = solve(L, U, b)
        for i in range(n):
            X[i][j] = x[i]
    return X

if __name__ == "__main__":
    # n = int(input("n = "))
    n = 4
    # print("A:")
    # A = [list(map(float, input().split(" "))) for _ in range(n)]
    A = [
        [-5, 2, 4, -4],
        [-4, 9, -7, -5],
        [0, -5, 6, 7, ],
        [4, 8, -6, -6]
    ]
    L, U, P, p = decomp(A)
    # print("b:")
    # b = list(map(float, input().split(" ")))
    b = [57, -23, 23, -20]
    x = solve(L, U, permut(b, P))
    print("x: ", x)
    print("det(A): ", det(U, p))
    print("A^-1:\n", '\n'.join([str(line) for line in inv(A)]))
