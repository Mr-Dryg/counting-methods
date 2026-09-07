import copy

def decomp(A: list[list[float]]) -> tuple[list[list[float]]]:
    n = len(A)
    L = [[0] * i + [1] + [0] * (n-i-1) for i in range(n)]

    U = copy.deepcopy(A)

    for j in range(n-1):
        for i in range(j+1, n):
            mu = U[i][j] / U[j][j]
            L[i][j] = mu

            # i-ая строка -= (i-1)-ая строка
            for k in range(j, n):
                U[i][k] -= mu * U[j][k]

    return L, U

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

if __name__ == "__main__":
    n = int(input("n = "))
    print("A:")
    L, U = decomp([list(map(int, input().split(" "))) for _ in range(n)])
    print("b:")
    b = list(map(int, input().split(" ")))
    x = solve(L, U, b)
    print("x:", x)
