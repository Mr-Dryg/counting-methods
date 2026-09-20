from typing import Iterator

def iter_diagonals(A: list[list[float]]) -> Iterator[tuple[float]]:
    """Return: (a_i, b_i, c_i)"""
    yield (0, A[0][0], A[0][1])
    for i in range(1, len(A)-1):
        yield (A[i][i-1], A[i][i], A[i][i+1])
    i += 1
    yield (A[i][i-1], A[i][i], 0)

def calc_P_Q(A: list[list[float]], d: list[float]) -> tuple[list[float]]:
    p = []
    q = []

    diags = iter_diagonals(A)
    a, b, c = next(diags)
    p.append(-c / b)
    q.append(d[0] / b)

    for i in range(1, len(A)):
        a, b, c = next(diags)
        z = b + a * p[i-1]
        p.append(-c / z)
        q.append((d[i] - a * q[i-1]) / z)

    return p, q

def dtma(A: list[list[float]], d: list[float]) -> list[float]:
    p, q = calc_P_Q(A, d)
    n = len(A)
    x = [0] * n
    x[n-1] = q[n-1]
    for i in range(2, n+1):
        x[n-i] = p[n-i] * x[n-i+1] + q[n-i]
    return x

if __name__ == "__main__":
    n = int(input("n = "))
    print("A:")
    A = [list(map(float, input().split(" "))) for _ in range(n)]
    print("b:")
    b = list(map(float, input().split(" ")))
    x = dtma(A, b)
    print("x: ", x)
