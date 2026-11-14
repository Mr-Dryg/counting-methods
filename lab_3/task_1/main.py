import numpy as np

# Многочлен Лагранжа
def lagrang(X, y, dest_X):
    # y = lambda x: np.log(x)
    # X = [0.1, 0.5, 0.9, 1.3]
    # dest_X = 0.8

    w = lambda x, i: np.prod([x - X[j] for j in range(len(X)) if j != i])
    dw = lambda i: np.prod([X[i] - X[j] for j in range(len(X)) if j != i])

    L = lambda x: np.sum([y(X[i]) * w(x, i) / dw(i) for i in range(len(X))])

    print(f"L(x) = ", end="")
    for i in range(len(X)):
        print(f"{y(X[i]) / dw(i):.4f}{''.join(f"(x - {X[j]})" for j in range(len(X)) if j != i)}{" + " if i < len(X) - 1 else ""}", end="")
    print()

    L_X = L(dest_X)
    y_X = y(dest_X)
    print(f"L({dest_X}) = {L_X}")
    print(f"y({dest_X}) = {y_X}")
    print(f"Δ(L({dest_X})) = {abs(L_X - y_X)}")

# Многочлен Ньютона
def newton(X, y, dest_X):
    # y = lambda x: np.sin(np.pi / 6 * x)
    # X = [0, 1.0, 2.0, 3.0]
    # dest_X = 1.5

    mem = {}
    def f(*args):
        if args in mem:
            return mem[args]
        if len(args) == 1:
            mem[args] = y(X[args[0]])
            return mem[args]
        else:
            result = (f(*args[:-1]) - f(*args[1:])) / (X[args[0]] - X[args[-1]])
            mem[args] = result
            return result

    P = lambda x: np.sum([f(*range(i+1)) * np.prod([x - X[j] for j in range(i)]) for i in range(len(X))])

    print("P(x) = ", end="")
    for i in range(len(X)):
        print(f"{f(*range(i+1)):.4f}{''.join(f"(x - {X[j]})" for j in range(i) if j != i)}{" + " if i < len(X) - 1 else ""}", end="")
    print()

    P_X = P(dest_X)
    y_X = y(dest_X)
    print(f"P({dest_X}) = {P_X}")
    print(f"y({dest_X}) = {y_X}")
    print(f"Δ(P({dest_X})) = {abs(P_X - y_X)}")

if __name__ == "__main__":
    y = lambda x: x**2 * np.e**x
    X1 = [-1.2, -0.7, -0.2, 0.3]
    X2 = [-1.2, -0.7, 0, 0.3]
    dest_X = -0.5

    print("a)")
    lagrang(X1, y, dest_X)
    newton(X1, y, dest_X)

    print("b)")
    lagrang(X2, y, dest_X)
    newton(X2, y, dest_X)