def rectangles(y, a, b, h):
    N = round((b - a) / h)
    return h * sum(y(a + (i - 0.5) * h) for i in range(1, N + 1))


def trapezoids(y, a, b, h):
    N = round((b - a) / h)
    X = [a + i * h for i in range(N + 1)]
    return h * (y(X[0]) / 2 + sum(y(x) for x in X[1:-1]) + y(X[-1]) / 2)


def simpson(y, a, b, h):
    N = round((b - a) / h)
    X = [a + i * h for i in range(N + 1)]
    return h / 3 * (y(X[0])
                    + sum((4 if i % 2 else 2) * y(X[i]) for i in range(1, N))
                    + y(X[-1]))


if __name__ == "__main__":
    y = lambda x: x / (3 * x + 4) ** 2
    a, b = -1.0, 1.0
    h1, h2 = 0.5, 0.25

    methods = {
        "прямоугольников": (rectangles, 2),
        "трапеций": (trapezoids, 2),
        "Симпсона": (simpson, 4),
    }

    k = h1 / h2

    for name, (method, p) in methods.items():
        F1 = method(y, a, b, h1)
        F2 = method(y, a, b, h2)

        # Рунге-Ромберг: F = F_h + (F_h - F_kh) / (k^p - 1), здесь h = h2, kh = h1
        err = (F2 - F1) / (k**p - 1)
        F_rr = F2 + err

        print(f"Метод {name}:")
        print(f"  F(h={h1}) = {F1:.4f}")
        print(f"  F(h={h2}) = {F2:.4f}")
        print(f"  оценка погрешности = {abs(err):.4f}")
        print(f"  уточнённое значение = {F_rr:.4f}")