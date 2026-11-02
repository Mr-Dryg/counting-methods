from math import log10, log, e

def newton(f, df, x_0, eps) -> list[tuple[int, float, float, float]]:
    iters = []
    x_k = x_0
    k = 0
    while True:
        fx = f(x_k)
        d = - fx / df(x_k)
        iters.append((k, x_k, fx, d))
        x_k += d
        if abs(d) <= eps:
            iters.append((k+1, x_k, f(x_k), 0.0))
            return iters
        k += 1

def simple(fi, x_0, q, eps) -> list[tuple[int, float, float, float]]:
    iters = []
    x_k = x_0
    k = 0
    while True:
        fi_x = fi(x_k)
        eps_k = q / (1 - q) * abs(fi_x - x_k)
        iters.append((k, x_k, fi_x, eps_k))
        if eps_k <= eps:
            iters.append((k+1, fi_x, 0.0, 0.0))
            return iters
        x_k = fi_x
        k += 1

if __name__ == "__main__":
    eps = 1e-3

    # f = lambda x: e**(2*x) + 3*x - 4
    # df = lambda x: 2*e**(2*x) + 3
    # x_0 = 0.6

    f = lambda x: x*log10(x+2) + x**2 - 1
    ln10 = log(10)
    df = lambda x: log10(x+2) + x/(ln10*(x+2)) + 2*x
    x_0 = 1.0

    iters = newton(f, df, x_0, eps)
    print(f"{'k':4} | {'x_k':10} | {'f(x_k)':10} | {'- f(x_k) / f\'(x_k)'}")
    for k, x_k, fx, d in iters:
        print(f"{k:4} | {x_k:10.6} | {fx:10.4} | {d:6.4}")

    # fi = lambda x: log(4 - 3*x) / 2
    # q = 0.64
    # x_0 = 0.475

    fi = lambda x: (1 - x*log10(x+2))**0.5
    a, b = 0.0, 1.0

    dfi = lambda x: -(log10(x + 2) + x / ((x + 2) * log(10))) / (2 * fi(x))

    xs = a
    q = abs(dfi(xs))
    while xs <= b:
        xs += 1e-4
        q = max(q, abs(dfi(xs)))
    x_0 = 0.5

    print(f"\nq = {q}")

    iters = simple(fi, x_0, q, eps)
    print(f"\n{'k':4} | {'x_k':10} | {'fi(x_k)':10} | {'eps_k':10}")
    for k, x_k, fi_x, eps_k in iters:
        print(f"{k:4} | {x_k:10.6} | {fi_x:10.4} | {eps_k:10.4}")