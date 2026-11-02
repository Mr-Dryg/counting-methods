import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1, 4, 400)

f1 = lambda x: x*np.log10(x + 2)
f2 = lambda x: 1 - x**2

y1 = f1(x)
y2 = f2(x)

plt.figure(figsize=(8, 6))

plt.plot(x, y1, label=r"$y = x \log_10 {x + 2}$", color="blue", linewidth=2)
plt.plot(x, y2, label=r"$y = 1 - x^2$", color="red", linewidth=2)

plt.xlabel("$x$")
plt.ylabel("$y$")
plt.axhline(0, color="gray", linestyle="--", linewidth=0.5)
plt.axvline(0, color="gray", linestyle="--", linewidth=0.5)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()

plt.show()
