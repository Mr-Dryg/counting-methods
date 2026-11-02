import matplotlib.pyplot as plt
import numpy as np

x1 = np.linspace(-1, 4, 400)
x2 = np.linspace(-1, 3, 400)
X1, X2 = np.meshgrid(x1, x2)

F1 = np.exp(X1 * X2) + X1 - 6
F2 = X1**2 - 6 * X2 - 1

plt.figure(figsize=(8, 6))

contour1 = plt.contour(X1, X2, F1, levels=[0], colors="blue")
contour2 = plt.contour(X1, X2, F2, levels=[0], colors="red")

plt.plot([], [], color="blue", label=r"$e^{x_1 x_2} + x_1 - 6 = 0$")
plt.plot([], [], color="red", label=r"$x_1^2 - 6x_2 - 1 = 0$")

plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.axhline(0, color="gray", linestyle="--", linewidth=0.5)
plt.axvline(0, color="gray", linestyle="--", linewidth=0.5)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()

plt.show()
