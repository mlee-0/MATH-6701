import numpy as np
import matplotlib.pyplot as plt


# Plot an example loss curve.
x = np.linspace(-2, 2, 100)
y = x ** 2
x_ = 0.9
slope = 2 * x_
x_slope = np.array([x_-0.5, x_+0.5])
y_slope = (x_ ** 2) + np.array([-1, 1]) * (slope * 0.5)
plt.figure()
plt.plot(x, y, color=(0.0,)*3)
plt.vlines(x_, 0, x_**2, color=(0.5,)*3, linestyles='dotted')
plt.plot(x_slope, y_slope)
plt.text(0.9+0.1, 0, 'x = 0.9', color=(0.5,)*3)
plt.xlabel('x')
plt.ylabel('Loss')
plt.margins(0)
plt.show()