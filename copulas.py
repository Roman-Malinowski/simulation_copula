import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X = np.arange(0, 1, 0.01)
Y = np.arange(0, 1, 0.01)
X, Y = np.meshgrid(X, Y)

# Creating Copulas
# Lukasiewicz Copula
C_L = np.maximum(np.zeros(shape=X.shape), X+Y-1) 

# Minimum Copula
C_min = np.minimum(X,Y)

# Product Copula
C_pi = np.multiply(X,Y)

# Plot the surface.

c_l = ax.plot_surface(X, Y, C_L, label="Lukasiewicz")
c_min = ax.plot_surface(X, Y, C_min, label="C_min")
c_pi = ax.plot_surface(X, Y, C_pi, label="Product")

c_l._facecolors2d = c_l._facecolor3d
c_l._edgecolors2d = c_l._edgecolor3d

c_min._facecolors2d = c_min._facecolor3d
c_min._edgecolors2d = c_min._edgecolor3d

c_pi._facecolors2d = c_pi._facecolor3d
c_pi._edgecolors2d = c_pi._edgecolor3d

# Customize the z axis.
ax.set_zlim(-0.1, 1.1)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

ax.set_xlabel("X")
ax.set_ylabel("Y")

ax.legend()

plt.show()

