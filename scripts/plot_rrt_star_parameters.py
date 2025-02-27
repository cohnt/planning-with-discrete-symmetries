import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits import mplot3d

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
theta = np.arange(0, 1/4, 0.01)
mu = np.arange(0, 1, 0.01)
theta, mu = np.meshgrid(theta, mu)

d = 6
epsilon = 0.1
c_star = 5
F = 10
B = 1

gamma = (2+theta) * ((((1 + epsilon / 4) * c_star * F) / ((d + 1) * theta * (1 - mu) * B)) ** (1 / (d + 1)))


# Plot the surface.
surf = ax.plot_surface(theta, mu, gamma, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
#ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
