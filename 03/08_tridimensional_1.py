import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)

x, y = np.meshgrid(x,y)
z = np.sin(np.sqrt(x**2 + y ** 2))

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

surface = ax.plot_surface(x, y, z, cmap='viridis')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10)
plt.show()


