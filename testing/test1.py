import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import numpy as np
import pandas as pd


def f(x, y):
    # return np.sin(np.sqrt(x ** 2 + y ** 2))
    return -np.log(np.exp(x)/(np.exp(x)+np.exp(y)))

 
    # dx = np.exp(x)/(np.exp(x)+np.exp(y))-1
    # dy = np.exp(x)/(np.exp(x)+np.exp(y))

    # return np.array((-dx, -dy))
    # return dy

x = np.linspace(0, 100, 100)
y = np.linspace(0, 100, 100)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.contour3D(X, Y, Z, 50, cmap='binary')





ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.set_title('surface')

plt.show()