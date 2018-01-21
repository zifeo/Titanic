from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


def plot_bands(e, size=75):
    plt.title('band 1 | band_2')
    plt.imshow(np.c_[e.band_1.reshape(size, size), e.band_2.reshape(size, size)])

    
def plot_bands_3d(e, size=75, angle=30):
    x, y = np.meshgrid(range(size), range(size))
    b1 = e.band_1.reshape(size, size)[(x, y)]
    b2 = e.band_2.reshape(size, size)[(x, y)]

    f = plt.figure()

    ax = f.add_subplot(131, projection='3d')
    ax.set_title('band 1')
    ax.plot_surface(x, y, b1, cmap=cm.coolwarm)
    ax.view_init(30, angle)

    ax = f.add_subplot(132, projection='3d')
    ax.set_title('band 2')
    ax.plot_surface(x, y, b2, cmap=cm.coolwarm)
    ax.view_init(30, angle)
    
    ax = f.add_subplot(133, projection='3d')
    ax.set_title('band average')
    ax.plot_surface(x, y, (b1 + b2) / 2, cmap=cm.coolwarm)
    ax.view_init(30, angle)
    