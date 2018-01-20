from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def plot_bands(e, size=75):
    plt.title('band 1 | band_2')
    plt.imshow(np.c_[e.band_1.reshape(size, size), e.band_2.reshape(size, size)])
    
    
def plot_bands_3d(e, size=75):
    x, y = np.meshgrid(range(size), range(size))
    f = plt.figure()

    ax = f.add_subplot(121, projection='3d')
    ax.set_title('band 1')
    ax.plot_surface(x, y, e.band_1.reshape(size, size)[(x, y)], cmap='coolwarm')
    ax.view_init(30, 30)

    ax = f.add_subplot(122, projection='3d')
    ax.set_title('band 2')
    ax.plot_surface(x, y, e.band_2.reshape(size, size)[(x, y)], cmap='coolwarm')
    ax.view_init(30, 30)