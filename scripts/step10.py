import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

class poisson_2d():
    """
    Solving Poisson's equation in 2-D
    """
    def __init__(self):
        self.grid_length_x = 2
        self.grid_length_y = 1
        self.grid_points_x = 31
        self.grid_points_y = 31
        self.dx = self.grid_length_x / (self.grid_points_x - 1)
        self.dy = self.grid_length_y / (self.grid_points_y - 1)

        self.c = 1
        self.nt = 100

        self.x_points = np.linspace(0, self.grid_length_x, self.grid_points_x)
        self.y_points = np.linspace(0, self.grid_length_y, self.grid_points_y)

        # setting up initial conditions
        self.p = np.zeros((self.grid_points_x, self.grid_points_y))
        self.b = np.zeros((self.grid_points_x, self.grid_points_y))
        self.b[int(self.grid_points_y / 4), int(self.grid_points_x / 4)]  = 100
        self.b[int(3 * self.grid_points_y / 4), int(3 * self.grid_points_x / 4)] = -100

    def disc_2d(self):
        """
        Applying 2d discretisation; to solve the poisson equation
        """
        for i in range(0, self.nt):
            pd = self.p.copy()

            self.p[1: -1, 1: -1] = (((pd[1: -1, 2:] + pd[1: -1, :-2]) * self.dy**2 +
                            (pd[2:, 1: -1] + pd[:-2, 1: -1]) * self.dx**2 -
                            self.b[1: -1, 1: -1] * self.dx**2 * self.dy**2) /
                            (2 * (self.dx**2 + self.dy**2)))

            self.p[0, :] = 0
            self.p[self.grid_points_y-1, :] = 0
            self.p[:, 0] = 0
            self.p[:, self.grid_points_x-1] = 0

if __name__ == '__main__':
    x = poisson_2d()
    fig = plt.figure(figsize=(11, 7), dpi=100)
    ax = fig.gca(projection="3d")
    X, Y = np.meshgrid(x.x_points, x.y_points)
    surf = ax.plot_surface(X, Y, x.b[:], cmap=cm.viridis)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$u$")
    ax.text2D(0.35, 0.95, "2-D Poisson", transform=ax.transAxes)
    ax.view_init(azim=145)
    plt.show()

    x.disc_2d()
    fig = plt.figure(figsize=(11, 7), dpi=100)
    ax = fig.gca(projection="3d")
    X, Y = np.meshgrid(x.x_points, x.y_points)
    surf = ax.plot_surface(X, Y, x.p[:], cmap=cm.viridis)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$u$")
    ax.text2D(0.35, 0.95, "2-D Poisson", transform=ax.transAxes)
    ax.view_init(azim=145)
    plt.show()
