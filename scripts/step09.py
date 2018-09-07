import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

class laplace_2d():
    """
    Solving Laplace (or steady-state heat) equation in 2-D
    """
    def __init__(self):
        self.grid_length_x = 2
        self.grid_length_y = 1
        self.grid_points_x = 31
        self.grid_points_y = 31
        self.dx = self.grid_length_x / (self.grid_points_x - 1)
        self.dy = self.grid_length_y / (self.grid_points_y - 1)

        self.c = 1

        self.x_points = np.linspace(0, self.grid_length_x, self.grid_points_x)
        self.y_points = np.linspace(0, self.grid_length_y, self.grid_points_y)

        # setting up initial conditions
        self.p = np.zeros((self.grid_points_x, self.grid_points_y))
        self.p[:, 0] = 0
        self.p[:, -1] = self.y_points
        self.p[0, :] = self.p[1, :]
        self.p[-1, :] = self.p[-2, :]

    def disc_2d(self, l1norm_target = 1e-4):
        """
        Applying 2d discretisation; to solve the Laplace equation;
        Laplace has no time dependency; so will have to find the end steady state
        """
        l1norm = 1
        pn = np.empty_like(self.p)
        while l1norm > l1norm_target:
            pn = self.p.copy()

            self.p[1: -1, 1: -1] = ((self.dy ** 2 * (pn[1: -1, 2:] + pn[1: -1, 0: -2]) +
                                    self.dx ** 2 * (pn[2:, 1: -1] + pn[0: -2, 1: -1])) /
                                    (2 * (self.dx ** 2 + self.dy **2)))

            # boundary conditions
            self.p[:, 0] = 0  # p = 0 at x = 0
            self.p[:, -1] = self.y_points  # p = y at x = 2
            self.p[0, :] = self.p[1, :]  # dp/dy = 0 at y = 0
            self.p[-1, :] = self.p[-2, :]  # dp/dy = 0 at y = 1

            l1norm = (np.sum(np.abs(self.p[:]) - np.abs(pn[:])) / np.sum(np.abs(pn[:])))

        # self.p should be good now



if __name__ == '__main__':
    x = laplace_2d()
    fig = plt.figure(figsize=(11, 7), dpi=100)
    ax = fig.gca(projection="3d")
    X, Y = np.meshgrid(x.x_points, x.y_points)
    surf = ax.plot_surface(X, Y, x.p[:], cmap=cm.viridis)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$u$")
    ax.text2D(0.35, 0.95, "2-D Laplace", transform=ax.transAxes)
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
    ax.text2D(0.35, 0.95, "2-D Laplace", transform=ax.transAxes)
    ax.view_init(azim=145)
    plt.show()
