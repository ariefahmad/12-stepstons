import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

class nonlin_2dconv():
    """
    2-D Non-Linear Convection
    """
    def __init__(self):
        self.grid_length_x = 2
        self.grid_length_y = 2
        self.grid_points_x = 101
        self.grid_points_y = 101
        self.dx = self.grid_length_x / (self.grid_points_x - 1)
        self.dy = self.grid_length_y / (self.grid_points_y - 1)

        self.nt = 80
        self.c = 1
        self.nCFL = 0.2
        self.dt = self.nCFL * self.dx

        self.x_points = np.linspace(0, self.grid_length_x, self.grid_points_x)
        self.y_points = np.linspace(0, self.grid_length_y, self.grid_points_y)

        # setting up initial conditions
        self.u = np.ones((self.grid_points_x, self.grid_points_y))
        self.v = np.ones((self.grid_points_x, self.grid_points_y))
        self.u[int(0.5 / self.dy): int(1 / self.dy + 1), int(0.5 / self.dx): int(1 / self.dx + 1)] = 2
        self.v[int(0.5 / self.dy): int(1 / self.dy + 1), int(0.5 / self.dx): int(1 / self.dx + 1)] = 2

    def disc_2d(self):
        """
        Applying 2d discretisation; for non-linear convection;
        use vector processes instead of nested for loops when dealing with 2d
        """
        for n in range(0, self.nt + 1):
            un = self.u.copy()
            vn = self.v.copy()

            self.u[1:, 1:] = (un[1:, 1:] - (un[1:, 1:] * self.c * (self.dt / self.dx) * (un[1:, 1:] - un[1:, :-1])) - (vn[1:, 1:] * self.c * (self.dt / self.dx) * (un[1:, 1:] - un[:-1, 1:])))
            self.v[1:, 1:] = (vn[1:, 1:] - (un[1:, 1:] * self.c * (self.dt / self.dx) * (vn[1:, 1:] - vn[1:, :-1])) - (vn[1:, 1:] * self.c * (self.dt / self.dx) * (vn[1:, 1:] - vn[:-1, 1:])))

            self.u[0, :], self.u[-1, :], self.u[:, 0], self.u[:, -1] = [1,1,1,1]  # boundary conditions
            self.v[0, :], self.v[-1, :], self.v[:, 0], self.v[:, -1] = [1,1,1,1]


if __name__ == '__main__':
    x = nonlin_2dconv()
    fig = plt.figure(figsize=(11, 7), dpi=100)
    ax = fig.gca(projection="3d")
    X, Y = np.meshgrid(x.x_points, x.y_points)
    surf = ax.plot_surface(X, Y, x.u[:], cmap=cm.viridis)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$u$")
    ax.text2D(0.35, 0.95, "2-D Non-Linear Convection", transform=ax.transAxes)
    plt.show()

    x.disc_2d()
    fig = plt.figure(figsize=(11, 7), dpi=100)
    ax = fig.gca(projection="3d")
    X, Y = np.meshgrid(x.x_points, x.y_points)
    surf = ax.plot_surface(X, Y, x.u[:], cmap=cm.viridis)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$u$")
    ax.text2D(0.35, 0.95, "2-D Non-Linear Convection", transform=ax.transAxes)
    plt.show()
