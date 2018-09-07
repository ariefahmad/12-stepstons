import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

class cavityflow_NS2d():
    """
    Solving a Navier Stokes equation (a cavity flow problem)
    """
    def __init__(self, nt=500):
        self.nx = 41
        self.ny = 41
        self.nt = nt
        self.nit = 50
        self.c = 1
        self.dx = 2 / (self.nx -1)
        self.dy = 2 / (self.ny -1)
        self.dt = 1e-3

        self.x_points = np.linspace(0, 2, self.nx)
        self.y_points = np.linspace(0, 2, self.ny)

        self.rho = 1.
        self.nu = 0.1

        self.u = np.zeros((self.nx, self.ny))
        self.v = np.zeros((self.nx, self.ny))
        self.p = np.zeros((self.nx, self.ny))
        self.b = np.zeros((self.nx, self.ny))

    def build_up_b(self):
        """
        Representing contents in square brackets of Pressure-Poisson equation
        """
        self.b[1:-1, 1:-1] = (self.rho * (1 / self.dt *
                        ((self.u[1:-1, 2:] - self.u[1:-1, 0:-2]) /
                         (2 * self.dx) + (self.v[2:, 1:-1] - self.v[0:-2, 1:-1]) / (2 * self.dy)) -
                        ((self.u[1:-1, 2:] - self.u[1:-1, 0:-2]) / (2 * self.dx))**2 -
                          2 * ((self.u[2:, 1:-1] - self.u[0:-2, 1:-1]) / (2 * self.dy) *
                               (self.v[1:-1, 2:] - self.v[1:-1, 0:-2]) / (2 * self.dx))-
                              ((self.v[2:, 1:-1] - self.v[0:-2, 1:-1]) / (2 * self.dy))**2))

    def pressure_poisson(self):
        """
        Sub iterating nit to ensure a divergence-free field
        """
        for q in range(0, self.nit):
            pn = self.p.copy()
            self.p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * self.dy**2 +
                              (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * self.dx**2) /
                              (2 * (self.dx**2 + self.dy**2)) -
                              self.dx**2 * self.dy**2 / (2 * (self.dx**2 + self.dy**2)) *
                              self.b[1:-1,1:-1])

            self.p[:, -1] = self.p[:, -2]  # dp/dy = 0 at x = 2
            self.p[0, :] = self.p[1, :]   # dp/dy = 0 at y = 0
            self.p[:, 0] = self.p[:, 1]  # dp/dx = 0 at x = 0
            self.p[-1, :] = 0  # p = 0 at y = 2

    def cavity_flow(self):
        for n in range(0, self.nt):
            un = self.u.copy()
            vn = self.v.copy()

            self.build_up_b()
            self.pressure_poisson()

            self.u[1:-1, 1:-1] = (un[1:-1, 1:-1]-
                             un[1:-1, 1:-1] * self.dt / self.dx *
                            (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                             self.v[1:-1, 1:-1] * self.dt / self.dy *
                            (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                             self.dt / (2 * self.rho * self.dx) * (self.p[1:-1, 2:] - self.p[1:-1, 0:-2]) +
                             self.nu * (self.dt / self.dx**2 *
                            (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                             self.dt / self.dy**2 *
                            (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

            self.v[1:-1,1:-1] = (vn[1:-1, 1:-1] -
                            un[1:-1, 1:-1] * self.dt / self.dx *
                           (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                            vn[1:-1, 1:-1] * self.dt / self.dy *
                           (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                            self.dt / (2 * self.rho * self.dy) * (self.p[2:, 1:-1] - self.p[0:-2, 1:-1]) +
                            self.nu * (self.dt / self.dx**2 *
                           (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                            self.dt / self.dy**2 *
                           (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

            self.u[0, :] = 0
            self.u[:, 0] = 0
            self.u[:, -1] = 0
            self.u[-1, :] = 1  # set velocity on cavity lid equal to 1
            self.v[0, :] = 0
            self.v[-1, :] = 0
            self.v[:, 0] = 0
            self.v[:, -1] = 0

if __name__ == '__main__':
    x = cavityflow_NS2d()
    fig = plt.figure(figsize=(11, 7), dpi=100)
    X, Y = np.meshgrid(x.x_points, x.y_points)
    plt.contourf(X, Y, x.p, alpha=0.5, cmap=cm.viridis)
    plt.colorbar()
    plt.contour(X, Y, x.p, cmap=cm.viridis)
    # plt.quiver(X[::2, ::2], Y[::2, ::2], x.u[::2, ::2], x.v[::2, ::2])
    plt.streamplot(X, Y, x.u, x.v)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Lid Cavity Flow")
    plt.show()

    x.cavity_flow()
    fig = plt.figure(figsize=(11, 7), dpi=100)
    X, Y = np.meshgrid(x.x_points, x.y_points)
    plt.contourf(X, Y, x.p, alpha=0.5, cmap=cm.viridis)
    plt.colorbar()
    plt.contour(X, Y, x.p, cmap=cm.viridis)
    # plt.quiver(X[::2, ::2], Y[::2, ::2], x.u[::2, ::2], x.v[::2, ::2])
    plt.streamplot(X, Y, x.u, x.v)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Lid Cavity Flow")
    plt.show()
