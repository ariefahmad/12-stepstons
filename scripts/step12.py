import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

class channelflow_NS2d():
    """
    Solving a Navier Stokes equation (a channel flow problem)
    """
    def __init__(self, nt=10):
        self.nx = 41
        self.ny = 41
        self.nt = nt
        self.nit = 50
        self.c = 1
        self.dx = 2 / (self.nx -1)
        self.dy = 2 / (self.ny -1)
        self.dt = 1e-2

        self.x_points = np.linspace(0, 2, self.nx)
        self.y_points = np.linspace(0, 2, self.ny)

        self.rho = 1.
        self.nu = 0.1
        self.F = 1.

        self.u = np.zeros((self.nx, self.ny))
        self.v = np.zeros((self.nx, self.ny))
        self.p = np.zeros((self.nx, self.ny))
        self.b = np.zeros((self.nx, self.ny))

    def build_up_b(self):
        """
        Representing contents in square brackets of Pressure-Poisson equation;
        but will now have to account periodic boundary condition (i.e. at x = 0 & x = 2)
        """
        self.b[1:-1, 1:-1] = (self.rho * (1 / self.dt * ((self.u[1:-1, 2:] - self.u[1:-1, 0:-2]) / (2 * self.dx) +
                                          (self.v[2:, 1:-1] - self.v[0:-2, 1:-1]) / (2 * self.dy)) -
                                ((self.u[1:-1, 2:] - self.u[1:-1, 0:-2]) / (2 * self.dx))**2 -
                                2 * ((self.u[2:, 1:-1] - self.u[0:-2, 1:-1]) / (2 * self.dy) *
                                     (self.v[1:-1, 2:] - self.v[1:-1, 0:-2]) / (2 * self.dx))-
                                ((self.v[2:, 1:-1] - self.v[0:-2, 1:-1]) / (2 * self.dy))**2))

        # Periodic BC Pressure at x = 2
        self.b[1:-1, -1] = (self.rho * (1 / self.dt * ((self.u[1:-1, 0] - self.u[1:-1,-2]) / (2 * self.dx) +
                                        (self.v[2:, -1] - self.v[0:-2, -1]) / (2 * self.dy)) -
                              ((self.u[1:-1, 0] - self.u[1:-1, -2]) / (2 * self.dx))**2 -
                              2 * ((self.u[2:, -1] - self.u[0:-2, -1]) / (2 * self.dy) *
                                   (self.v[1:-1, 0] - self.v[1:-1, -2]) / (2 * self.dx)) -
                              ((self.v[2:, -1] - self.v[0:-2, -1]) / (2 * self.dy))**2))

        # Periodic BC Pressure at x = 0
        self.b[1:-1, 0] = (self.rho * (1 / self.dt * ((self.u[1:-1, 1] - self.u[1:-1, -1]) / (2 * self.dx) +
                                       (self.v[2:, 0] - self.v[0:-2, 0]) / (2 * self.dy)) -
                             ((self.u[1:-1, 1] - self.u[1:-1, -1]) / (2 * self.dx))**2 -
                             2 * ((self.u[2:, 0] - self.u[0:-2, 0]) / (2 * self.dy) *
                                  (self.v[1:-1, 1] - self.v[1:-1, -1]) / (2 * self.dx))-
                             ((self.v[2:, 0] - self.v[0:-2, 0]) / (2 * self.dy))**2))

    def pressure_poisson_periodic(self):
        """
        Sub iterating nit to ensure a divergence-free field; but accounting periodic boundary conditions
        """
        for q in range(0, self.nit):
            pn = self.p.copy()
            self.p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * self.dy**2 +
                              (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * self.dx**2) /
                             (2 * (self.dx**2 + self.dy**2)) -
                             self.dx**2 * self.dy**2 / (2 * (self.dx**2 + self.dy**2)) * self.b[1:-1, 1:-1])

            # Periodic BC Pressure at x = 2
            self.p[1:-1, -1] = (((pn[1:-1, 0] + pn[1:-1, -2])* self.dy**2 +
                            (pn[2:, -1] + pn[0:-2, -1]) * self.dx**2) /
                           (2 * (self.dx**2 + self.dy**2)) -
                           self.dx**2 * self.dy**2 / (2 * (self.dx**2 + self.dy**2)) * self.b[1:-1, -1])

            # Periodic BC Pressure at x = 0
            self.p[1:-1, 0] = (((pn[1:-1, 1] + pn[1:-1, -1])* self.dy**2 +
                           (pn[2:, 0] + pn[0:-2, 0]) * self.dx**2) /
                          (2 * (self.dx**2 + self.dy**2)) -
                          self.dx**2 * self.dy**2 / (2 * (self.dx**2 + self.dy**2)) * self.b[1:-1, 0])

            # Wall boundary conditions, pressure
            self.p[-1, :] =self.p[-2, :]  # dp/dy = 0 at y = 2
            self.p[0, :] = self.p[1, :]  # dp/dy = 0 at y = 0

    def channel_flow(self, acc=1e-3):
        du = 1
        while du > acc:
            un = self.u.copy()
            vn = self.v.copy()

            self.build_up_b()
            self.pressure_poisson_periodic()

            self.u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                             un[1:-1, 1:-1] * self.dt / self.dx *
                            (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                             vn[1:-1, 1:-1] * self.dt / self.dy *
                            (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                             self.dt / (2 * self.rho * self.dx) *
                            (self.p[1:-1, 2:] - self.p[1:-1, 0:-2]) +
                             self.nu * (self.dt / self.dx**2 *
                            (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                             self.dt / self.dy**2 *
                            (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])) +
                             self.F * self.dt)

            self.v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                             un[1:-1, 1:-1] * self.dt / self.dx *
                            (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                             vn[1:-1, 1:-1] * self.dt / self.dy *
                            (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                             self.dt / (2 * self.rho * self.dy) *
                            (self.p[2:, 1:-1] - self.p[0:-2, 1:-1]) +
                             self.nu * (self.dt / self.dx**2 *
                            (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                             self.dt / self.dy**2 *
                            (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

            # Periodic BC u at x = 2
            self.u[1:-1, -1] = (un[1:-1, -1] - un[1:-1, -1] * self.dt / self.dx *
                          (un[1:-1, -1] - un[1:-1, -2]) -
                           vn[1:-1, -1] * self.dt / self.dy *
                          (un[1:-1, -1] - un[0:-2, -1]) -
                           self.dt / (2 * self.rho * self.dx) *
                          (self.p[1:-1, 0] - self.p[1:-1, -2]) +
                           self.nu * (self.dt / self.dx**2 *
                          (un[1:-1, 0] - 2 * un[1:-1,-1] + un[1:-1, -2]) +
                           self.dt / self.dy**2 *
                          (un[2:, -1] - 2 * un[1:-1, -1] + un[0:-2, -1])) + self.F * self.dt)

            # Periodic BC u at x = 0
            self.u[1:-1, 0] = (un[1:-1, 0] - un[1:-1, 0] * self.dt / self.dx *
                         (un[1:-1, 0] - un[1:-1, -1]) -
                          vn[1:-1, 0] * self.dt / self.dy *
                         (un[1:-1, 0] - un[0:-2, 0]) -
                          self.dt / (2 * self.rho * self.dx) *
                         (self.p[1:-1, 1] - self.p[1:-1, -1]) +
                          self.nu * (self.dt / self.dx**2 *
                         (un[1:-1, 1] - 2 * un[1:-1, 0] + un[1:-1, -1]) +
                          self.dt / self.dy**2 *
                         (un[2:, 0] - 2 * un[1:-1, 0] + un[0:-2, 0])) + self.F * self.dt)

            # Periodic BC v at x = 2
            self.v[1:-1, -1] = (vn[1:-1, -1] - un[1:-1, -1] * self.dt / self.dx *
                          (vn[1:-1, -1] - vn[1:-1, -2]) -
                           vn[1:-1, -1] * self.dt / self.dy *
                          (vn[1:-1, -1] - vn[0:-2, -1]) -
                           self.dt / (2 * self.rho * self.dy) *
                          (self.p[2:, -1] - self.p[0:-2, -1]) +
                           self.nu * (self.dt / self.dx**2 *
                          (vn[1:-1, 0] - 2 * vn[1:-1, -1] + vn[1:-1, -2]) +
                           self.dt / self.dy**2 *
                          (vn[2:, -1] - 2 * vn[1:-1, -1] + vn[0:-2, -1])))

            # Periodic BC v at x = 0
            self.v[1:-1, 0] = (vn[1:-1, 0] - un[1:-1, 0] * self.dt / self.dx *
                         (vn[1:-1, 0] - vn[1:-1, -1]) -
                          vn[1:-1, 0] * self.dt / self.dy *
                         (vn[1:-1, 0] - vn[0:-2, 0]) -
                          self.dt / (2 * self.rho * self.dy) *
                         (self.p[2:, 0] - self.p[0:-2, 0]) +
                          self.nu * (self.dt / self.dx**2 *
                         (vn[1:-1, 1] - 2 * vn[1:-1, 0] + vn[1:-1, -1]) +
                          self.dt / self.dy**2 *
                         (vn[2:, 0] - 2 * vn[1:-1, 0] + vn[0:-2, 0])))

            # Wall BC: u,v = 0 at y = 0,2
            self.u[0, :] = 0
            self.u[-1, :] = 0
            self.v[0, :] = 0
            self.v[-1, :]=0

            du = (np.sum(self.u) - np.sum(un)) / np.sum(self.u)
        # self.u & self.v should be good now..

if __name__ == '__main__':
    x = channelflow_NS2d()
    fig = plt.figure(figsize=(11, 7), dpi=100)
    X, Y = np.meshgrid(x.x_points, x.y_points)
    plt.quiver(X[::2, ::2], Y[::2, ::2], x.u[::2, ::2], x.v[::2, ::2])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Channel Flow")
    plt.show()

    x.channel_flow()
    fig = plt.figure(figsize=(11, 7), dpi=100)
    X, Y = np.meshgrid(x.x_points, x.y_points)
    plt.quiver(X[::2, ::2], Y[::2, ::2], x.u[::2, ::2], x.v[::2, ::2])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Channel Flow")
    plt.show()
