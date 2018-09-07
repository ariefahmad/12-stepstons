import numpy as np
import matplotlib.pyplot as plt

class lin_1dconv():
    """
    1-D linear Convection
    """
    def __init__(self):
        self.grid_length = 10
        self.grid_points = 41
        self.dx = self.grid_length / (self.grid_points - 1)  # spacing between grids
        self.nt = 400  # number of timesteps
        self.dt = 0.025  # time intervals
        self.c = 1  # wave speed

        self.u = np.ones(self.grid_points)  # applying initial conditions
        self.u[int(.5 / self.dx): int(1 / self.dx +1)] = 2

    def disc_1d(self):
        """
        Applying 1d discretisation; for linear convection
        """
        un = np.ones(self.grid_points)
        for n in range(0, self.nt):
            un = self.u.copy()
            for i in range(1, self.grid_points):
                self.u[i] = un[i] - self.c * (self.dt / self.dx) * (un[i] - un[i-1])

if __name__ == '__main__':
    x = lin_1dconv()
    plt.plot(np.linspace(0, x.grid_length, x.grid_points), x.u, label="Initial")
    x.disc_1d()
    plt.plot(np.linspace(0, x.grid_length, x.grid_points), x.u, label="After")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend()
    plt.tick_params(direction="in", top=True, right=True)
    plt.title("1-D Linear Convection")
    plt.show()
