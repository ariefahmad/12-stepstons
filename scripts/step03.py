import numpy as np
import matplotlib.pyplot as plt

class diff_1d():
    """
    1-D diffusion
    """
    def __init__(self):
        self.grid_length = 10
        self.grid_points = 41
        self.dx = self.grid_length / (self.grid_points - 1)  # spacing between grids
        self.nt = 400  # number of timesteps

        self.nu = 0.3  # new viscosity
        self.nCFL = 0.2  # CFL number
        self.dt = self.nCFL * (self.dx ** 2) / self.nu  # dynamic timescales, since dealing with CFL conditions now

        self.u = np.ones(self.grid_points)  # applying initial conditions
        self.u[int(.5 / self.dx): int(1 / self.dx +1)] = 2

    def disc_1d(self):
        """
        Applying 1d discretisation; for diffusion
        """
        un = np.ones(self.grid_points)
        for n in range(0, self.nt):
            un = self.u.copy()
            for i in range(1, self.grid_points - 1):
                self.u[i] = un[i] + ((self.nu * self.dt) / (self.dx **2)) * (un[i+1] - 2 * un[i] + un[i-1])

if __name__ == '__main__':
    x = diff_1d()
    plt.plot(np.linspace(0, x.grid_length, x.grid_points), x.u, label="Initial")
    x.disc_1d()
    plt.plot(np.linspace(0, x.grid_length, x.grid_points), x.u, label="After")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend()
    plt.tick_params(direction="in", top=True, right=True)
    plt.title("1-D Diffusion")
    plt.show()
