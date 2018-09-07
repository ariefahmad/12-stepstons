import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

class burgers_1d():
    """
    Solving Burger's equation  in 1D
    """
    def __init__(self):
        self.grid_length = 2
        self.grid_points = 101
        self.dx = self.grid_length * np.pi / (self.grid_points - 1)  # spacing between grids
        self.nt = 150  # number of timesteps

        self.nu = 0.07  # new viscosity
        self.nCFL = 0.2  # CFL number
        self.dt = self.dx * self.nu  # scale dt based on grid size; safety step to ensure convergence

        # applying initial conditions
        self.x_points = np.linspace(0, 2 * np.pi, self.grid_points)
        t = 0
        self.u = np.asarray([self.burger_func(t, x0, self.nu) for x0 in self.x_points])

    def burger_func(self, t, x, nu):
        """
        Convection + Diffusion?? -> Burgers!!
        use sympy to make expression; then lambdify for usage in python
        Will be used to set up initial conditions as in __init__
        """

        x_sp, nu_sp, t_sp = sp.symbols("x_spsp nu_sp t_sp")
        phi = (sp.exp(-(x_sp - 4 * t_sp) ** 2 / (4 * nu_sp * (t_sp+1))) + sp.exp(-(x_sp - 4 * t_sp - 2 * np.pi) ** 2 / (4 * nu_sp * (t_sp+1))))
        phi_dx = phi.diff(x_sp)

        u = -2 * nu_sp * (phi_dx / phi) + 4
        ufunc = sp.utilities.lambdify((t_sp, x_sp, nu_sp), u)  # usable function

        return ufunc(t, x, nu)

    def disc_1d(self):
        """
        Applying 1d-discretisation; for Burgers equation
        Will introduce periodic boundary conditions here, btw
        also get analytical results here
        """

        for n in range(0, self.nt):
            un = self.u.copy()
            for i in range(1, self.grid_points - 1):
                self.u[i] = un[i] - un[i] * (self.dt / self.dx) * (un[i] - un[i-1]) + self.nu * (self.dt / (self.dx ** 2)) * (un[i+1] - 2 * un[i] + un[i-1])

            self.u[0] = un[0] - un[0] * (self.dt / self.dx) * (un[0] - un[-2]) + (self.nu * self.dt / (self.dx ** 2)) * (un[1] - 2 * un[0] + un[-2])
            self.u[-1] = self.u[0]  # PBC

        # solving analytically
        self.anal_u = np.asarray([self.burger_func(self.nt*self.dt, xi, self.nu) for xi in self.x_points])


if __name__ == '__main__':
    x = burgers_1d()

    x.disc_1d()
    plt.plot(np.linspace(0, x.grid_length, x.grid_points), x.u, label="Computational")
    plt.plot(np.linspace(0, x.grid_length, x.grid_points), x.anal_u, label="Analytical")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend()
    plt.tick_params(direction="in", top=True, right=True)
    plt.title("1-D Burgers (End)")
    plt.show()
