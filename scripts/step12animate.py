import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import step12

if __name__ == '__main__':
    x = step12.channelflow_NS2d()
    X, Y = np.meshgrid(x.x_points, x.y_points)
    fig, ax = plt.subplots()
    fig.set_size_inches(11, 7)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.text(0.45, 1.05, "Channel Flow", transform=ax.transAxes)

    res_qv = ax.quiver(X[::3, ::3], Y[::3, ::3], x.u[::3, ::3], x.v[::3, ::3])

    def initialise():
        ax.clear()
        res_qv = ax.quiver(X[::3, ::3], Y[::3, ::3], x.u[::3, ::3], x.v[::3, ::3])
        return res_qv

    def animate(j):
        un = x.u.copy()
        vn = x.v.copy()

        x.build_up_b()
        x.pressure_poisson_periodic()

        x.u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * x.dt / x.dx *
                        (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * x.dt / x.dy *
                        (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                         x.dt / (2 * x.rho * x.dx) *
                        (x.p[1:-1, 2:] - x.p[1:-1, 0:-2]) +
                         x.nu * (x.dt / x.dx**2 *
                        (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                         x.dt / x.dy**2 *
                        (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])) +
                         x.F * x.dt)

        x.v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * x.dt / x.dx *
                        (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * x.dt / x.dy *
                        (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                         x.dt / (2 * x.rho * x.dy) *
                        (x.p[2:, 1:-1] - x.p[0:-2, 1:-1]) +
                         x.nu * (x.dt / x.dx**2 *
                        (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                         x.dt / x.dy**2 *
                        (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

        # Periodic BC u at x = 2
        x.u[1:-1, -1] = (un[1:-1, -1] - un[1:-1, -1] * x.dt / x.dx *
                      (un[1:-1, -1] - un[1:-1, -2]) -
                       vn[1:-1, -1] * x.dt / x.dy *
                      (un[1:-1, -1] - un[0:-2, -1]) -
                       x.dt / (2 * x.rho * x.dx) *
                      (x.p[1:-1, 0] - x.p[1:-1, -2]) +
                       x.nu * (x.dt / x.dx**2 *
                      (un[1:-1, 0] - 2 * un[1:-1,-1] + un[1:-1, -2]) +
                       x.dt / x.dy**2 *
                      (un[2:, -1] - 2 * un[1:-1, -1] + un[0:-2, -1])) + x.F * x.dt)

        # Periodic BC u at x = 0
        x.u[1:-1, 0] = (un[1:-1, 0] - un[1:-1, 0] * x.dt / x.dx *
                     (un[1:-1, 0] - un[1:-1, -1]) -
                      vn[1:-1, 0] * x.dt / x.dy *
                     (un[1:-1, 0] - un[0:-2, 0]) -
                      x.dt / (2 * x.rho * x.dx) *
                     (x.p[1:-1, 1] - x.p[1:-1, -1]) +
                      x.nu * (x.dt / x.dx**2 *
                     (un[1:-1, 1] - 2 * un[1:-1, 0] + un[1:-1, -1]) +
                      x.dt / x.dy**2 *
                     (un[2:, 0] - 2 * un[1:-1, 0] + un[0:-2, 0])) + x.F * x.dt)

        # Periodic BC v at x = 2
        x.v[1:-1, -1] = (vn[1:-1, -1] - un[1:-1, -1] * x.dt / x.dx *
                      (vn[1:-1, -1] - vn[1:-1, -2]) -
                       vn[1:-1, -1] * x.dt / x.dy *
                      (vn[1:-1, -1] - vn[0:-2, -1]) -
                       x.dt / (2 * x.rho * x.dy) *
                      (x.p[2:, -1] - x.p[0:-2, -1]) +
                       x.nu * (x.dt / x.dx**2 *
                      (vn[1:-1, 0] - 2 * vn[1:-1, -1] + vn[1:-1, -2]) +
                       x.dt / x.dy**2 *
                      (vn[2:, -1] - 2 * vn[1:-1, -1] + vn[0:-2, -1])))

        # Periodic BC v at x = 0
        x.v[1:-1, 0] = (vn[1:-1, 0] - un[1:-1, 0] * x.dt / x.dx *
                     (vn[1:-1, 0] - vn[1:-1, -1]) -
                      vn[1:-1, 0] * x.dt / x.dy *
                     (vn[1:-1, 0] - vn[0:-2, 0]) -
                      x.dt / (2 * x.rho * x.dy) *
                     (x.p[2:, 0] - x.p[0:-2, 0]) +
                      x.nu * (x.dt / x.dx**2 *
                     (vn[1:-1, 1] - 2 * vn[1:-1, 0] + vn[1:-1, -1]) +
                      x.dt / x.dy**2 *
                     (vn[2:, 0] - 2 * vn[1:-1, 0] + vn[0:-2, 0])))

        # Wall BC: u,v = 0 at y = 0,2
        x.u[0, :] = 0
        x.u[-1, :] = 0
        x.v[0, :] = 0
        x.v[-1, :] = 0

        ax.clear()
        res_qv = ax.quiver(X[::3, ::3], Y[::3, ::3], x.u[::3, ::3], x.v[::3, ::3])

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.text(0.45, 1.05, "Channel Flow", transform=ax.transAxes)

        return res_qv


    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=15, bitrate= 1800)
    anim = animation.FuncAnimation(fig, animate, init_func = initialise, frames = 200, interval=10)
    anim.save("step12.mp4", writer=writer)  # animation takes a while..
