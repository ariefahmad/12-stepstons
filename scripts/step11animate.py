import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import step11

if __name__ == '__main__':
    x = step11.cavityflow_NS2d()
    X, Y = np.meshgrid(x.x_points, x.y_points)
    fig, ax = plt.subplots()
    fig.set_size_inches(11, 7)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.text(0.45, 1.05, "Lid Cavity Flow", transform=ax.transAxes)

    div = make_axes_locatable(ax)
    cax = div.append_axes("right", "5%", "5%")

    res_cf = ax.contourf(X, Y, x.p, alpha=0.5, cmap=cm.viridis)
    res_cb = fig.colorbar(res_cf, cax=cax)
    res_cr = ax.contour(X, Y, x.p, cmap=cm.viridis)
    # res_qv = ax.quiver(X[::2, ::2], Y[::2, ::2], x.u[::2, ::2], x.v[::2, ::2])
    res_qv = ax.streamplot(X, Y, x.u, x.v)

    def initialise():
        ax.clear()
        res_cf = ax.contourf(X, Y, x.p, alpha=0.5, cmap=cm.viridis)
        res_cc = ax.contour(X, Y, x.p, cmap=cm.viridis)
        # res_qv = ax.quiver(X[::2, ::2], Y[::2, ::2], x.u[::2, ::2], x.v[::2, ::2])
        res_qv = ax.streamplot(X, Y, x.u, x.v)
        return res_cf, res_cc, res_qv

    def animate(j):
            un = x.u.copy()
            vn = x.v.copy()

            x.build_up_b()
            x.pressure_poisson()

            x.u[1:-1, 1:-1] = (un[1:-1, 1:-1]-
                             un[1:-1, 1:-1] * x.dt / x.dx *
                            (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                             x.v[1:-1, 1:-1] * x.dt / x.dy *
                            (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                             x.dt / (2 * x.rho * x.dx) * (x.p[1:-1, 2:] - x.p[1:-1, 0:-2]) +
                             x.nu * (x.dt / x.dx**2 *
                            (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                             x.dt / x.dy**2 *
                            (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

            x.v[1:-1,1:-1] = (vn[1:-1, 1:-1] -
                            un[1:-1, 1:-1] * x.dt / x.dx *
                           (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                            vn[1:-1, 1:-1] * x.dt / x.dy *
                           (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                            x.dt / (2 * x.rho * x.dy) * (x.p[2:, 1:-1] - x.p[0:-2, 1:-1]) +
                            x.nu * (x.dt / x.dx**2 *
                           (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                            x.dt / x.dy**2 *
                           (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

            x.u[0, :] = 0
            x.u[:, 0] = 0
            x.u[:, -1] = 0
            x.u[-1, :] = 1  # set velocity on cavity lid equal to 1
            x.v[0, :] = 0
            x.v[-1, :] = 0
            x.v[:, 0] = 0
            x.v[:, -1] = 0


            ax.clear()
            res_cf = ax.contourf(X, Y, x.p, alpha=0.5, cmap=cm.viridis)
            res_cc = ax.contour(X, Y, x.p, cmap=cm.viridis)
            cax.cla()
            res_cb = fig.colorbar(res_cf, cax=cax)
            # res_qv = ax.quiver(X[::2, ::2], Y[::2, ::2], x.u[::2, ::2], x.v[::2, ::2])
            res_qv = ax.streamplot(X, Y, x.u, x.v)

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.text(0.45, 1.05, "Lid Cavity Flow", transform=ax.transAxes)

            return res_cf, res_cc, res_qv


    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=15, bitrate= 1800)
    anim = animation.FuncAnimation(fig, animate, init_func = initialise, frames = x.nt, interval=20)
    anim.save("step11.mp4", writer=writer)  # animation takes a while..
