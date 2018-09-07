import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation
import step06

if __name__ == '__main__':
    x = step06.nonlin_2dconv()
    fig = plt.figure(figsize=(11, 7), dpi=100)
    ax = fig.gca(projection="3d")
    X, Y = np.meshgrid(x.x_points, x.y_points)
    surf = ax.plot_surface(X, Y, x.u[:], cmap=cm.viridis)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$u$")
    ax.text2D(0.35, 0.95, "2-D Non-Linear Convection", transform=ax.transAxes)


    def initialise():
        ax.clear()
        surf = ax.plot_surface(X, Y, x.u[:], cmap=cm.viridis)
        return surf

    def animate(j):
        un = x.u.copy()
        vn = x.v.copy()

        x.u[1:, 1:] = (un[1:, 1:] - (un[1:, 1:] * x.c * (x.dt / x.dx) * (un[1:, 1:] - un[1:, :-1])) - (vn[1:, 1:] * x.c * (x.dt / x.dx) * (un[1:, 1:] - un[:-1, 1:])))
        x.v[1:, 1:] = (vn[1:, 1:] - (un[1:, 1:] * x.c * (x.dt / x.dx) * (vn[1:, 1:] - vn[1:, :-1])) - (vn[1:, 1:] * x.c * (x.dt / x.dx) * (vn[1:, 1:] - vn[:-1, 1:])))

        x.u[0, :], x.u[-1, :], x.u[:, 0], x.u[:, -1] = [1,1,1,1]  # boundary conditions
        x.v[0, :], x.v[-1, :], x.v[:, 0], x.v[:, -1] = [1,1,1,1]

        ax.clear()
        surf = ax.plot_surface(X, Y, x.u[:], cmap=cm.viridis)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_zlabel("$u$")
        ax.text2D(0.35, 0.95, "2-D Non-Linear Convection", transform=ax.transAxes)
        return surf


    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=15, bitrate= 1800)
    anim = animation.FuncAnimation(fig, animate, init_func = initialise, frames = x.nt, interval=20)
    anim.save("step06.mp4", writer=writer)  # animation takes a while..
