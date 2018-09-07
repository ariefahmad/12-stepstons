import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation
import step09

if __name__ == '__main__':
    x = step09.laplace_2d()
    fig = plt.figure(figsize=(11, 7), dpi=100)
    ax = fig.gca(projection="3d")
    X, Y = np.meshgrid(x.x_points, x.y_points)
    surf = ax.plot_surface(X, Y, x.p[:], cmap=cm.viridis)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$u$")
    ax.text2D(0.35, 0.95, "2-D Laplace", transform=ax.transAxes)
    ax.view_init(azim=145)


    def initialise():
        ax.clear()
        surf = ax.plot_surface(X, Y, x.p[:], cmap=cm.viridis)
        return surf

    def animate(j):
        pn = x.p.copy()

        x.p[1: -1, 1: -1] = ((x.dy ** 2 * (pn[1: -1, 2:] + pn[1: -1, 0: -2]) +
                                x.dx ** 2 * (pn[2:, 1: -1] + pn[0: -2, 1: -1])) /
                                (2 * (x.dx ** 2 + x.dy **2)))

        # boundary conditions
        x.p[:, 0] = 0  # p = 0 at x = 0
        x.p[:, -1] = x.y_points  # p = y at x = 2
        x.p[0, :] = x.p[1, :]  # dp/dy = 0 at y = 0
        x.p[-1, :] = x.p[-2, :]  # dp/dy = 0 at y = 1

        ax.clear()
        surf = ax.plot_surface(X, Y, x.p[:], cmap=cm.viridis)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_zlabel("$u$")
        ax.text2D(0.35, 0.95, "2-D Laplace", transform=ax.transAxes)
        ax.view_init(azim=145)
        return surf


    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=15, bitrate= 1800)
    anim = animation.FuncAnimation(fig, animate, init_func = initialise, frames = 200, interval=20)
    anim.save("step09.mp4", writer=writer)  # animation takes a while..
