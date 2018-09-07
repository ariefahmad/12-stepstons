import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation
import step10

if __name__ == '__main__':
    x = step10.poisson_2d()
    fig = plt.figure(figsize=(11, 7), dpi=100)
    ax = fig.gca(projection="3d")
    X, Y = np.meshgrid(x.x_points, x.y_points)
    surf = ax.plot_surface(X, Y, x.p[:], cmap=cm.viridis)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$u$")
    ax.text2D(0.35, 0.95, "2-D Poisson", transform=ax.transAxes)
    ax.view_init(azim=145)


    def initialise():
        ax.clear()
        surf = ax.plot_surface(X, Y, x.b[:], cmap=cm.viridis)
        return surf

    def animate(j):
        pd = x.p.copy()

        x.p[1: -1, 1: -1] = (((pd[1: -1, 2:] + pd[1: -1, :-2]) * x.dy**2 +
                        (pd[2:, 1: -1] + pd[:-2, 1: -1]) * x.dx**2 -
                        x.b[1: -1, 1: -1] * x.dx**2 * x.dy**2) /
                        (2 * (x.dx**2 + x.dy**2)))

        x.p[0, :] = 0
        x.p[x.grid_points_y-1, :] = 0
        x.p[:, 0] = 0
        x.p[:, x.grid_points_x-1] = 0


        ax.clear()
        surf = ax.plot_surface(X, Y, x.p[:], cmap=cm.viridis)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_zlabel("$u$")
        ax.text2D(0.35, 0.95, "2-D Poisson", transform=ax.transAxes)
        ax.view_init(azim=145)
        return surf


    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=15, bitrate= 1800)
    anim = animation.FuncAnimation(fig, animate, init_func = initialise, frames = 200, interval=20)
    anim.save("step10.mp4", writer=writer)  # animation takes a while..
