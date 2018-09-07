import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation
import step07

if __name__ == '__main__':
    x = step07.diff_2d()
    fig = plt.figure(figsize=(11, 7), dpi=100)
    ax = fig.gca(projection="3d")
    X, Y = np.meshgrid(x.x_points, x.y_points)
    surf = ax.plot_surface(X, Y, x.u[:], cmap=cm.viridis)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$u$")
    ax.text2D(0.35, 0.95, "2-D Diffusion", transform=ax.transAxes)


    def initialise():
        ax.clear()
        surf = ax.plot_surface(X, Y, x.u[:], cmap=cm.viridis)
        return surf

    def animate(j):
        un = x.u.copy()

        x.u[1: -1, 1: -1] = (un[1: -1, 1: -1] + x.nu * (x.dt / (x.dx ** 2) * (un[1: -1, 2:] - 2 * un[1: -1, 1: -1] + un[1: -1, 0: -2]) +
                                x.nu * x.dt / (x.dx ** 2) * (un[2:, 1: -1] - 2 * un[1: -1, 1: -1] + un[0: -2, 1: -1])))
        x.u[0, :], x.u[-1, :], x.u[:, 0], x.u[:, -1] = [1,1,1,1]  # boundary conditions

        ax.clear()
        surf = ax.plot_surface(X, Y, x.u[:], cmap=cm.viridis)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_zlabel("$u$")
        ax.text2D(0.35, 0.95, "2-D Diffusion", transform=ax.transAxes)
        return surf


    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=15, bitrate= 1800)
    anim = animation.FuncAnimation(fig, animate, init_func = initialise, frames = x.nt, interval=20)
    anim.save("step07.mp4", writer=writer)  # animation takes a while..
