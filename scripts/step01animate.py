import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import step01

if __name__ == '__main__':
    x = step01.lin_1dconv()
    fig, ax = plt.subplots()
    fig.set_size_inches(11, 7)
    ax.set_xlim((0, x.grid_length))
    ax.set_ylim((1, 2))

    plt.xlabel("x")
    plt.ylabel("u")
    plt.title("1-D Linear Convection")
    plt.tick_params(direction="in", top=True, right=True)

    line, = ax.plot([], [], lw = 2)

    def initialise():
        line.set_data([], [])
        return (line,)

    def animate(j):
        xx = np.linspace(0, x.grid_length, x.grid_points)
        un = x.u.copy()
        for i in range(1, x.grid_points):
            x.u[i] = un[i] - x.c * (x.dt / x.dx) * (un[i] - un[i-1])
        line.set_data(xx, x.u)
        return (line,)

    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=15, bitrate= 1800)
    anim = animation.FuncAnimation(fig, animate, init_func = initialise, frames = x.nt, interval=20)
    anim.save("step01.mp4", writer=writer)
