import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import step04

if __name__ == '__main__':
    x = step04.burgers_1d()
    fig, ax = plt.subplots()
    fig.set_size_inches(11, 7)
    ax.set_xlim((0, 2*np.pi))
    ax.set_ylim((0, 10))

    comp_line, = ax.plot([], [], marker="o", label="Computational", lw=2)
    anal_line, = ax.plot([], [], marker="o", label="Analytical", lw=2)

    plt.xlabel("x")
    plt.ylabel("u")
    plt.title("1-D Burgers")
    plt.tick_params(direction="in", top=True, right=True)
    plt.legend()

    def initialise():
        comp_line.set_data([], [])
        anal_line.set_data([], [])
        return (comp_line, anal_line,)

    def animate(j):
        un = x.u.copy()
        for i in range(1, x.grid_points - 1):
            x.u[i] = un[i] - un[i] * (x.dt / x.dx) * (un[i] - un[i-1]) + x.nu * (x.dt / (x.dx ** 2)) * (un[i+1] - 2 * un[i] + un[i-1])

        x.u[0] = un[0] - un[0] * (x.dt / x.dx) * (un[0] - un[-2]) + (x.nu * x.dt / (x.dx ** 2)) * (un[1] - 2 * un[0] + un[-2])
        x.u[-1] = x.u[0]  # PBC

        # solving analytically
        x.anal_u = np.asarray([x.burger_func(j*x.dt, xi, x.nu) for xi in x.x_points])

        comp_line.set_data(x.x_points, x.u)
        anal_line.set_data(x.x_points, x.anal_u)

        return (comp_line, anal_line,)

    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=15, bitrate= 1800)
    anim = animation.FuncAnimation(fig, animate, init_func = initialise, frames = x.nt, interval=20)
    anim.save("step04.mp4", writer=writer)  # animation takes a while..
