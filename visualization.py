"""
Contains methods for visualization and results saving
"""
import matplotlib.pyplot as plt
from matplotlib import animation
from pylab import *

# Animation
def animateU(U, xArgs, tArgs):
    # Detect the dependent variable limits
    uMatrixMin = U[0][0]
    uMatrixMax = U[0][0]
    for i in range(len(U)):
        for j in range(len(U[0])):
            if U[i][j] < uMatrixMin:
                uMatrixMin = U[i][j]
            if U[i][j] > uMatrixMax:
                uMatrixMax = U[i][j]
    fig = plt.figure()
    ax = plt.axes(xlim=(xArgs[0], xArgs[len(xArgs) - 1]), ylim=(uMatrixMin, uMatrixMax))
    line, = ax.plot([], [], lw=2)

    def init():
        line.set_data([], [])
        return line,

    def animate(j):
        x = xArgs
        y = [0] * len(xArgs)
        for i in range(len(xArgs)):
            y[i] = U[i][j]
            line.set_data(x, y)
        return line,

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(tArgs), interval=10, blit=True)
    anim.save('files/animation/BurgersEquation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    plt.show()


#Plot layer of U and saves image to disk
def plotLayerOfU(U, j, xArgs, t_min, delta_t):
    yValues = [0] * len(xArgs)
    strOut = str("t = " + str(t_min + j * delta_t))
    for i in range(len(xArgs)):
        yValues[i] = U[i][j]
    plt.suptitle(strOut, fontsize = 14)
    plt.clf()
    plt.plot(xArgs, yValues, linewidth=0.8)
    grid(True)
    str_screen_name = 'files/screenshots/screen_time_' + '{:06.3f}'.format(t_min + j * delta_t) + '.png'
    savefig(str_screen_name)
    # print("Saved figure: ", str_screen_name)
    # show()