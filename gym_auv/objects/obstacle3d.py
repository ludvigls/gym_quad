import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



class Obstacle():
    def __init__(self, radius, position):
        self.position = np.array(position)
        self.radius = radius
        self.observed = False
        self.collided = False


    def return_plot_variables(self):
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = self.position[0] + self.radius*np.cos(u)*np.sin(v)
        y = self.position[1] + self.radius*np.sin(u)*np.sin(v)
        z = self.position[2] + self.radius*np.cos(v)
        return [x,y,z]


if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    o1 = Obstacle(3, [0,0,0])
    ax.plot_surface(*o1.return_plot_variables())
    o2 = Obstacle(5, [10,0,0])
    ax.plot_surface(*o2.return_plot_variables(), color='r')
    ax.set_xlim([-20,20])
    ax.set_ylim([-20,20])
    ax.set_zlim([-20,20])
    plt.show()
