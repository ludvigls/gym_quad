import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fminbound
from mpl_toolkits.mplot3d import Axes3D


class QPMI():
    def __init__(self, waypoints):
        self.waypoints = waypoints
        self.wp_idx = 0
        self.us = self._calculate_us()
        self.length = self.us[-1]
        self.calculate_quadratic_params()
    
    def _calculate_us(self):
        diff = np.diff(self.waypoints, axis=0)
        seg_lengths = np.cumsum(np.sqrt(np.sum(diff**2, axis=1)))
        return np.array([0,*seg_lengths[:]])
    
    def calculate_quadratic_params(self):
        self.x_params = []
        self.y_params = []
        self.z_params = []
        for n in range(1, len(self.waypoints)-1):
            wp_prev = self.waypoints[n-1]
            wp_n = self.waypoints[n]
            wp_next = self.waypoints[n+1]
            
            u_prev = self.us[n-1]
            u_n = self.us[n]
            u_next = self.us[n+1]

            U_n = np.vstack([np.hstack([u_prev**2, u_prev, 1]),
                           np.hstack([u_n**2, u_n, 1]),
                           np.hstack([u_next**2, u_next, 1])])
            x_params_n = np.linalg.inv(U_n).dot(np.array([wp_prev[0], wp_n[0], wp_next[0]]))
            y_params_n = np.linalg.inv(U_n).dot(np.array([wp_prev[1], wp_n[1], wp_next[1]]))
            z_params_n = np.linalg.inv(U_n).dot(np.array([wp_prev[2], wp_n[2], wp_next[2]]))

            self.x_params.append(x_params_n)
            self.y_params.append(y_params_n)
            self.z_params.append(z_params_n)

    def get_u_index(self, u):
        n = 0
        while n < len(self.us)-1:
            if u <= self.us[n+1]:
                break
            else:
                n += 1
        return n

    def calculate_ur(self, u):
        n = self.get_u_index(u)
        ur = (u-self.us[n])/(self.us[n+1] - self.us[n])
        return ur

    def calculate_uf(self, u):
        n = self.get_u_index(u)
        uf = (self.us[n+1]-u)/(self.us[n+1] - self.us[n])
        return uf        

    def __call__(self, u):
        if u >= self.us[0] and u <= self.us[1]: # first stretch
            ax = self.x_params[0][0]
            ay = self.y_params[0][0]
            az = self.z_params[0][0]
            bx = self.x_params[0][1]
            by = self.y_params[0][1]
            bz = self.z_params[0][1]
            cx = self.x_params[0][2]
            cy = self.y_params[0][2]
            cz = self.z_params[0][2]
            
            x = ax*u**2 + bx*u + cx
            y = ay*u**2 + by*u + cy
            z = az*u**2 + bz*u + cz
        elif u >= self.us[-2]-0.001 and u <= self.us[-1]: # last stretch
            ax = self.x_params[-1][0]
            ay = self.y_params[-1][0]
            az = self.z_params[-1][0]
            bx = self.x_params[-1][1]
            by = self.y_params[-1][1]
            bz = self.z_params[-1][1]
            cx = self.x_params[-1][2]
            cy = self.y_params[-1][2]
            cz = self.z_params[-1][2]
            
            x = ax*u**2 + bx*u + cx
            y = ay*u**2 + by*u + cy
            z = az*u**2 + bz*u + cz
        else: # else we are in the intermediate waypoints and we use membership functions to calc polynomials
            n = self.get_u_index(u)
            ur = self.calculate_ur(u)
            uf = self.calculate_uf(u)
            
            ax1 = self.x_params[n-1][0]
            ay1 = self.y_params[n-1][0]
            az1 = self.z_params[n-1][0]
            bx1 = self.x_params[n-1][1]
            by1 = self.y_params[n-1][1]
            bz1 = self.z_params[n-1][1]
            cx1 = self.x_params[n-1][2]
            cy1 = self.y_params[n-1][2]
            cz1 = self.z_params[n-1][2]
            
            x1 = ax1*u**2 + bx1*u + cx1
            y1 = ay1*u**2 + by1*u + cy1
            z1 = az1*u**2 + bz1*u + cz1

            ax2 = self.x_params[n][0]
            ay2 = self.y_params[n][0]
            az2 = self.z_params[n][0]
            bx2 = self.x_params[n][1]
            by2 = self.y_params[n][1]
            bz2 = self.z_params[n][1]
            cx2 = self.x_params[n][2]
            cy2 = self.y_params[n][2]
            cz2 = self.z_params[n][2]
            
            x2 = ax2*u**2 + bx2*u + cx2
            y2 = ay2*u**2 + by2*u + cy2
            z2 = az2*u**2 + bz2*u + cz2

            x = ur*x2 + uf*x1
            y = ur*y2 + uf*y1
            z = ur*z2 + uf*z1
        return np.array([x,y,z])

    def calculate_gradient(self, u):
        if u >= self.us[0] and u <= self.us[1]: # first stretch
            ax = self.x_params[0][0]
            ay = self.y_params[0][0]
            az = self.z_params[0][0]
            bx = self.x_params[0][1]
            by = self.y_params[0][1]
            bz = self.z_params[0][1]
            
            dx = ax*u*2 + bx
            dy = ay*u*2 + by
            dz = az*u*2 + bz
        elif u >= self.us[-2]: # last stretch
            ax = self.x_params[-1][0]
            ay = self.y_params[-1][0]
            az = self.z_params[-1][0]
            bx = self.x_params[-1][1]
            by = self.y_params[-1][1]
            bz = self.z_params[-1][1]
            
            dx = ax*u*2 + bx
            dy = ay*u*2 + by
            dz = az*u*2 + bz
        else: # else we are in the intermediate waypoints and we use membership functions to calc polynomials
            n = self.get_u_index(u)
            ur = self.calculate_ur(u)
            uf = self.calculate_uf(u)
            
            ax1 = self.x_params[n-1][0]
            ay1 = self.y_params[n-1][0]
            az1 = self.z_params[n-1][0]
            bx1 = self.x_params[n-1][1]
            by1 = self.y_params[n-1][1]
            bz1 = self.z_params[n-1][1]
            
            dx1 = ax1*u*2 + bx1
            dy1 = ay1*u*2 + by1
            dz1 = az1*u*2 + bz1

            ax2 = self.x_params[n][0]
            ay2 = self.y_params[n][0]
            az2 = self.z_params[n][0]
            bx2 = self.x_params[n][1]
            by2 = self.y_params[n][1]
            bz2 = self.z_params[n][1]
            
            dx2 = ax2*u*2 + bx2
            dy2 = ay2*u*2 + by2
            dz2 = az2*u*2 + bz2

            dx = ur*dx2 + uf*dx1
            dy = ur*dy2 + uf*dy1
            dz = ur*dz2 + uf*dz1
        return np.array([dx,dy,dz])

    def get_direction_angles(self, u):
        dx, dy, dz = self.calculate_gradient(u)[:]
        azimuth = np.arctan2(dy, dx)
        elevation = np.arctan2(-dz, np.sqrt(dx**2 + dy**2))
        return azimuth, elevation
    
    def get_closest_u(self, position, wp_idx):
        x1 = self.us[wp_idx] - 10
        x2 = self.us[wp_idx+1] + 10 if wp_idx < len(self.us)-2 else self.length
        output = fminbound(lambda u: np.linalg.norm(self(u) - position), 
                        full_output=0, x1=x1, x2=x2, xtol=1e-6, maxfun=500)
        return output

    def get_closest_position(self, position, wp_idx):
        return self(self.get_closest_u(position))

    def get_endpoint(self):
        return self(self.length)

    def plot_path(self, wps_on=True):
        u = np.linspace(self.us[0], self.us[-1], 10000)
        quadratic_path = []
        for du in u:
                quadratic_path.append(self(du))
                self.get_direction_angles(du)
        quadratic_path = np.array(quadratic_path)
        ax = plt.axes(projection='3d')
        ax.plot3D(xs=quadratic_path[:,0], ys=quadratic_path[:,1], zs=quadratic_path[:,2], color="#3388BB", label="Path")
        if wps_on:
            for i, wp in enumerate(self.waypoints):
                if i == 1: ax.scatter3D(*wp, color="#EE6666", label="Waypoints")
                else: ax.scatter3D(*wp, color="#EE6666")
        return ax


def generate_random_waypoints(nwaypoints):
    waypoints = [np.array([0,0,0])]
    for i in range(nwaypoints-1):
        distance = 50
        azimuth = np.random.uniform(-np.pi/4, np.pi/4)
        elevation = np.random.uniform(-np.pi/4, np.pi/4)
        x = waypoints[i][0] + distance*np.cos(azimuth)*np.cos(elevation)
        y = waypoints[i][1] + distance*np.sin(azimuth)*np.cos(elevation)
        z = waypoints[i][2] - distance*np.sin(elevation)

        wp = np.array([x, y, z])
        waypoints.append(wp)
    return np.array(waypoints)


if __name__ == "__main__":
    wps = np.array([np.array([0,0,0]), np.array([20,10,15]), np.array([50,20,20]), np.array([80,20,15]), np.array([90,50,20]), np.array([80,80,15]), np.array([50,80,20]), np.array([20,60,15]), np.array([20,40,10]), np.array([0,0,0])])
    wps = np.array([np.array([0,0,0]), np.array([20,10,15]), np.array([50,20,20]), np.array([80,20,40]), np.array([90,50,50]),
                    np.array([80,80,60]), np.array([50,80,20]), np.array([20,60,15]), np.array([20,40,10]), np.array([0,0,0])])
    #wps = np.array([np.array([0,0,0]), np.array([20,25,22]), np.array([50,40,30]), np.array([90,55,60]), np.array([130,95,110]), np.array([155,65,86])])
    #wps = generate_random_waypoints(10)
    path = QPMI(wps)
   
    point = path(20)
    azi, ele = path.get_direction_angles(20)
    vec_x = point[0] + 20*np.cos(azi)*np.cos(ele)
    vec_y = point[1] + 20*np.sin(azi)*np.cos(ele)
    vec_z = point[2] - 20*np.sin(ele)
    
    ax = path.plot_path()
    ax.plot3D(xs=wps[:,0], ys=wps[:,1], zs=wps[:,2], linestyle="dashed", color="#33bb5c")
    #ax.plot3D(xs=[point[0],vec_x], ys=[point[1],vec_y], zs=[point[2], vec_z])
    #ax.scatter3D(*point)
    for wp in wps:
        ax.scatter3D(*wp, color="r")
    ax.legend(["QPMI path", "Linear piece-wise path", "Waypoints"], fontsize=14)
    plt.rc('lines', linewidth=3)
    ax.set_xlabel(xlabel="North [m]", fontsize=14)
    ax.set_ylabel(ylabel="East [m]", fontsize=14)
    ax.set_zlabel(zlabel="Down [m]", fontsize=14)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.zaxis.set_tick_params(labelsize=12)
    ax.view_init(elev=-165, azim=-15)
    plt.show()