import numpy as np
import gym_auv.utils.geomutils as geom
import matplotlib.pyplot as plt

from scipy.signal import medfilt

class Current():
    """
    Creates an instance of an ocean current that is used in simulations.
    Attributes
    """
    def __init__(self, mu, Vmin, Vmax, Vc_init, alpha_init, beta_init, t_step):
        self.mu = mu
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.Vc = Vc_init
        self.alpha = alpha_init
        self.beta = beta_init
        self.t_step = t_step

    
    def __call__(self, state):
        """Returns the current velcotiy in {b} to use in in AUV kinematics"""
        phi = state[3]
        theta = state[4]
        psi = state[5]
        omega_body = state[9:12]

        vel_current_NED = np.array([self.Vc*np.cos(self.alpha)*np.cos(self.beta), self.Vc*np.sin(self.beta), self.Vc*np.sin(self.alpha)*np.cos(self.beta)])
        vel_current_BODY = np.transpose(geom.Rzyx(phi, theta, psi)).dot(vel_current_NED)
        vel_current_BODY_dot = -geom.S_skew(omega_body).dot(vel_current_BODY)

        nu_c = np.array([*vel_current_BODY, 0, 0, 0])
        nu_c_dot = np.array([*vel_current_BODY_dot, 0, 0, 0])

        return nu_c


    def sim(self):
        w = np.random.normal(0, 1)
        if self.Vc >= self.Vmax and w >= 0 or self.Vc <= self.Vmin and w <= 0:
            Vc_dot = 0
        else:
            Vc_dot = -self.mu*self.Vc + w
        #self.Vc += Vc_dot*self.t_step
        Vc = self.Vc+Vc_dot*self.t_step
        self.Vc = 0.99*self.Vc + 0.01*Vc
        