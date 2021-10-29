import numpy as np
import gym
import gym_auv_3d.utils.geomutils as geom
import matplotlib.pyplot as plt
import skimage.measure

from gym_auv_3d.objects.auv3d import AUV3D
from gym_auv_3d.objects.current3d import Current
from gym_auv_3d.objects.QPMI import QPMI, generate_random_waypoints
from gym_auv_3d.objects.path3d import Path3D
from gym_auv_3d.objects.obstacle3d import Obstacle
from gym_auv_3d.utils.controllers import PI, PID


test_waypoints = np.array([np.array([0,0,0]), np.array([20,10,15]), np.array([50,20,20]), np.array([80,20,40]), np.array([90,50,50]),
                           np.array([80,80,60]), np.array([50,80,20]), np.array([20,60,15]), np.array([20,40,10]), np.array([0,0,0])])

test_waypoints = np.array([np.array([0,0,0]), np.array([50,15,5]), np.array([80,5,-5]), np.array([120,10,0]), np.array([150,0,0])])


class PathColav3d(gym.Env):
    """
    Creates an environment with a vessel, a path and obstacles.
    """
    def __init__(self, env_config, scenario="beginner"):
        for key in env_config:
            setattr(self, key, env_config[key])
        self.n_observations = self.n_obs_states + self.n_obs_errors + self.n_obs_inputs + self.sensor_input_size[0]*self.sensor_input_size[1]
        self.action_space = gym.spaces.Box(#low=np.array([-1, -1], dtype=np.float32),
                                           low=np.array([-1]*self.n_actuators, dtype=np.float32),
                                           high=np.array([1]*self.n_actuators, dtype=np.float32),
                                           dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=np.array([-1]*self.n_observations, dtype=np.float32),
                                                high=np.array([1]*self.n_observations, dtype=np.float32),
                                                dtype=np.float32)
        
        self.scenario = scenario
        
        self.n_sensor_readings = self.sensor_suite[0]*self.sensor_suite[1]
        max_horizontal_angle = self.sensor_span[0]/2
        max_vertical_angle = self.sensor_span[1]/2
        self.sectors_horizontal = np.linspace(-max_horizontal_angle*np.pi/180, max_horizontal_angle*np.pi/180, self.sensor_suite[0])
        self.sectors_vertical =  np.linspace(-max_vertical_angle*np.pi/180, max_vertical_angle*np.pi/180, self.sensor_suite[1])
        self.update_sensor_step= 1/(self.step_size*self.sensor_frequency)
        
        self.scenario_switch = {
            # Training scenarios
            "beginner": self.scenario_beginner,
            "intermediate": self.scenario_intermediate,
            "proficient": self.scenario_proficient,
            "advanced": self.scenario_advanced,
            "expert": self.scenario_expert,
            # Testing scenarios
            "test_path": self.scenario_test_path,
            "test_path_current": self.scenario_test_path_current,
            "test": self.scenario_test,
            "test_current": self.scenario_test_current,
            "horizontal": self.scenario_horizontal_test,
            "vertical": self.scenario_vertical_test,
            "deadend": self.scenario_deadend_test
        }

        self.reset()


    def reset(self):
        """
        Resets environment to initial state. 
        """
        #print("ENVIRONMENT RESET INITIATED")
        self.vessel = None
        self.path = None
        self.u_error = None
        self.e = None
        self.h = None
        self.chi_error = None
        self.upsilon_error = None
        self.waypoint_index = 0
        self.prog = 0
        self.path_prog = []
        self.success = False

        self.obstacles = []
        self.nearby_obstacles = []
        self.sensor_readings = np.zeros(shape=self.sensor_suite, dtype=float)
        self.collided = False
        self.penalize_control = 0.0

        self.observation = None
        self.action_derivative = np.zeros(self.n_actuators)
        self.past_states = []
        self.past_actions = []
        self.past_errors = []
        self.past_obs = []
        self.current_history = []
        self.time = []
        self.total_t_steps = 0
        self.reward = 0

        self._generate()
        #print("\tENVIRONMENT GENERATED")
        self.update_control_errors()
        #print("\tCONTROL ERRORS UPDATED")
        self.observation = self.observe(np.zeros(6, dtype=float))
        #print("COMPLETE")
        return self.observation


    def _generate(self):
        """
        Generates environment with a vessel, potentially ocean current and a 3D path.
        """     
        # Generate training/test scenario
        scenario = self.scenario_switch.get(self.scenario, lambda: print("Invalid scenario"))
        #print("\tGENERATING", self.scenario.upper())
        init_state = scenario()
        # Generate AUV
        #print("\tGENERATING AUV")
        self.vessel = AUV3D(self.step_size, init_state)
        #print("\tGENERATING PI-CONTROLLER")
        self.thrust_controller = PI()
    

    def plot_section3(self):
        plt.rc('lines', linewidth=3)
        ax = self.plot3D(wps_on=False)
        ax.set_xlabel(xlabel="North [m]", fontsize=14)
        ax.set_ylabel(ylabel="East [m]", fontsize=14)
        ax.set_zlabel(zlabel="Down [m]", fontsize=14)
        ax.xaxis.set_tick_params(labelsize=12)
        ax.yaxis.set_tick_params(labelsize=12)
        ax.zaxis.set_tick_params(labelsize=12)
        ax.set_xticks([0, 50, 100])
        ax.set_yticks([-50, 0, 50])
        ax.set_zticks([-50, 0, 50])
        ax.view_init(elev=-165, azim=-35)
        ax.scatter3D(*self.vessel.position, label="Initial Position", color="y")

        self.axis_equal3d(ax)
        ax.legend(fontsize=14)
        plt.show()



    def step(self, action):
        """
        Simulates the environment one time-step. 
        """
        # Simulate Current
        self.current.sim()
        nu_c = self.current(self.vessel.state)
        self.current_history.append(nu_c[0:3])

        # Simulate AUV dynamics one time-step and save action and state
        self.update_control_errors()
        #thrust = self.thrust_controller.u(self.u_error)
        #action = np.hstack((thrust, action))
        action = np.clip(action, np.array([-1]*self.n_actuators),[1]*self.n_actuators)
        if len(self.past_actions) > 0:
            self.action_derivative = (action[1:]-self.past_actions[-1][1:])/(self.step_size)

        self.vessel.step(action, nu_c)
        self.past_states.append(np.copy(self.vessel.state))
        self.past_errors.append(np.array([self.u_error, self.chi_error, self.e, self.upsilon_error, self.h]))
        self.past_actions.append(self.vessel.input)

        if self.path:
            self.prog = self.path.get_closest_u(self.vessel.position, self.waypoint_index)
            self.path_prog.append(self.prog)
            
            # Check if a waypoint is passed
            k = self.path.get_u_index(self.prog)
            if k > self.waypoint_index:
                print("Passed waypoint {:d}".format(k+1))
                self.waypoint_index = k
        
        # Calculate reward based on observation and actions
        done, step_reward = self.step_reward(self.observation, action)
        info = {}

        # Make next observation
        self.observation = self.observe(nu_c)
        self.past_obs.append(self.observation)

        # Save sim time info
        self.total_t_steps += 1
        self.time.append(self.total_t_steps*self.step_size)
        
        return self.observation, step_reward, done, info


    def observe(self, nu_c):
        """
        Returns observations of the environment. 
        """
        obs = np.zeros((self.n_observations,))
        obs[0] = np.clip(self.vessel.relative_velocity[0] / 2, -1, 1)
        obs[1] = np.clip(self.vessel.relative_velocity[1] / 0.3, -1, 1)
        obs[2] = np.clip(self.vessel.relative_velocity[2] / 0.3, -1, 1)
        obs[3] = np.clip(self.vessel.roll / np.pi, -1, 1)
        obs[4] = np.clip(self.vessel.pitch / np.pi, -1, 1)
        obs[5] = np.clip(self.vessel.heading / np.pi, -1, 1)
        obs[6] = np.clip(self.vessel.angular_velocity[0] / 1.2, -1, 1)
        obs[7] = np.clip(self.vessel.angular_velocity[1] / 0.4, -1, 1)
        obs[8] = np.clip(self.vessel.angular_velocity[2] / 0.4, -1, 1)
        obs[9] = np.clip(nu_c[0] / 1, -1, 1)
        obs[10] = np.clip(nu_c[1] / 1, -1, 1)
        obs[11] = np.clip(nu_c[2] / 1, -1, 1)
        obs[12] = self.chi_error
        obs[13] = self.upsilon_error

        # Update nearby obstacles and calculate distances
        if self.total_t_steps % self.update_sensor_step == 0:
            self.update_nearby_obstacles()
            self.update_sensor_readings()
            self.sonar_observations = skimage.measure.block_reduce(self.sensor_readings, (2,2), np.max)
            #self.update_sensor_readings_with_plots() #(Debugging)
        obs[14:] = self.sonar_observations.flatten()
        return obs


    def step_reward(self, obs, action):
        """
        Calculates the reward function for one time step. Also checks if the episode should end. 
        """
        done = False
        step_reward = 0 
        #reward_distance=self.reward_distance*(np.linalg.norm(self.path.get_endpoint()-self.vessel.position))
        angular_velocity=self.vessel.angular_velocity
        reward_steady=self.reward_steady*(angular_velocity[0]**2+angular_velocity[1]**2+angular_velocity[2]**2)
        #reward_roll = self.vessel.roll**2*self.reward_roll + self.vessel.angular_velocity[0]**2*self.reward_rollrate
        #reward_control = action[1]**2*self.reward_use_rudder + action[2]**2*self.reward_use_elevator
        reward_path_following = self.chi_error**2*self.reward_heading_error + self.upsilon_error**2*self.reward_pitch_error
        reward_collision_avoidance = self.penalize_obstacle_closeness()

        #step_reward = self.lambda_reward*reward_path_following + (1-self.lambda_reward)*reward_collision_avoidance \
         #           + reward_roll #+ reward_control
        step_reward =  self.lambda_reward*reward_path_following + (1-self.lambda_reward)*reward_collision_avoidance+reward_steady#reward_roll#+reward_distance#+ reward_steady
        self.reward += step_reward

        # Check collision
        for obstacle in self.nearby_obstacles:
            if np.linalg.norm(obstacle.position - self.vessel.position) <= obstacle.radius + self.vessel.safety_radius:
                self.collided = True
        
        end_cond_1 = self.reward < self.min_reward
        end_cond_2 = self.total_t_steps >= self.max_t_steps
        end_cond_3 = np.linalg.norm(self.path.get_endpoint()-self.vessel.position) < self.accept_rad and self.waypoint_index == self.n_waypoints-2
        end_cond_4 = abs(self.prog - self.path.length) <= self.accept_rad/2.0
        if end_cond_1 or end_cond_2 or end_cond_3 or end_cond_4:
            if end_cond_3:
                print("AUV reached target!")
                self.success = True
            elif self.collided:
                print("AUV collided!")
                print(np.round(self.sensor_readings,2))
                self.success = False
            print("Episode finished after {} timesteps with reward: {}".format(self.total_t_steps, self.reward.round(1)))
            done = True
        return done, step_reward


    def update_control_errors(self):
        # Update cruise speed error
        self.u_error = np.clip((self.cruise_speed - self.vessel.relative_velocity[0])/2, -1, 1)
        self.chi_error = 0.0
        self.e = 0.0
        self.upsilon_error = 0.0
        self.h = 0.0

        # Get path course and elevation
        s = self.prog
        chi_p, upsilon_p = self.path.get_direction_angles(s)

        # Calculate tracking errors
        SF_rotation = geom.Rzyx(0,upsilon_p,chi_p)
        epsilon = np.transpose(SF_rotation).dot(self.vessel.position-self.path(self.prog))
        e = epsilon[1]
        h = epsilon[2]

        # Calculate course and elevation errors from tracking errors
        chi_r = np.arctan2(-e, self.la_dist)
        upsilon_r = np.arctan2(h, np.sqrt(e**2 + self.la_dist**2))
        chi_d = chi_p + chi_r
        upsilon_d = upsilon_p + upsilon_r
        self.chi_error = np.clip(geom.ssa(self.vessel.chi - chi_d)/np.pi, -1, 1)
        #self.e = np.clip(e/12, -1, 1)
        self.e = e
        self.upsilon_error = np.clip(geom.ssa(self.vessel.upsilon - upsilon_d)/np.pi, -1, 1)
        #self.h = np.clip(h/12, -1, 1)
        self.h = h


    def update_nearby_obstacles(self):
        """
        Updates the nearby_obstacles array.
        """
        self.nearby_obstacles = []
        for obstacle in self.obstacles:
            distance_vec_NED = obstacle.position - self.vessel.position
            distance = np.linalg.norm(distance_vec_NED)
            distance_vec_BODY = np.transpose(geom.Rzyx(*self.vessel.attitude)).dot(distance_vec_NED)
            heading_angle_BODY = np.arctan2(distance_vec_BODY[1], distance_vec_BODY[0])
            pitch_angle_BODY = np.arctan2(distance_vec_BODY[2], np.sqrt(distance_vec_BODY[0]**2 + distance_vec_BODY[1]**2))
            # check if the obstacle is inside the sonar window
            if distance - self.vessel.safety_radius - obstacle.radius <= self.sonar_range and abs(heading_angle_BODY) <= self.sensor_span[0]*np.pi/180 \
            and abs(pitch_angle_BODY) <= self.sensor_span[1]*np.pi/180:
                self.nearby_obstacles.append(obstacle)
            elif distance <= obstacle.radius + self.vessel.safety_radius:
                self.nearby_obstacles.append(obstacle)


    def update_sensor_readings(self):
        """
        Updates the sonar data closeness array.
        """
        self.sensor_readings = np.zeros(shape=self.sensor_suite, dtype=float)
        for obstacle in self.nearby_obstacles:
            for i in range(self.sensor_suite[0]):
                alpha = self.vessel.heading + self.sectors_horizontal[i]
                for j in range(self.sensor_suite[1]):
                    beta = self.vessel.pitch + self.sectors_vertical[j]
                    _, closeness = self.calculate_object_distance(alpha, beta, obstacle)
                    self.sensor_readings[j,i] = max(closeness, self.sensor_readings[j,i]) 


    def update_sensor_readings_with_plots(self):
        """
        Updates the sonar data array and renders the simulations as 3D plot. Used for debugging.
        """
        print("Time: {}, Nearby Obstacles: {}".format(self.total_t_steps, len(self.nearby_obstacles)))
        self.sensor_readings = np.zeros(shape=self.sensor_suite, dtype=float)
        ax = self.plot3D()
        ax2 = self.plot3D()
        for obstacle in self.nearby_obstacles:
            for i in range(self.sensor_suite[0]):
                alpha = self.vessel.heading + self.sectors_horizontal[i]
                for j in range(self.sensor_suite[1]):
                    beta = self.vessel.pitch + self.sectors_vertical[j]
                    s, closeness = self.calculate_object_distance(alpha, beta, obstacle)
                    self.sensor_readings[j,i] = max(closeness, self.sensor_readings[j,i])              
                    color = "#05f07a" if s >= self.sonar_range else "#a61717"
                    s = np.linspace(0, s, 100)
                    x = self.vessel.position[0] + s*np.cos(alpha)*np.cos(beta)
                    y = self.vessel.position[1] + s*np.sin(alpha)*np.cos(beta)
                    z = self.vessel.position[2] - s*np.sin(beta)
                    ax.plot3D(x, y, z, color=color)
                    if color == "#a61717": ax2.plot3D(x, y, z, color=color)
                plt.rc('lines', linewidth=3)
        ax.set_xlabel(xlabel="North [m]", fontsize=14)
        ax.set_ylabel(ylabel="East [m]", fontsize=14)
        ax.set_zlabel(zlabel="Down [m]", fontsize=14)
        ax.xaxis.set_tick_params(labelsize=12)
        ax.yaxis.set_tick_params(labelsize=12)
        ax.zaxis.set_tick_params(labelsize=12)
        ax.scatter3D(*self.vessel.position, color="y", s=40, label="AUV")
        print(np.round(self.sensor_readings,3))
        self.axis_equal3d(ax)
        ax.legend(fontsize=14)
        ax2.set_xlabel(xlabel="North [m]", fontsize=14)
        ax2.set_ylabel(ylabel="East [m]", fontsize=14)
        ax2.set_zlabel(zlabel="Down [m]", fontsize=14)
        ax2.xaxis.set_tick_params(labelsize=12)
        ax2.yaxis.set_tick_params(labelsize=12)
        ax2.zaxis.set_tick_params(labelsize=12)
        ax2.scatter3D(*self.vessel.position, color="y", s=40, label="AUV")
        self.axis_equal3d(ax2)
        ax2.legend(fontsize=14)
        plt.show()


    def calculate_object_distance(self, alpha, beta, obstacle):
        """
        Searches along a sonar ray for an object
        """
        s = 0
        while s < self.sonar_range:
            x = self.vessel.position[0] + s*np.cos(alpha)*np.cos(beta)
            y = self.vessel.position[1] + s*np.sin(alpha)*np.cos(beta)
            z = self.vessel.position[2] - s*np.sin(beta)
            if np.linalg.norm(obstacle.position - [x,y,z]) <= obstacle.radius:
                break
            else:
                s += 1
        closeness = np.clip(1-(s/self.sonar_range), 0, 1)
        return s, closeness


    def penalize_obstacle_closeness(self):
        """
        Calculates the colav reward
        """
        reward_colav = 0
        sensor_suite_correction = 0
        gamma_c = self.sonar_range/2
        epsilon = 0.05
        epsilon_closeness = 0.05
        horizontal_angles = np.linspace(-self.sensor_span[0]/2, self.sensor_span[0]/2, self.sensor_suite[0])
        vertical_angles = np.linspace(-self.sensor_span[1]/2, self.sensor_span[1]/2, self.sensor_suite[1])
        for i, horizontal_angle in enumerate(horizontal_angles):
            horizontal_factor = (1-(abs(horizontal_angle)/horizontal_angles[-1]))
            for j, vertical_angle in enumerate(vertical_angles):
                vertical_factor = (1-(abs(vertical_angle)/vertical_angles[-1]))
                beta = vertical_factor*horizontal_factor + epsilon
                sensor_suite_correction += beta
                reward_colav += (beta*(1/(gamma_c*max(1-self.sensor_readings[j,i], epsilon_closeness)**2)))**2
        return - reward_colav / sensor_suite_correction

    
    def plot3D(self, wps_on=True):
        """
        Returns 3D plot of path and obstacles.
        """
        ax = self.path.plot_path(wps_on)
        for obstacle in self.obstacles:    
            ax.plot_surface(*obstacle.return_plot_variables(), color='r')
        return self.axis_equal3d(ax)


    def axis_equal3d(self, ax):
        """
        Shifts axis in 3D plots to be equal. Especially useful when plotting obstacles, so they appear spherical.
        
        Parameters:
        ----------
        ax : matplotlib.axes
            The axes to be shifted. 
        """
        extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
        sz = extents[:,1] - extents[:,0]
        centers = np.mean(extents, axis=1)
        maxsize = max(abs(sz))
        r = maxsize/2
        for ctr, dim in zip(centers, 'xyz'):
            getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
        return ax


    def check_object_overlap(self, new_obstacle):
        """
        Checks if a new obstacle is overlapping one that already exists or the target position.
        """
        overlaps = False
        # check if it overlaps target:
        if np.linalg.norm(self.path.get_endpoint() - new_obstacle.position) < new_obstacle.radius + 5:
            return True
        # check if it overlaps already placed objects
        for obstacle in self.obstacles:
            if np.linalg.norm(obstacle.position - new_obstacle.position) < new_obstacle.radius + obstacle.radius + 5:
                overlaps = True
        return overlaps


    def scenario_beginner(self):
        initial_state = np.zeros(6)
        self.current = Current(mu=0, Vmin=0, Vmax=0, Vc_init=0, alpha_init=0, beta_init=0, t_step=0) #Current object with zero velocity
        waypoints = generate_random_waypoints(self.n_waypoints)
        self.path = QPMI(waypoints)
        init_pos = [np.random.uniform(0,2)*(-5), np.random.normal(0,1)*5, np.random.normal(0,1)*5]
        init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]]) #change to zero vector?
        initial_state = np.hstack([init_pos, init_attitude])
        return initial_state


    def scenario_intermediate(self):
        #print("\t\t\tfunc scenario_intermediate init")
        initial_state = self.scenario_beginner()
        #print("\t\t\tfunc scenario_intermediate got beginner")
        rad = np.random.uniform(4, 10)
        pos = self.path(self.path.length/2)
        self.obstacles.append(Obstacle(radius=rad, position=pos))
        lengths = np.linspace(self.path.length*1/3, self.path.length*2/3, self.n_int_obstacles)
        for l in lengths:
            obstacle_radius = np.random.uniform(low=4,high=10)
            obstacle_coords = self.path(l)
            obstacle = Obstacle(obstacle_radius, obstacle_coords)
            if self.check_object_overlap(obstacle):
                continue
            else:
                self.obstacles.append(obstacle)
        #print("\n\t\tfunc scenario_intermediate generated", len(self.obstacles), "obstacles")
        #print("\t\t\tfunc scenario_intermediate exit")
        return initial_state


    def scenario_proficient(self):
        #print("\t\tfunc scenario_proficient init")
        initial_state = self.scenario_intermediate()
        #print("\t\t\tgot intermediate (", len(self.obstacles), " obstacles)", sep="")
        lengths = np.random.uniform(self.path.length*1/3, self.path.length*2/3, self.n_pro_obstacles)
        #print("\t\t\tgot", len(lengths), "lengths")
        print("")
        n_checks = 0
        while len(self.obstacles) < self.n_pro_obstacles and n_checks < 1000:
            for l in lengths:
                obstacle_radius = np.random.uniform(low=4,high=10)
                obstacle_coords = self.path(l)
                obstacle = Obstacle(obstacle_radius, obstacle_coords)
                if self.check_object_overlap(obstacle):
                    n_checks += 1
                    #print("\r\t\t\tOVERLAP CHECK TRIGGERED", n_checks, "TIMES", end="", flush=True)
                    continue

                else:
                    self.obstacles.append(obstacle)
        print("\t\tfunc scenario_proficient() --> OVERLAP CHECK TRIGGERED", n_checks, "TIMES") if n_checks > 1 else None
        #print("\n\t\t\t", len(self.obstacles), " obstacles in total", sep="")
        #print("\t\tfunc scenario_proficient exit")
        return initial_state


    def scenario_advanced(self):
        initial_state = self.scenario_proficient()
        while len(self.obstacles) < self.n_adv_obstacles: # Place the rest of the obstacles randomly
            s = np.random.uniform(self.path.length*1/3, self.path.length*2/3)
            obstacle_radius = np.random.uniform(low=4,high=10)
            obstacle_coords = self.path(s) + np.random.uniform(low=-(obstacle_radius+10), high=(obstacle_radius+10), size=(1,3))
            obstacle = Obstacle(obstacle_radius, obstacle_coords[0])
            if self.check_object_overlap(obstacle):
                continue
            else:
                self.obstacles.append(obstacle)
        return initial_state


    def scenario_expert(self):
        initial_state = self.scenario_advanced()
        self.current = Current(mu=0.2, Vmin=0.5, Vmax=1.0, Vc_init=np.random.uniform(0.5, 1), \
                                    alpha_init=np.random.uniform(-np.pi, np.pi), beta_init=np.random.uniform(-np.pi/4, np.pi/4), t_step=self.step_size)
        self.penalize_control = 1.0
        return initial_state


    def scenario_test_path(self):
        self.n_waypoints = len(test_waypoints)
        self.path = QPMI(test_waypoints)
        self.current = Current(mu=0, Vmin=0, Vmax=0, Vc_init=0, alpha_init=0, beta_init=0, t_step=0)
        init_pos = [0,0,0]
        init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([init_pos, init_attitude])
        return initial_state
        

    def scenario_test_path_current(self):
        initial_state = self.scenario_test_path()
        self.current = Current(mu=0, Vmin=0.75, Vmax=0.75, Vc_init=0.75, alpha_init=np.pi/4, beta_init=np.pi/6, t_step=0)
        return initial_state 


    def scenario_test(self):
        initial_state = self.scenario_test_path()
        points = np.linspace(self.path.length/4, 3*self.path.length/4, 3)
        self.obstacles.append(Obstacle(radius=10, position=self.path(self.path.length/2)))
        return initial_state
        """
        radius = 6
        for p in points:
            pos = self.path(p)
            self.obstacles.append(Obstacle(radius=radius, position=pos))
        return initial_state
        """


    def scenario_test_current(self):
        initial_state = self.scenario_test()
        self.current = Current(mu=0, Vmin=0.75, Vmax=0.75, Vc_init=0.75, alpha_init=np.pi/4, beta_init=np.pi/6, t_step=0) # Constant velocity current (reproducability for report)
        return initial_state


    def scenario_horizontal_test(self):
        waypoints = [(0,0,0), (50,0,0), (100,0,0)]
        self.path = QPMI(waypoints)
        self.current = Current(mu=0, Vmin=0, Vmax=0, Vc_init=0, alpha_init=0, beta_init=0, t_step=0)
        self.obstacles = []
        for i in range(7):
            y = -30+10*i
            self.obstacles.append(Obstacle(radius=5, position=[50,y,0]))
        init_pos = [0,0,0]
        init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([init_pos, init_attitude])
        return initial_state


    def scenario_vertical_test(self):
        waypoints = [(0,0,0), (50,0,0), (100,0,0)]
        self.path = QPMI(waypoints)
        self.current = Current(mu=0, Vmin=0, Vmax=0, Vc_init=0, alpha_init=0, beta_init=0, t_step=0)
        self.obstacles = []
        for i in range(7):
            z = -30+10*i
            self.obstacles.append(Obstacle(radius=5, position=[50,0,z]))
        init_pos = [0,0,0]
        init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([init_pos, init_attitude])
        return initial_state


    def scenario_deadend_test(self):
        waypoints = [(0,0,0), (50,0,0), (100,0,0)]
        self.path = QPMI(waypoints)
        self.current = Current(mu=0, Vmin=0, Vmax=0, Vc_init=0, alpha_init=0, beta_init=0, t_step=0)
        radius = 25
        angles = np.linspace(-90, 90, 10)*np.pi/180
        obstalce_radius = (angles[1]-angles[0])*radius/2
        for ang1 in angles:
            for ang2 in angles:
                x = 30+radius*np.cos(ang1)*np.cos(ang2)
                y = radius*np.cos(ang1)*np.sin(ang2)
                z = -radius*np.sin(ang1)
                self.obstacles.append(Obstacle(obstalce_radius, [x, y, z]))
        init_pos = [0,0,0]
        init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([init_pos, init_attitude])
        return initial_state