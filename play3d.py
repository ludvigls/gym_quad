import numpy as np
import matplotlib.pyplot as plt
import gym
import gym_auv_3d
import os

from gym_auv_3d.utils.controllers import PI, PID
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3 import PPO
from utils import *


    
def play3d():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    experiment_dir, agent_path, scenario = parse_experiment_info()
    print(agent_path)
    env = gym.make("PathColav3d-v0", scenario=scenario)
    agent = PPO.load(agent_path)
    sim_df = simulate_environment(env, agent)
    sim_df.to_csv(r'simdata.csv')
    calculate_IAE(sim_df)
    plot_attitude(sim_df)
    plot_velocity(sim_df)
    #plot_angular_velocity(sim_df)
    #plot_control_inputs([sim_df])
    #plot_control_errors([sim_df])
    plot_reward(sim_df)
    plot_3d(env, sim_df)
    #plot_current_data(sim_df)

#def __main__():
#    play3d()