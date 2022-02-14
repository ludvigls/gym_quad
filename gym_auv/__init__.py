import numpy as np

from gym.envs.registration import register


pid_pathcolav_config = {
    "step_size": 0.05,#0.10,
    "max_t_steps": 100000,
    "min_reward": -1000,
    "n_obs_states": 12,
    "cruise_speed": 1.5,
    "lambda_reward": 0.6,
    "reward_roll": -1,
    "reward_rollrate": -1,
    "reward_control_derivative": [-0.005, -0.005],
    "reward_heading_error": -1,
    "reward_crosstrack_error": -0.0001,
    "reward_pitch_error": -1,
    "reward_verticaltrack_error": -0.0001,
    "reward_use_rudder": -0.1,
    "reward_use_elevator": -0.1,
    "reward_collision": 0,
    "sensor_span": (140,140), # the horizontal and vertical span of the sensors
    "sensor_suite": (15, 15), # the number of sensors covering the horizontal and vertical span
    "sensor_input_size": (8,8), # the shape of FLS data passed to the neural network. Max pooling from raw data is used
    "sensor_frequency": 1,
    "sonar_range": 25,
    "n_obs_errors": 2,
    "n_obs_inputs": 0,
    "n_actuators": 4,
    "la_dist": 3,
    "accept_rad": 1,
    "n_waypoints": 7,
    "n_int_obstacles": 1,
    "n_pro_obstacles": 3,
    "n_adv_obstacles": 8
}

register(
    id='PathColav3d-v0',
    entry_point='gym_auv.envs:PathColav3d',
    kwargs={'env_config': pid_pathcolav_config}
)
