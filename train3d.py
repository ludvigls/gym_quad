import os
import gym
import gym_auv
import stable_baselines3.common.results_plotter as results_plotter
import numpy as np

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3 import PPO
#from stable_baselines3.common.results_plotter import load_results, ts2xy
from utils import parse_experiment_info

from typing import Callable

scenarios = ["beginner"]#, "intermediate", "proficient", "advanced", "expert"]

"""
hyperparams = {
    'n_steps': 1024,
    #"'nminibatches': 256,
    'learning_rate': 1e-5,
    'batch_size': 32,
    'gae_lambda': 0.95,
    'gamma': 0.99,
    'n_epochs': 4,
    'clip_range': 0.2,
    'ent_coef': 0.01,
    'verbose': 2
    }
"""
hyperparams = {
    'n_steps': 1024,
    'learning_rate': 2.5e-4,
    'batch_size': 64,
    'gae_lambda': 0.95,
    'gamma': 0.99,
    'n_epochs': 4,
    'clip_range': 0.2,
    'ent_coef': 0.001,
    'verbose': 2
    }

def callback2(_locals, _globals):
    global n_steps, best_mean_reward
    if (n_steps + 1) % 50000 == 0:
        _self = _locals["self"]
        _self.save(os.path.join(agents_dir, "model_" + str(n_steps+1) + ".pkl"))
    n_steps += 1
    return True


def make_env(env_id: str, scenario: str, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """

    def _init() -> gym.Env:
        #env = gym.make(env_id, scenario=scenario)
        env = gym.make("PathColav3d-v0", scenario=scen)
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init

if __name__ == '__main__':
    experiment_dir, agent_path, _ = parse_experiment_info()
    
    for i, scen in enumerate(scenarios):
        agents_dir = os.path.join(experiment_dir, scen, "agents")
        tensorboard_dir = os.path.join(experiment_dir, scen, "tensorboard")
        #agents_dir=agent_path
        os.makedirs(agents_dir, exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)
        hyperparams["tensorboard_log"] = tensorboard_dir

        #if os.path.exists(os.path.join(experiment_dir, scen, "agents", "last_model.pkl")):
        #    print(experiment_dir, "ALREADY FINISHED TRAINING IN,", scen.upper(), "SKIPPING TO THE NEXT STAGE")
        #    continue

        num_envs = 2
        print("INITIALIZING", num_envs, scen.upper(), "ENVIRONMENTS...", end="")
        if num_envs > 1:
            env = SubprocVecEnv(
                [lambda: Monitor(gym.make("PathColav3d-v0", scenario=scen), agents_dir, allow_early_resets=True)
                for i in range(num_envs)]
            )
        else:
            env = DummyVecEnv(
                [lambda: Monitor(gym.make("PathColav3d-v0", scenario=scen), agents_dir, allow_early_resets=True)]
            )
        print("DONE")

        print("INITIALIZING AGENT...", end="")

        if scen == "beginner":
            agent = PPO('MlpPolicy', env, **hyperparams)
        elif os.path.exists(os.path.join(experiment_dir, scen, "agents", "last_model1.pkl")):
            agent=PPO.load(agent_path)
        else:
            continual_model = os.path.join(experiment_dir, scenarios[i-1], "agents", "last_model.pkl")
            agent = PPO.load(continual_model, _init_setup_model=True, env=env, **hyperparams)
        print("DONE")
        best_mean_reward, n_steps, timesteps = -np.inf, 0, int(15e6)#int(300e3 + i*150e3)
        print("TRAINING FOR", timesteps, "TIMESTEPS")
        agent.learn(total_timesteps=timesteps, tb_log_name="PPO2", callback=callback2)
        print("FINISHED TRAINING AGENT IN", scen.upper())
        save_path = os.path.join(agents_dir, "last_model1.pkl")
        agent.save(save_path)
        print("SAVE SUCCESSFUL")