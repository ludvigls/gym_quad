
import os
import gym
import gym_auv
import stable_baselines3.common.results_plotter as results_plotter
import numpy as np
import torch
import onnx
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from radarCNN import *

#from stable_baselines3.common.results_plotter import load_results, ts2xy
from utils import parse_experiment_info

from typing import Callable

scenarios = ["line","line_new","horizontal_new", "3d_new","intermediate", "proficient", "advanced", "expert"]
#scenarios=['3d_new']
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
    'verbose': 2,
    'device':'cuda'
    }
policy_kwargs = dict(
                        features_extractor_class = PerceptionNavigationExtractor,
                        features_extractor_kwargs = dict(sensor_dim_x=15,sensor_dim_y=15,features_dim=16),
                        #net_arch = [128, 64, dict(pi=[32]), dict(vf=[32])]
                        #net_arch=[dict(pi=[64, 64], vf=[64, 64])]
                        net_arch=[dict(pi=[128, 64, 32], vf=[128, 64, 32])]
                    )
def callback2(_locals, _globals):
    global n_steps, best_mean_reward
    if (n_steps + 1) % 50000 == 0:
        _self = _locals["self"]
        _self.save(os.path.join(agents_dir, "model_" + str(n_steps+1) + ".pkl"))
    n_steps += 1
    return True
class StatsCallback(BaseCallback):
    def __init__(self):
        self.n_steps=0
        self.n_calls=0
        self.prev_stats=None
        self.ob_names=["u","v","w","roll","pitch","yaw","p","q","r","nu_c0","nu_c1","nu_c2","chi_err","upsilon_err","chi_err_1","upsilon_err_1","chi_err_5","upsilon_err_5"]
        self.state_names=["x","y","z","roll","pitch","yaw","u","v","w","p","q","r"]
        self.error_names=["e", "h"]
    def _on_step(self):

        done_array=np.array(self.locals.get("dones") if self.locals.get("dones") is not None else self.locals.get("dones"))
        stats=self.locals.get("self").get_env().env_method("get_stats")
        global n_steps
        
        for i in range(len(done_array)):
            #done,rewards in zip(done_array,rewards):   
            if done_array[i]:
                #print(done_array[i])
                #print(self.prev_stats[i])
                if  self.prev_stats is not None:
                    for stat in self.prev_stats[i].keys():
                        """
                        if stat=="obs":
                            obs=self.prev_stats[i][stat]
                            if len(obs)>0:
                                for k in range(len(obs[0])):
                                    temp=[]
                                    for j in range(len(obs)):
                                        temp.append(obs[j][k])
                                    self.logger.record('obs/'+self.ob_names[k]+'_mean',np.mean(temp))
                                    self.logger.record('obs/'+self.ob_names[k]+'_std',np.std(temp))
                        elif stat=="states":
                            states=self.prev_stats[i][stat]
                            if len(states)>0:
                                for k in range(len(states[0])):
                                    temp=[]
                                    for j in range(len(states)):
                                        temp.append(states[j][k])
                                    self.logger.record('states/'+self.state_names[k]+'_mean',np.mean(temp))
                                    self.logger.record('states/'+self.state_names[k]+'_std',np.std(temp))
                        elif stat=="errors":
                            errors=self.prev_stats[i][stat]
                            if len(errors)>0:
                                for k in range(len(errors[0])):
                                    temp=[]
                                    for j in range(len(errors)):
                                        temp.append(errors[j][k])
                                    self.logger.record('errors/'+self.error_names[k]+'_mean',np.mean(temp))
                                    self.logger.record('errors/'+self.error_names[k]+'_std',np.std(temp))
                        """
                        self.logger.record('stats/'+stat,self.prev_stats[i][stat])
        self.prev_stats=stats
        if (n_steps + 1) % 50000 == 0:
            _self = self.locals.get("self")
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
    print("makeenv")
    def _init() -> gym.Env:
        #env = gym.make(env_id, scenario=scenario)
        env = gym.make("PathColav3d-v0", scenario=scen)
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init

if __name__ == '__main__':
    experiment_dir, _, _ = parse_experiment_info()
    
    seed=np.random.randint(0,10000)
    with open('seed.txt', 'w') as file:
        file.write(str(seed))
    print("set seed"+" "+ experiment_dir)
    for i, scen in enumerate(scenarios):
        agents_dir = os.path.join(experiment_dir, scen, "agents")
        tensorboard_dir = os.path.join(experiment_dir, scen, "tensorboard")
        os.makedirs(agents_dir, exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)
        hyperparams["tensorboard_log"] = tensorboard_dir

        if os.path.exists(os.path.join(experiment_dir, scen, "agents", "last_model.pkl")):
            print(experiment_dir, "ALREADY FINISHED TRAINING IN,", scen.upper(), "SKIPPING TO THE NEXT STAGE")
            if scen!="intermediate":
                continue

        num_envs = 1#6
        print("INITIALIZING", num_envs, scen.upper(), "ENVIRONMENTS...", end="")
        if num_envs > 1:
            env = SubprocVecEnv(
                [lambda: Monitor(gym.make("PathColav3d-v0", scenario=scen), agents_dir, allow_early_resets=True)
                for i in range(num_envs)]
            )
        else:
            env = DummyVecEnv(
                [lambda: Monitor(gym.make("PathColav3d-v0", scenario=scen), agents_dir,allow_early_resets=True)]
            )
        print("DONE")
        print("INITIALIZING AGENT...", end="")
        if scen == "line":
            #agent = PPO('MlpPolicy', env, **hyperparams,policy_kwargs=policy_kwargs,seed=seed)
            agent = PPO('MultiInputPolicy', env, **hyperparams,policy_kwargs=policy_kwargs,seed=seed)
            #onnxable_model = OnnxablePolicy(model.policy.mlp_extractor, model.policy.action_net, model.policy.value_net)
            #observation_size=242
        
            #dummy_input={'perception':torch.randn(1,1,15,15).cuda(), 'navigation': torch.randn(1,1,16).cuda()} 
            #torch.onnx.export(agent.policy, dummy_input, "my_ppo_model.onnx", opset_version=9)
            #torch.onnx.export(agent.policy, dummy_input, "my_ppo_model.onnx", opset_version=9)
        elif scen=="intermediate":
            continual_model = os.path.join(experiment_dir, scenarios[i], "agents", "last_model.pkl")
            agent = PPO.load(continual_model, _init_setup_model=True, env=env, **hyperparams)
        else:
            continual_model = os.path.join(experiment_dir, scenarios[i-1], "agents", "last_model.pkl")
            agent = PPO.load(continual_model, _init_setup_model=True, env=env, **hyperparams)
        print("DONE")

        #if i<3:
        #if scen=='intermediate':
        #    best_mean_reward, n_steps, timesteps = -np.inf, 0, int(30e6)# + i*150e3)
        #else:
        #    
        best_mean_reward, n_steps, timesteps = -np.inf, 0, int(100e6)# + i*150e3)
        print("TRAINING FOR", timesteps, "TIMESTEPS")
        agent.learn(total_timesteps=timesteps, tb_log_name="PPO2",callback=StatsCallback())
        print("FINISHED TRAINING AGENT IN", scen.upper())
        save_path = os.path.join(agents_dir, "last_model.pkl")
        agent.save(save_path)
        print("SAVE SUCCESSFUL")
