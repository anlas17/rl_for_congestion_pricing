#IMPORTS
import random
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch as th
from stable_baselines3 import PPO, DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_util import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

from oop_simulation_script import Simulation, CommuteEnv
    
def make_env(seed):
    def _init():
        env = Monitor(CommuteEnv())
        env.seed(seed)
        return env
    return _init

def main():

    env = make_env(10)() # Seed as argument
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=30)
    env = VecNormalize(env, norm_obs=True, norm_reward=True) 

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1*np.ones(n_actions))

    # Setting up model and hyperparameters
    policy_kwargs = dict(net_arch=dict(pi=[64, 64], qf=[64, 64]))
    #model = TD3("MlpPolicy", env, learning_rate=0.0001, device='cuda', policy_kwargs=policy_kwargs, buffer_size=150, learning_starts=30, batch_size=30, action_noise=action_noise, verbose=1, tensorboard_log="./tensorboards/TD3/")
    model = TD3.load("./models/225td3_tollprice", device='cuda')
    model.set_env(env)

    # Training the model
    model.learn(total_timesteps=2250, tb_log_name="train_run", reset_num_timesteps=False)

    # Saving training statistics
    np.save("./stats/training/300td3_tt.npy", env.env_method('get_tt')[0])
    np.save("./stats/training/300td3_sw.npy", env.env_method('get_sw')[0])
    np.save("./stats/training/300td3_cs.npy", env.env_method('get_cs')[0])
    np.save("./stats/training/300td3_gc.npy", env.env_method('get_gc')[0])
    np.save("./stats/training/300td3_tc.npy", env.env_method('get_tc')[0])
    np.save("./stats/training/300td3_ttcs.npy", env.env_method('get_ttcs')[0])

    # Saving trained model
    model.save("./models/300td3_tollprice")

if __name__ == "__main__":
    main()