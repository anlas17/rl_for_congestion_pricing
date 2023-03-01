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
    # Set up environment
    env = make_env(10)() # Seed as argument
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=30)
    env = VecNormalize(env, norm_obs=True, norm_reward=True) 

    # Set hyperparameters and model
    policy_kwargs = dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])], activation_fn=th.nn.Tanh) 
    #model = PPO("MlpPolicy", env, learning_rate=0.0001, n_steps=150, verbose=1, batch_size=150, target_kl=0.05, n_epochs=80, gae_lambda=0.97, seed=0, policy_kwargs=policy_kwargs, tensorboard_log="./tensorboards/PPO/")
    model = PPO.load("./models/abs/225ppo(abs)_tollprice")
    model.set_env(env)

    # Train model
    model.learn(total_timesteps=2250, tb_log_name="train_run_abs", reset_num_timesteps=False)

    # Save stats and model
    np.save("./stats/training/300ppo(abs)_tt.npy", env.env_method('get_tt')[0])
    np.save("./stats/training/300ppo(abs)_sw.npy", env.env_method('get_sw')[0])
    np.save("./stats/training/300ppo(abs)_cs.npy", env.env_method('get_cs')[0])
    np.save("./stats/training/300ppo(abs)_gc.npy", env.env_method('get_gc')[0])
    np.save("./stats/training/300ppo(abs)_tc.npy", env.env_method('get_tc')[0])
    np.save("./stats/training/300ppo(abs)_ttcs.npy", env.env_method('get_ttcs')[0])

    model.save("./models/300ppo(abs)_tollprice")

if __name__ == "__main__":
    main()