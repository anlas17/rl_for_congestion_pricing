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
    # Setup environment
    eval_env = make_env(10)() # Seed as argument
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecFrameStack(eval_env, n_stack=30) 
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)

    # Load trained model
    #model = PPO.load("./models/225ppo_tollprice")
    #model.set_env(eval_env)
    model = TD3.load("./models/n,tp/300td3tp_tollprice")
    model.set_env(eval_env)

    obs = eval_env.reset()

    # Test that model is able to take actions
    for i in range(30):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = eval_env.step(action)

    obs = eval_env.reset()
    
    # Evaluate trained model
    ep_rews, ep_lens = evaluate_policy(model, eval_env, n_eval_episodes=10, return_episode_rewards=True, deterministic=True)
    ep_rews = np.array(ep_rews)
    mean_reward = np.mean(ep_rews)
    std_reward = np.std(ep_rews)
    print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    np.save("./stats/new_test/300td3tp_rew.npy", ep_rews)

    # Save test stats
    print("Saving stats...")
    np.save("./stats/new_test/300td3tp_tt.npy", eval_env.env_method('get_tt')[0])
    np.save("./stats/new_test/300td3tp_sw.npy", eval_env.env_method('get_sw')[0])
    np.save("./stats/new_test/300td3tp_cs.npy", eval_env.env_method('get_cs')[0])
    np.save("./stats/new_test/300td3tp_gc.npy", eval_env.env_method('get_gc')[0])
    np.save("./stats/new_test/300td3tp_tc.npy", eval_env.env_method('get_tc')[0])
    np.save("./stats/new_test/300td3tp_ttcs.npy", eval_env.env_method('get_ttcs')[0])

if __name__ == "__main__":
    main()