from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
import os
import numpy as np
import matplotlib.pyplot as plt

# Assuming you have the correct path to your model
model_path = '/home/karol/Adversarial_DRL/rl-baselines3-zoo/logs/ppo/PongNoFrameskip-v4_1/PongNoFrameskip-v4.zip'

# Create the environment correctly
env_id = "PongNoFrameskip-v4"
env = make_atari_env(env_id, n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4)  # Correctly applying VecFrameStack

# Load the pre-trained model
model = PPO.load(model_path, env=env)

def play_game_and_inspect_obs(env, model):
    obs = env.reset()
    steps = 5

    # Play for a specified number of steps
    for step in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        if done.any():
            print("Episode finished early")
            break

    # After 5 steps, inspect the observation
    if not done.any():
        # The observation here includes all stacked frames
        full_obs = obs[0]  # Assuming single environment

        # 1. Print the shape and size of the entire observation
        print(f"Observation shape: {full_obs.shape}")
        print(f"Observation size: {full_obs.size}")

        # 2. Save the values of the entire observation into a file
        np.savetxt("full_observation_values.csv", full_obs.reshape(-1), delimiter=",")

        # 3. Print the value range of the entire observation
        print(f"Observation value range: min {np.min(full_obs)}, max {np.max(full_obs)}")

        # 4. Save each frame within the stacked observation as a separate PNG
        for i in range(full_obs.shape[-1]):
            plt.imsave(f"observation_frame_{i+1}.png", full_obs[:, :, i], cmap='gray')

    env.close()

# Execute the function
play_game_and_inspect_obs(env, model)