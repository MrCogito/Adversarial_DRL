from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from pettingzoo.atari import pong_v3
import supersuit as ss
import matplotlib.pyplot as plt
import numpy as np

# Initialize SB3 environment
def init_sb3_env():
    env_id = "PongNoFrameskip-v4"
    env = make_atari_env(env_id, n_envs=1, seed=0)
    env = VecFrameStack(env, n_stack=4)
    return env

# Initialize and preprocess PettingZoo environment
def init_preprocessed_pettingzoo_env():
    env = pong_v3.parallel_env(obs_type='rgb_image')
    env = ss.color_reduction_v0(env, mode='B')  # Grayscale
    env = ss.resize_v1(env, x_size=84, y_size=84)  # Resize
    env = ss.frame_stack_v1(env, 4)  # Frame stack
    return env

# Function to save a frame
def save_frame(frame, filename):
    plt.imsave(filename, frame, cmap='gray')

def save_obs_to_csv(observation, filename):
    # Flatten the observation to save it as a 2D array (1D array would be saved as a single row)
    np.savetxt(filename, observation.reshape(1, -1), delimiter=",")

# Function to step through an environment and print + save first observation
def inspect_and_save_first_frame(env, env_name):
    obs = env.reset()
    # Take a single step to get an observation
    if env_name == "SB3":
        # Take a step to get an observation for SB3 environment
        obs, _, _, _ = env.step([env.action_space.sample()])
        full_obs = obs[0] # Take the first frame of the stacked frames
        
    else:
        # For PettingZoo, simulate both agents taking a step
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)
        full_obs = obs['first_0']  # Assuming 'first_0' is still in the game
      
        
    print(f"{env_name} Observation shape: {full_obs.shape}")
    print(f"{env_name} Observation size: {full_obs.size}")
    print(f"{env_name} Value range: min {np.min(full_obs)}, max {np.max(full_obs)}")
    
    # Save the frame
    save_frame(full_obs, f"{env_name}_first_frame.png")
    save_obs_to_csv(full_obs, f"{env_name}_observation_values.csv")

# Initialize environments
sb3_env = init_sb3_env()
pettingzoo_env = init_preprocessed_pettingzoo_env()

# Inspect and save first frame for SB3 environment
inspect_and_save_first_frame(sb3_env, "SB3")

# Inspect and save first frame for PettingZoo environment
inspect_and_save_first_frame(pettingzoo_env, "PettingZoo")
