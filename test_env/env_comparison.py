from stable_baselines3.common.env_util import make_atari_env
from pettingzoo.utils import BaseWrapper
from stable_baselines3.common.vec_env import VecFrameStack
from pettingzoo.utils.conversions import parallel_wrapper_fn
from pettingzoo.atari import pong_v3
import supersuit as ss
import matplotlib.pyplot as plt
import numpy as np
import gym

import cv2  # Make sure to install opencv-python with pip install opencv-python

class WarpFrameGrayScale(BaseWrapper):
    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self.width = width
        self.height = height

    def reset(self, **kwargs):
        super().reset(**kwargs)  # Call the parent reset, but don't try to process observation here
        # In AEC, after reset, you should obtain observation by calling observe() on the active agent
        # Since reset doesn't return an observation directly, there's no observation to process here

    def step(self, action):
        super().step(action)
        # Similarly, step doesn't return observation directly in AEC environments
        # Observation must be obtained using observe() after this call

    def observe(self, agent):
        # Adding a print statement to check the shape of processed frame
        processed_frame = super().observe(agent)
        processed_frame = self._process_frame(processed_frame)
        return processed_frame


    def _process_frame(self, frame):
        if frame is not None:
            # Ensure the source color space is correctly specified here
            # For a typical OpenCV image this would be BGR2GRAY
            # For images that are already in RGB format (as most image files are), this should be RGB2GRAY
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            resized_frame = cv2.resize(gray_frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
            return np.expand_dims(resized_frame, axis=-1)

class CustomFrameStack(BaseWrapper):
    def __init__(self, env, stack_size=4):
        super().__init__(env)
        self.stack_size = stack_size
        # Initialize the stack with None; it will be properly initialized after the first reset
        self.stack = None

    def reset(self, **kwargs):
        super().reset(**kwargs)
        # Obtain an initial observation to initialize the stack
        first_obs = self.env.observe(self.env.agent_selection)
        # Initialize the stack with duplicates of the first observation
        self.stack = np.repeat(first_obs[np.newaxis, ...], self.stack_size, axis=0)

    def observe(self, agent):
        obs = super().observe(agent)
        # Add the new observation to the stack and remove the oldest one
        self.stack = np.roll(self.stack, shift=-1, axis=0)
        self.stack[-1, ...] = obs
        # Return the stacked observations as a single array
        return np.concatenate(self.stack, axis=-1)

    def step(self, action):
        # Ensure the base environment's step is called to properly update the state
        super().step(action)


# Initialize SB3 environment
def init_sb3_env():
    env_id = "PongNoFrameskip-v4"
    env = make_atari_env(env_id, n_envs=1, seed=0)
    env = VecFrameStack(env, n_stack=4)
    return env

def init_preprocessed_pettingzoo_env():
    env = pong_v3.env(obs_type='rgb_image')
    env = ss.resize_v1(env, x_size=84, y_size=84)

    # Apply the grayscale conversion wrapper
    env = WarpFrameGrayScale(env)

    # Debug: Check the observation by manually invoking observe() for an agent after reset
    env.reset()


    # If the shape is correct (84, 84, 1), proceed to apply frame stacking
    env = CustomFrameStack(env, stack_size=4)
    


    return env




# Function to save a frame
def save_frame(frame, filename):
    #if len(frame.shape) == 3 and frame.shape[2] == 1:  # If grayscale, remove the channel dimension
    #    frame = frame.squeeze()
    plt.imsave(filename, frame, cmap="gray")

def save_obs_to_csv(observation, filename):
    np.savetxt(filename, observation.reshape(1, -1), delimiter=",")

# Function to step through an environment and print + save first observation
# Function to step through an environment and print + save first observation
def inspect_and_save_first_frame(env, env_name):
    obs = env.reset()
    if env_name == "SB3":
        # For SB3 environments, sample an action for each environment within the vectorized setup.
        actions = [env.action_space.sample() for _ in range(env.num_envs)]
        obs, _, _, _ = env.step(actions)
        # Assuming you want to process the first environment's observation
        full_obs = obs[0]
    else:
        # Corrected for PettingZoo AEC environments
        env.reset()
        for agent in env.agent_iter():
            if agent == 'first_0':  # Only proceed if it's the agent we're interested in
                obs, reward, done, truncation, info = env.last()  # Correctly unpack all return values
                if not done:
                    action = 0  # "No operation" action, or whatever default you wish
                    env.step(action)
                else:
                    env.step(None)  # Pass None if the agent is done
                full_obs = obs
                break  # Exit after processing the first agent's data
    
    
    print(f"{env_name} Observation shape: {full_obs[:, :, 0].shape}")
    print(f"{env_name} Observation size: {full_obs.size}")
    print(f"{env_name} Value range: min {np.min(full_obs)}, max {np.max(full_obs)}")
    
    save_frame(full_obs, f"{env_name}_first_frameXD.png")
    save_obs_to_csv(full_obs, f"{env_name}_observation_values.csv")



# Initialize environments
sb3_env = init_sb3_env()
pettingzoo_env = init_preprocessed_pettingzoo_env()

# Inspect and save first frame for SB3 environment
inspect_and_save_first_frame(sb3_env, "SB3")

# Inspect and save first frame for PettingZoo environment
inspect_and_save_first_frame(pettingzoo_env, "PettingZoo")
