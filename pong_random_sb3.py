from stable_baselines3 import PPO
from pettingzoo.atari import pong_v3
import supersuit as ss
import numpy as np
import random

# Load the pretrained model
model_path = '/home/karol/Adversarial_DRL/rl-baselines3-zoo/logs/ppo/PongNoFrameskip-v4_1/PongNoFrameskip-v4.zip'
model = PPO.load(model_path)

# Initialize the environment
#env = pong_v3.parallel_env(obs_type='rgb_image', render_mode='human')
env = pong_v3.parallel_env(obs_type='rgb_image')

# Preprocess the environment
env = ss.color_reduction_v0(env, mode='B')  # Convert to grayscale
env = ss.resize_v1(env, x_size=84, y_size=84)  # Resize the observations
env = ss.frame_stack_v1(env, 4)  # Stack the last 4 frames

# Reset environment
observations, infos = env.reset()  # Directly use the returned dictionary

num_games = 1
wins_first_0 = 0
wins_second_0 = 0

for game in range(num_games):
    score_first_0 = 0
    while env.agents:
        actions = {}
        for agent in env.agents:
            if agent == 'first_0':
                obs = observations[agent]  # Directly access the observation
                action, _states = model.predict(obs, deterministic=True)
                actions[agent] = action
            elif agent == 'second_0':
                actions[agent] = random.choice(list(range(6)))
            else:
                raise ValueError(f"Unknown agent: {agent}")

        # Step the environment and use the returned observations directly
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        score_first_0 += rewards['first_0']
        if any(terminations.values()) or any(truncations.values()):
            break

    # Reset environment for the next game
    observations, infos = env.reset()

    if score_first_0 > 0:
        wins_first_0 += 1
    else:
        wins_second_0 += 1

env.close()

# Print the number of wins for each agent
print(f"first_0 won {wins_first_0} times.")
print(f"second_0 won {wins_second_0} times.")
