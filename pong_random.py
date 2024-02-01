from stable_baselines3 import PPO
from pettingzoo.atari import pong_v3
import supersuit as ss
import numpy as np
import random

# Load the pretrained model
model_path = '/home/karol/Adversarial_DRL/rl-baselines3-zoo/logs/ppo/PongNoFrameskip-v4_1/PongNoFrameskip-v4.zip'
model = PPO.load(model_path)

# Initialize the environment
env = pong_v3.parallel_env(obs_type='rgb_image', render_mode='human')

# Reset environment and unpack observations and additional info
observations, info = env.reset()  # Assuming env.reset() returns observations directly

num_games = 1
wins_first_0 = 0
wins_second_0 = 0

for game in range(num_games):
    score_first_0 = 0
    while env.agents:
        actions = {}
        for agent in env.agents:
            if agent == 'first_0':
                obs = observations[agent]  # Correctly access observations
                action, _states = model.predict(obs, deterministic=True)
                actions[agent] = action
            elif agent == 'second_0':
                actions[agent] = random.choice(list(range(6)))  # Random actions for the second agent
            else:
                raise ValueError(f"Unknown agent: {agent}")

        # Step the environment and properly unpack observations
        temp, rewards, terminations, truncations, infos = env.step(actions)
        if isinstance(temp, tuple):  # Proper check for unpacking
            observations, _ = temp  # If env.step() returns a tuple
        else:
            observations = temp  # Direct assignment if not a tuple
        
        score_first_0 += rewards['first_0']
        if any(terminations.values()) or any(truncations.values()):
            break

    temp = env.reset()  # Reset for the next game
    if isinstance(temp, tuple):  # Check and unpack
        observations, _ = temp
    else:
        observations = temp

    # Determine game outcome
    if score_first_0 > 0:
        wins_first_0 += 1
    else:
        wins_second_0 += 1

env.close()

# Print the number of wins for each agent
print(f"first_0 won {wins_first_0} times.")
print(f"second_0 won {wins_second_0} times.")
