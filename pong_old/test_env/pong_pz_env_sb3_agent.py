from stable_baselines3 import PPO
from pettingzoo.atari import pong_v3
import supersuit as ss
import random
import numpy as np
import matplotlib.pyplot as plt


# Load the pretrained model
model_path = '/home/karol/Adversarial_DRL/rl-baselines3-zoo/logs/ppo/PongNoFrameskip-v4_1/PongNoFrameskip-v4.zip'
model = PPO.load(model_path)

# Initialize and preprocess the environment
env = pong_v3.parallel_env(obs_type='rgb_image', render_mode='human')
#env = pong_v3.parallel_env(obs_type='rgb_image')
env = ss.color_reduction_v0(env, mode='B')
env = ss.resize_v1(env, x_size=84, y_size=84)
env = ss.frame_stack_v1(env, 4)
print(f"PettingZoo Action Space: {env.action_space('first_0').n}")



# Reset environment
observations, infos = env.reset()
print(f"Environment Observation Space: {env.observation_space('first_0').shape}")
print(f"Environment Action Space: {env.action_space('first_0').n}")

num_games = 1
wins_first_0 = 0
wins_second_0 = 0

# Initialize debug variables
actions_taken = {agent: [] for agent in env.agents}
episode_rewards = {agent: 0 for agent in env.agents}
episode_lengths = {agent: 0 for agent in env.agents}

for game in range(num_games):
    observations, infos = env.reset()
    while env.agents:
        actions = {}
        for agent in env.agents:
            obs = observations[agent]
            
            if agent == 'first_0':
                    
                # Since the observation shape is already in the correct format (84, 84, 4),
                # there's no need to transpose it for visualization or saving.
                # plt.imshow(obs)
                # plt.show()
                # # Save the image directly without transposing
                # plt.imsave(f'observation_{game}_{agent}.png', obs)
                
                action, _states = model.predict(obs, deterministic=True)
                #print(f"Predicted Action: {action}")

            else:
                action = random.choice(list(range(env.action_space(agent).n)))
            actions[agent] = action

            # Log actions for debugging
            actions_taken[agent].append(action)

        observations, rewards, terminations, truncations, infos = env.step(actions)
        for agent, reward in rewards.items():
            episode_rewards[agent] += reward
            episode_lengths[agent] += 1

        if any(terminations.values()) or any(truncations.values()):
            break

    # Debug log at the end of each game
    print(f"Game {game+1} Summary")
    for agent in env.agents:
        print(f"{agent} - Avg Action: {np.mean(actions_taken[agent]):.2f}, Total Reward: {episode_rewards[agent]}, Episode Length: {episode_lengths[agent]}")
        # Reset debug variables for the next episode
        actions_taken[agent] = []
        episode_rewards[agent] = 0
        episode_lengths[agent] = 0
    break
env.close()

print(f"first_0 won {wins_first_0} times.")
print(f"second_0 won {wins_second_0} times.")
