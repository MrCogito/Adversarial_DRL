from stable_baselines3 import PPO
from pettingzoo.atari import pong_v3
import supersuit as ss
import random

# Load the pretrained model
model_path = '/home/karol/Adversarial_DRL/rl-baselines3-zoo/logs/ppo/PongNoFrameskip-v4_1/PongNoFrameskip-v4.zip'
model = PPO.load(model_path)

# Initialize and preprocess the environment
env = pong_v3.parallel_env(obs_type='rgb_image', render_mode='human')
env = ss.color_reduction_v0(env, mode='B')
env = ss.resize_v1(env, x_size=84, y_size=84)
env = ss.frame_stack_v1(env, 4)

# Reset environment
observations, infos = env.reset()

num_games = 1
wins_first_0 = 0
wins_second_0 = 0

for game in range(num_games):
    score_first_0 = 0
    while env.agents:
        actions = {}
        for agent in env.agents:
            if agent == 'first_0':
                obs = observations[agent]
                action, _states = model.predict(obs, deterministic=True)
                actions[agent] = action
                # Debugging: Log observed actions and observations
                print(f"Agent {agent}, Action: {action}")
            elif agent == 'second_0':
                actions[agent] = random.choice(list(range(env.action_space(agent).n)))
        observations, rewards, terminations, truncations, infos = env.step(actions)
        score_first_0 += rewards['first_0']
        # Debugging: Log rewards and game termination info
        print(f"Rewards: {rewards}, Terminations: {terminations}, Truncations: {truncations}")
        if any(terminations.values()) or any(truncations.values()):
            break
    observations, infos = env.reset()
    if score_first_0 > 0:
        wins_first_0 += 1
    else:
        wins_second_0 += 1

env.close()

print(f"first_0 won {wins_first_0} times.")
print(f"second_0 won {wins_second_0} times.")
