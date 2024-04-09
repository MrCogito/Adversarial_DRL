import sys
print(sys.executable)
from pettingzoo.atari import pong_v3
import torch
import torch.nn as nn
import numpy as np
import random
from torch.distributions import Categorical


# Define the PPOAgent class
class PPOAgent(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPOAgent, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)
        )

        self.value = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.policy(state), self.value(state)

    def act(self, state):
        state = torch.from_numpy(state).float().view(1, -1)
        probs, value = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        return action.item(), log_prob, value.squeeze()

# Initialize the environment
env = pong_v3.parallel_env(render_mode="human")

# Initialize the PPO agent with the corrected input dimension
output_dim = env.action_space('first_0').n
input_dim = np.prod(env.observation_space('first_0').shape)
agent = PPOAgent(input_dim, output_dim)

# Load the trained model
agent.load_state_dict(torch.load('trained_agent_epoch_10.pth'))
agent.eval()  # Set the agent to evaluation mode

num_games = 1
wins_first_0 = 0
wins_second_0 = 0

for game in range(num_games):
    observations, infos = env.reset()
    score_first_0 = 0

    while env.agents:
        actions = {}
        for agent_name in env.agents:
            if agent_name == 'first_0':
                action, _, _ = agent.act(observations[agent_name].flatten())
                actions[agent_name] = action
            elif agent_name == 'second_0':
                actions[agent_name] = random.choice(list(range(6)))  # Keep the random action for the second agent
            else:
                raise ValueError(f"Unknown agent: {agent_name}")

        observations, rewards, terminations, truncations, infos = env.step(actions)
        score_first_0 += rewards['first_0']

    env.close()

    if score_first_0 > 0:
        score_first_0_adjusted = 21
        score_second_0_adjusted = 21 - score_first_0
    else:
        score_second_0_adjusted = 21
        score_first_0_adjusted = 21 + score_first_0

    if score_first_0_adjusted > score_second_0_adjusted:
        wins_first_0 += 1
    elif score_first_0_adjusted < score_second_0_adjusted:
        wins_second_0 += 1

# Print the number of wins for each agent
print(f"first_0 won {wins_first_0} times.")
print(f"second_0 won {wins_second_0} times.")
