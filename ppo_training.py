import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import wandb

class PPOAgent(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPOAgent, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Softmax(dim=-1)
        )
        
        self.value = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, state):
        return self.policy(state), self.value(state)
    
    def act(self, state):
        state = torch.from_numpy(state).float().view(1, -1)
        probs, value = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        entropy = m.entropy()
        return action.item(), log_prob, value.squeeze(), entropy

def train_ppo(agent, env, optimizer, experiment_name, epochs, gamma, save_folder):
    # Initialize wandb
    wandb.init(project=experiment_name)

    # Paths for saving metrics and model
    model_save_path = lambda epoch: os.path.join(save_folder, f'trained_agent_epoch_{epoch}.pth')
    final_model_save_path = os.path.join(save_folder, 'trained_agent.pth')

    os.makedirs(save_folder, exist_ok=True)

    # Training loop
    for epoch in range(epochs):
        observations, _ = env.reset()
        done = {'first_0': False, 'second_0': False}
        total_reward = {'first_0': 0, 'second_0': 0}
        epoch_policy_loss = 0
        epoch_value_loss = 0
        epoch_entropy = 0

        while not all(done.values()):
            actions = {}
            log_probs = {}
            values = {}
            rewards = {}
            entropy_sum = 0

            for agent_name, obs in observations.items():
                action, log_prob, value, entropy = agent.act(obs.flatten())
                actions[agent_name] = action
                log_probs[agent_name] = log_prob
                values[agent_name] = value
                entropy_sum += entropy

            next_observations, rewards, done, _ = env.step(actions)

            for agent_name, reward in rewards.items():
                total_reward[agent_name] += reward
                next_value = 0 if done[agent_name] else agent.value(torch.from_numpy(next_observations[agent_name]).float().view(1, -1)).item()
                advantage = reward + gamma * next_value - values[agent_name].item()
                policy_loss = -log_probs[agent_name] * advantage
                value_loss = advantage ** 2
                loss = policy_loss + value_loss

                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_entropy += entropy_sum

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            observations = next_observations

        # Save agent every 3 epochs
        if epoch % 3 == 0:
            torch.save(agent.state_dict(), model_save_path(epoch))

        # Log metrics to wandb
        metrics = {
            'policy_loss': epoch_policy_loss,
            'value_loss': epoch_value_loss,
            'total_reward_first': total_reward['first_0'],
            'total_reward_second': total_reward['second_0'],
            'entropy': epoch_entropy
        }
        wandb.log(metrics, step=epoch)

    # Final save after training completion
    torch.save(agent.state_dict(), final_model_save_path)
