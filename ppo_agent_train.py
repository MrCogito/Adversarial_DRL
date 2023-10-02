print("Script started")
from pettingzoo.atari import pong_v3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

# Load the environment
env = pong_v3.parallel_env()

# Check the observation and action spaces using the recommended functions
print("Observation space: ", env.observation_space('first_0'))
print("Action space: ", env.action_space('first_0'))

# Reset the environment and check the observation size
observations, _ = env.reset()
first_agent_observation = observations['first_0']
print("Observation size: ", first_agent_observation.shape)


# Correctly calculate the input dimension
input_dim = np.prod(env.observation_space('first_0').shape)

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
        # Ensure the state is correctly flattened
        state = torch.from_numpy(state).float().view(1, -1)
        probs, value = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        return action.item(), log_prob, value.squeeze()

def train_ppo(agent, env, optimizer, epochs=100, gamma=0.99):
    for epoch in range(epochs):
        observations, _ = env.reset()
        done = {'first_0': False, 'second_0': False}
        total_reward = {'first_0': 0, 'second_0': 0}
        epoch_policy_loss = 0
        epoch_value_loss = 0
        step_counter = 0
        if epoch % 10 == 0:  # Save the model every 10 epochs
            print(f"Epoch: {epoch}, Total Reward: {total_reward}, Policy Loss: {epoch_policy_loss}, Value Loss: {epoch_value_loss}")
            torch.save(agent.state_dict(), f'/zhome/59/9/198225/Adversarial_DRL/agents/trained_agent_epoch_{epoch}.pth')
        while not all(done.values()):
            step_counter += 1  # Increment step counter
            if step_counter % 100 == 0:  # Print progress every 'print_interval' steps
                print(f"Epoch: {epoch}, Step: {step_counter}")
            actions = {}
            log_probs = {}
            values = {}
            rewards = {}
            for agent_name, obs in observations.items():
                action, log_prob, value = agent.act(obs.flatten())
                actions[agent_name] = action
                log_probs[agent_name] = log_prob
                values[agent_name] = value
                
            next_observations, rewards, done, _, _ = env.step(actions)
            
            # Update the agent
            for agent_name, reward in rewards.items():
                total_reward[agent_name] += reward
                next_value = 0 if done[agent_name] else agent.value(torch.from_numpy(next_observations[agent_name]).float().view(1, -1)).item()
                advantage = torch.tensor(reward + gamma * next_value - values[agent_name].item(), dtype=torch.float32, requires_grad=True)
                policy_loss = -log_probs[agent_name].detach() * advantage
                value_loss = advantage.pow(2)
                loss = policy_loss + value_loss
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            observations = next_observations
        
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Total Reward: {total_reward}, Policy Loss: {epoch_policy_loss}, Value Loss: {epoch_value_loss}")


# Initialize the PPO agent with the corrected input dimension
output_dim = env.action_space('first_0').n
input_dim = np.prod(env.observation_space('first_0').shape)
agent = PPOAgent(input_dim, output_dim)
optimizer = optim.Adam(agent.parameters(), lr=1e-2)

# Train the agent
train_ppo(agent, env, optimizer)
torch.save(agent.state_dict(), '/zhome/59/9/198225/Adversarial_DRL/agents/trained_agent.pth')
