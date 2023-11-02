print("Script started")
from pettingzoo.atari import pong_v3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import os
import mlflow

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
        # Ensure the state is correctly flattened
        state = torch.from_numpy(state).float().view(1, -1)
        probs, value = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        entropy = m.entropy()
        return action.item(), log_prob, value.squeeze(), entropy


def train_ppo(agent, env, optimizer, experiment_name="PPO_Atari_Pong", epochs=100, gamma=0.99, save_folder='/zhome/59/9/198225/Adversarial_DRL/agents/'):
    # Set up MLflow
    mlflow.set_experiment(experiment_name)

    # Paths for saving metrics and model
    metrics_file_path = os.path.join(save_folder, 'training_metrics.txt')
    model_save_path = lambda epoch: os.path.join(save_folder, f'trained_agent_epoch_{epoch}.pth')
    final_model_save_path = os.path.join(save_folder, 'trained_agent.pth')

    # Create directory if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)

    with mlflow.start_run():
        # Log parameters (optional)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("gamma", gamma)
        
        with open(metrics_file_path, "w") as file:
            file.write("Epoch,Policy Loss,Value Loss,Total Reward First,Total Reward Second,Entropy\n")

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

                # Log metrics to file and MLflow
                metrics = {
                    'policy_loss': epoch_policy_loss,
                    'value_loss': epoch_value_loss,
                    'total_reward_first': total_reward['first_0'],
                    'total_reward_second': total_reward['second_0'],
                    'entropy': epoch_entropy
                }
                file.write(f"{epoch},{epoch_policy_loss},{epoch_value_loss},{total_reward['first_0']},{total_reward['second_0']},{epoch_entropy}\n")
                mlflow.log_metrics(metrics, step=epoch)

                # Optional: Print the metrics
                print(f"Epoch: {epoch}, Metrics: {metrics}")

    # Final save after training completion
    torch.save(agent.state_dict(), final_model_save_path)



# Initialize the PPO agent with the corrected input dimension
output_dim = env.action_space('first_0').n
input_dim = np.prod(env.observation_space('first_0').shape)
agent = PPOAgent(input_dim, output_dim)
optimizer = optim.Adam(agent.parameters(), lr=3e-4)

# Define the folder path for saving models and metrics
save_folder = '/zhome/59/9/198225/Adversarial_DRL/agents/'

# Train the agent using the updated train_ppo function
train_ppo(
    agent=agent,
    env=env,
    optimizer=optimizer,
    experiment_name="PPO_Atari_Pong",
    epochs=100,  # Set the number of epochs
    gamma=0.99,  # Set the discount factor
    save_folder=save_folder
)
