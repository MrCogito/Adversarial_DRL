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

def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def train_ppo(agent, env, optimizer, experiment_name, epochs, gamma, save_folder, batch_size):
    # Initialize wandb
    wandb.init(project="adversarial_dlr", name=experiment_name)

    # Paths for saving metrics and model
    model_save_path = lambda epoch: os.path.join(save_folder, f'trained_agent_epoch_{epoch}.pth')
    final_model_save_path = os.path.join(save_folder, 'trained_agent.pth')

    os.makedirs(save_folder, exist_ok=True)

    # Training loop
    for epoch in range(epochs):
        observations, _ = env.reset()
        done = False
        total_reward = 0
        batch_states, batch_actions, batch_log_probs, batch_rewards, batch_values, batch_dones = [], [], [], [], [], []
        batch_entropy = 0
        total_steps = 0

        while total_steps < batch_size:
            state = observations.flatten()
            action, log_prob, value, entropy = agent.act(state)

            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            batch_states.append(state)
            batch_actions.append(action)
            batch_log_probs.append(log_prob)
            batch_values.append(value)
            batch_rewards.append(reward)
            batch_dones.append(done)
            batch_entropy += entropy

            total_steps += 1
            observations = next_state

            if done:
                observations, _ = env.reset()

        # Convert lists to tensors
        batch_states = torch.stack(batch_states)
        batch_actions = torch.tensor(batch_actions, dtype=torch.float)
        batch_log_probs = torch.stack(batch_log_probs)
        batch_values = torch.stack(batch_values)
        batch_rewards = torch.tensor(batch_rewards, dtype=torch.float)
        batch_dones = torch.tensor(batch_dones, dtype=torch.float)
        
        # Compute returns and advantages
        with torch.no_grad():
            next_value = agent.value(batch_states[-1]).squeeze()
        returns = compute_gae(next_value, batch_rewards, batch_dones, batch_values, gamma)
        returns = torch.tensor(returns, dtype=torch.float)
        advantages = returns - batch_values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute policy loss and value loss
        ratio = torch.exp(batch_log_probs - agent.policy(batch_states).log_prob(batch_actions))
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = 0.5 * (returns - agent.value(batch_states).squeeze()).pow(2).mean()

        # Total loss
        loss = policy_loss + value_loss - 0.01 * batch_entropy.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save agent every few epochs
        if epoch % 3 == 0:
            torch.save(agent.state_dict(), model_save_path(epoch))

        

        
        # Calculate additional metrics
        policy_loss_val = policy_loss.item()
        value_loss_val = value_loss.item()
        entropy_val = batch_entropy.mean().item()
        average_reward = total_reward / total_steps
        average_advantage = advantages.mean().item()
        episode_length = total_steps
       

        # Log metrics to wandb
        wandb.log({
            'total_reward': total_reward,
            'average_reward': average_reward,
            'policy_loss': policy_loss_val,
            'value_loss': value_loss_val,
            'entropy': entropy_val,
            'average_advantage': average_advantage,
            'episode_length': episode_length,
            
        }, step=epoch) 

    # Final save after training completion
    torch.save(agent.state_dict(), final_model_save_path)


