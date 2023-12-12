import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import torch.nn.init as init
import wandb

class PPOAgent(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPOAgent, self).__init__()
        # Moderately increasing the number of neurons in each layer
        self.policy = nn.Sequential(
            nn.Linear(input_dim, 192),  # Increased from 128 to 192
            nn.ReLU(),
            nn.Linear(192, 384),       # Increased from 256 to 384
            nn.ReLU(),
            nn.Linear(384, output_dim),
            nn.Softmax(dim=-1)
        )
        
        self.value = nn.Sequential(
            nn.Linear(input_dim, 192),  # Increased from 128 to 192
            nn.ReLU(),
            nn.Linear(192, 384),       # Increased from 256 to 384
            nn.ReLU(),
            nn.Linear(384, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.fill_(0.01)
    
    def forward(self, state):
        return self.policy(state), self.value(state)
    
    def act(self, state, device):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        elif isinstance(state, torch.Tensor):
            state = state.float()
        else:
            raise TypeError("Expected state to be np.ndarray or torch.Tensor")

        state = state.view(1, -1).to(device)
        probs, value = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        entropy = m.entropy()
        return action.item(), log_prob, value.squeeze(), entropy

def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = torch.cat([values, next_value.unsqueeze(0)], dim=0)
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def train_ppo(agent, opponent_agent, env, optimizer, opponent_optimizer, name, epochs, gamma, entropy_coeff, save_folder, batch_size, device):
    wandb.init(project="adversarial_dlr", name=name)
    
    model_save_path = lambda epoch, agent_name: os.path.join(save_folder, f'{name}_{agent_name}_trained_agent_epoch_{epoch}.pth')
    final_model_save_path = lambda agent_name: os.path.join(save_folder, f'{name}_{agent_name}_trained_agent.pth')

    os.makedirs(save_folder, exist_ok=True)

    best_reward = -float('inf')
    best_state_dict = agent.state_dict()

    for epoch in range(epochs):
        print(f"Starting Epoch {epoch+1}/{epochs}")
        observations, _ = env.reset()
        observations = {k: torch.from_numpy(v).float().to(device) for k, v in observations.items()}
        done = False
        total_reward = 0
        batch_data = {agent_name: [] for agent_name in env.agents}
        total_steps = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_loss = 0
        total_entropy = 0

        while total_steps < batch_size:
            actions = {}
            for agent_name, state in observations.items():
                flat_state = torch.from_numpy(state.flatten()).float() if isinstance(state, np.ndarray) else state.flatten().float()
                flat_state = flat_state.to(device)
                current_agent = agent if agent_name == 'first_0' else opponent_agent
                action, log_prob, value, entropy = current_agent.act(flat_state, device)

                actions[agent_name] = action
                batch_data[agent_name].append((flat_state, action, log_prob, value, entropy))

            step_results = env.step(actions)
            next_observations, rewards, dones = step_results[:3]
            total_reward += sum(rewards.values())

            for agent_name in env.agents:
                batch_data[agent_name][-1] += (rewards[agent_name], dones[agent_name])

            total_steps += 1
            observations = next_observations

            if done:
                observations, _ = env.reset()

        for agent_name, data in batch_data.items():
            policy_loss, value_loss, loss, entropy = process_batch_data(agent_name, data, agent, opponent_agent, optimizer, opponent_optimizer, gamma, entropy_coeff, device)
            total_policy_loss += policy_loss
            total_value_loss += value_loss
            total_loss += loss
            total_entropy += entropy

        if total_reward > best_reward:
            best_reward = total_reward
            best_state_dict = agent.state_dict()

        if epoch % 100 == 0 and epoch != 0:
            agent.load_state_dict(best_state_dict)

        if epoch % 400 == 0:
            for agent_name in ['agent', 'opponent_agent']:
                current_agent = agent if agent_name == 'agent' else opponent_agent
                torch.save(current_agent.state_dict(), model_save_path(epoch, agent_name))
        if epoch % 10 == 0:
            wandb.log({
                'epoch': epoch,
                'average_policy_loss': total_policy_loss / total_steps,
                'average_value_loss': total_value_loss / total_steps,
                'average_total_loss': total_loss / total_steps,
                'average_entropy': total_entropy / total_steps,
                'total_reward': total_reward,
                'average_reward': total_reward / total_steps
            })

    for agent_name in ['agent', 'opponent_agent']:
        current_agent = agent if agent_name == 'agent' else opponent_agent
        torch.save(current_agent.state_dict(), final_model_save_path(agent_name))

def process_batch_data(agent_name, data, agent, opponent_agent, optimizer, opponent_optimizer, gamma, entropy_coeff, device):
    states, actions, log_probs, values, entropies, rewards, dones = zip(*data)

    states = [torch.from_numpy(state).float().view(1, -1).to(device) if isinstance(state, np.ndarray) else state.view(1, -1).to(device) for state in states]
    states = torch.cat(states, dim=0)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    log_probs = torch.stack(log_probs).to(device)
    values = torch.stack(values).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float).to(device)
    dones = torch.tensor(dones, dtype=torch.float).to(device)

    with torch.no_grad():
        next_value = agent.value(states[-1]).squeeze() if agent_name == 'first_0' else opponent_agent.value(states[-1]).squeeze()
    returns = compute_gae(next_value, rewards, dones, values, gamma)
    returns = torch.tensor(returns, dtype=torch.float).to(device)
    advantages = returns - values

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    total_entropy = 0
    for entropy in entropies:
        total_entropy += entropy

    current_agent = agent if agent_name == 'first_0' else opponent_agent
    current_optimizer = optimizer if agent_name == 'first_0' else opponent_optimizer

    current_agent_policy, _ = current_agent(states)
    old_probs = torch.exp(log_probs)
    new_probs = current_agent_policy.gather(1, actions.unsqueeze(1)).squeeze(1)
    ratio = new_probs / old_probs

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    _, current_agent_value = current_agent(states)
    value_loss = 0.5 * (returns - current_agent_value.squeeze()).pow(2).mean()

    loss = policy_loss + value_loss - entropy_coeff * torch.stack(entropies).mean()

    current_optimizer.zero_grad()
    loss.backward()
    current_optimizer.step()

    return policy_loss.item(), value_loss.item(), loss.item(), total_entropy / len(entropies)
