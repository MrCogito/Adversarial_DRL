import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import torch.nn.init as init
import wandb
import random
import torch.nn as nn
import torch.nn.functional as F

class PPOAgent(nn.Module):
    def __init__(self, input_channels, input_height, input_width, output_dim):
        super(PPOAgent, self).__init__()

        # CNN layers for feature extraction
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate the size of the flattened feature map after the conv layers
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_width, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_height, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64

        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(linear_input_size, 192),
            nn.ReLU(),
            nn.Linear(192, 384),
            nn.ReLU(),
            nn.Linear(384, output_dim),
            nn.Softmax(dim=-1)
        )
        
        # Value network
        self.value = nn.Sequential(
            nn.Linear(linear_input_size, 192),
            nn.ReLU(),
            nn.Linear(192, 384),
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
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # Reshape the output for the linear layers
        policy = self.policy(x)
        value = self.value(x)
        return policy, value
    
    def act(self, state, device, action_space, epsilon=0.1):
        if random.random() < epsilon:
            # Exploration: Choose a random action
            action = action_space.sample()
            log_prob = torch.zeros(1, device=device)
            value = self.forward(state)[1]
            entropy = torch.zeros(1, device=device)
            return action, log_prob, value, entropy
        else:
            # Policy action
            probs, value = self.forward(state)
            m = Categorical(probs)
            action = m.sample()
            log_prob = m.log_prob(action)
            entropy = m.entropy()
            return action.item(), log_prob, value, entropy




def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    next_value = next_value.unsqueeze(0) if next_value.dim() == 1 else next_value
    values = torch.cat([values, next_value], dim=0)
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def train_ppo(agent, env, optimizer, name, epochs, gamma, entropy_coeff, save_folder, batch_size, device):
    wandb.init(project="adversarial_dlr", name=name)
    
    model_save_path = lambda epoch, agent_name: os.path.join(save_folder, f'{name}_{agent_name}_trained_agent_epoch_{epoch}.pth')
    final_model_save_path = lambda agent_name: os.path.join(save_folder, f'{name}_{agent_name}_trained_agent.pth')
    input_channels = 3  # because observation shape has 3 channels (RGB)
    input_height = 210
    input_width = 160
    output_dim = 6
    output_dim = 6  # Number of actions in Pong
    agent = PPOAgent(input_channels, input_height, input_width, output_dim).to(device)
    opponent_agent = PPOAgent(input_channels, input_height, input_width, output_dim).to(device) 
    opponent_agent.load_state_dict(agent.state_dict())

    os.makedirs(save_folder, exist_ok=True)
    best_state_dict = agent.state_dict()
    best_score = 100

    lr_decay_rate = 0.99
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay_rate)

    for epoch in range(epochs):
        observations, _ = env.reset()
        observations = {k: torch.from_numpy(v).float().unsqueeze(0).to(device) for k, v in observations.items()}
        action_counts = {agent_name: {action: 0 for action in range(env.action_space(agent_name).n)} for agent_name in env.possible_agents}
        done = False
        
        scores = {agent_name: 0 for agent_name in env.possible_agents}
        rewards_history = {agent_name: [] for agent_name in env.possible_agents}
        batch_data = {agent_name: [] for agent_name in env.possible_agents}
        episode_length = 0
        max_episode_length = 5000
        batch_data_threshold = 1000
        total_policy_loss = 0
        total_value_loss = 0
        total_loss = 0
        total_entropy = 0
        action_distribution = {}

        while not done and episode_length < max_episode_length:
            actions = {}
            log_probs = {}
            values = {}
            entropies = {}
            
            for agent_name, state in observations.items():
                
               # print("State before preprocessing:", state.shape) 
                if isinstance(state, np.ndarray):
                    # Convert from NumPy array to tensor, normalize, and permute dimensions
                    state = torch.from_numpy(state).float().to(device) / 255.0
                    state = state.permute(2, 0, 1).unsqueeze(0)  # Convert to (batch, channels, height, width)
                else:
                    # If already a tensor, normalize if necessary and check dimensions
                    state = state.float().to(device) / 255.0
                    if state.shape[-1] == 3 and state.ndim == 3:
                        # If it's (height, width, channels), permute to (channels, height, width) and add batch dimension
                        state = state.permute(2, 0, 1).unsqueeze(0)
                    elif state.ndim == 4 and state.shape[1] != 3:
                        # If it's (batch, height, width, channels), permute to (batch, channels, height, width)
                        state = state.permute(0, 3, 1, 2)

                #print("State after preprocessing:", state.shape) 
                current_agent = agent if agent_name == 'first_0' else opponent_agent
                action_space = env.action_space(agent_name)
                action, log_prob, value, entropy = current_agent.act(state, device, action_space)
                actions[agent_name] = action
                log_probs[agent_name] = log_prob
                values[agent_name] = value
                entropies[agent_name] = entropy
                action_counts[agent_name][action] += 1

            step_results = env.step(actions)
            next_observations, rewards, dones, _, _ = step_results

            for agent_name in env.agents:
                scores[agent_name] += rewards[agent_name]
                rewards_history[agent_name].append(rewards[agent_name])
                # Collect data for batch processing
                flat_state = torch.from_numpy(observations[agent_name].flatten()).float() if isinstance(observations[agent_name], np.ndarray) else observations[agent_name].flatten().float()
                flat_state = flat_state.to(device)
                batch_data[agent_name].append(
                                             (flat_state, actions[agent_name], log_probs[agent_name], values[agent_name], entropies[agent_name], rewards[agent_name], dones[agent_name])
                                             )                           

            if len(batch_data) >= batch_data_threshold:
                states, actions, log_probs, values, entropies, rewards, dones = zip(*batch_data)
                policy_loss, value_loss, loss, entropy = process_batch_data(
                    states, actions, log_probs, values, entropies, rewards, dones,
                    agent, optimizer, gamma, entropy_coeff, device
                )
                total_policy_loss += policy_loss
                total_value_loss += value_loss
                total_loss += loss
                total_entropy += entropy
                action_distribution[agent_name] = {}
                for action, count in action_counts[agent_name].items():
                    if count > 0:
                        action_distribution[agent_name][action] = count / sum(action_counts[agent_name].values())
                    else:
                        action_distribution[agent_name][action] = 0.0
                batch_data[agent_name].clear()

                
                
            episode_length += 1
            done = any(dones.values())
            observations = next_observations

        current_score = abs(scores['first_0'])
        if current_score < best_score:
            best_score = current_score
            best_state_dict = agent.state_dict()

        scheduler.step()

        wandb.log({
            'epoch': epoch,
            'episode_length': episode_length,
            'best_score': best_score,
            'current_score_agent': scores['first_0'],
            'current_score_opponent': scores['second_0'],
            'average_policy_loss': total_policy_loss / episode_length,
            'average_value_loss': total_value_loss / episode_length,
            'average_total_loss': total_loss / episode_length,
            'average_entropy': total_entropy / episode_length,
            'agent_rewards':  rewards,
            'opponent_rewards': rewards_history['first_0'],
            'action_distribution': dict(action_counts)
        })

        if epoch == 15 or (epoch > 15 and epoch % 5 == 0):
            agent.load_state_dict(best_state_dict)
            opponent_agent.load_state_dict(agent.state_dict())

        if epoch % 1000 == 0:
            for agent_name in ['agent', 'opponent_agent']:
                current_agent = agent if agent_name == 'agent' else opponent_agent
                # torch.save(current_agent.state_dict(), model_save_path(epoch, agent_name))
    
    # torch.save(best_state_dict, final_model_save_path('best_agent'))

    
def process_batch_data(states, actions, log_probs, values, entropies, rewards, dones, agent, optimizer, gamma, entropy_coeff, device):
    # Convert lists to tensors
    states = torch.stack(states).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    log_probs = torch.stack(log_probs).to(device)
    values = torch.stack(values).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float).to(device)
    dones = torch.tensor(dones, dtype=torch.float).to(device)

    with torch.no_grad():
        next_value = agent.value(states[-1]).squeeze()
    returns = compute_gae(next_value, rewards, dones, values, gamma)
    returns = torch.tensor(returns, dtype=torch.float).to(device)
    advantages = returns - values

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    total_entropy = 0
    for entropy in entropies:
        total_entropy += entropy

    current_agent_policy, _ = agent(states)
    old_probs = torch.exp(log_probs)
    new_probs = current_agent_policy.gather(1, actions.unsqueeze(1)).squeeze(1)
    ratio = new_probs / old_probs

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    _, current_agent_value = agent(states)
    value_loss = 0.5 * (returns - current_agent_value.squeeze()).pow(2).mean()

    loss = policy_loss + value_loss - entropy_coeff * total_entropy / len(entropies)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return policy_loss.item(), value_loss.item(), loss.item(), total_entropy / len(entropies)
