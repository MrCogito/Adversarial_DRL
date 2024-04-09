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
        #print("Debug: Shape before flattening:", x.shape)  # Add this line

        x = x.reshape(x.size(0), -1)  # Reshape the output for the linear layers
        #print("Debug: Shape before linear layers:", x.shape)
        policy = self.policy(x)
        value = self.value(x)

        return policy, value
    
    def act(self, state, device, action_space, epsilon=0.1):
        if random.random() < epsilon:
            # Exploration: Choose a random action
            action = action_space.sample()
            log_prob = torch.zeros(1, device=device)
            policy, value = self.forward(state)
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




def compute_gae(next_value, rewards, masks, values, gamma=1, tau=0.95):
    # Ensure next_value is a 1D tensor with a single element
    next_value = next_value.squeeze()  # Remove extra dimensions if any
    if next_value.dim() == 0:
        next_value = next_value.unsqueeze(0)  # Add batch dimension if missing

    print("Debug: Shape of next_value after processing:", next_value.shape)  # Debug print

    # Check if values need reshaping
    if values.dim() == 3:
        values = values.squeeze(1)  # Adjust values to be 2D if necessary

    print("Debug: Shape of values before concatenation:", values.shape)  # Debug print

    # Add next_value as the last element of values
    values = torch.cat([values, next_value.unsqueeze(0)], dim=0)
    print("Debug: Shape of values after concatenation:", values.shape)  # Debug print

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
    opponent_agent.load_state_dict(agent.state_dict().copy())

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
                #print("OBSRERVATIONS", state.shape)
                
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
                 # Normalize the score
                normalized_score = scores[agent_name] / 20 #20 is max num of points
                
                rewards_history[agent_name].append(rewards[agent_name])
                # Collect data for batch processing
                if isinstance(observations[agent_name], np.ndarray):
                    # Normalize and convert to PyTorch tensor
                    obs_tensor = torch.from_numpy(observations[agent_name]).float() / 255.0

                    # Reshape to [C, H, W]. The original shape is assumed to be [H, W, C]
                    obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0)  # Add batch dimension
                else:
                    # If it's already a tensor, normalize if necessary
                    obs_tensor = observations[agent_name].float() / 255.0
                    if obs_tensor.shape[-1] == 3 and obs_tensor.ndim == 3:
                        # If it's (H, W, C), permute to (C, H, W) and add batch dimension
                        obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0)
                    elif obs_tensor.ndim == 4 and obs_tensor.shape[1] != 3:
                        # If it's (B, H, W, C), permute to (B, C, H, W)
                        obs_tensor = obs_tensor.permute(0, 3, 1, 2)

                # Move to the appropriate device (e.g., GPU if available)
                obs_tensor = obs_tensor.to(device)

                # Now, obs_tensor is in the correct shape and range to be fed into your CNN
             

                #state = state.squeeze(0)
                

                batch_data[agent_name].append(
                    (obs_tensor, actions[agent_name], log_probs[agent_name], values[agent_name], entropies[agent_name], normalized_score, dones[agent_name])
                )             
                         
            if len(batch_data['first_0']) >= batch_data_threshold:
                # Process the batch data for 'first_0'
                states, actions, log_probs, values, entropies, rewards, dones = zip(*batch_data['first_0'])
                policy_loss, value_loss, loss, entropy = process_batch_data(
                    states, actions, log_probs, values, entropies, rewards, dones,
                    agent, optimizer, gamma, entropy_coeff, device
                )
                total_policy_loss += policy_loss
                total_value_loss += value_loss
                total_loss += loss
                total_entropy += entropy

                # Clear the data list for 'first_0' after processing
                batch_data['first_0'].clear()

                # Reset action counts for 'first_0'
                action_counts['first_0'] = {action: 0 for action in range(env.action_space('first_0').n)}


                
                    
            episode_length += 1
            # desired_keys = ['first_0', 'second_0']
            # print("dones", dones.type)
            # if any(dones.get(key, False) for key in desired_keys):
            #     done = True
            # else:
            #     done = False
            # print("done", done)
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
            'agent_rewards_normalized':  normalized_score,
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
    states = torch.stack(states).to(device)  # states are already in the correct format
    #print("Debug: Shape of states after stacking:", states.shape)  # Debug info
    #states = states.squeeze(1)
    #print("Debug: Shape of states after stacking squeeze:", states.shape)  # Debug info
    states = states.squeeze(1)  # Remove if the extra dimension is not needed
  #  print("Debug: Shape of states after potential squeeze:", states.shape)

    actions = torch.tensor(actions, dtype=torch.long).to(device)
    log_probs = torch.stack(log_probs).to(device)
    values = torch.stack(values).to(device)
    print("Debug: Shape of values after stacking:", values.shape)  # Add this line
    rewards = torch.tensor(rewards, dtype=torch.float).to(device)
    dones = torch.tensor(dones, dtype=torch.float).to(device)

    with torch.no_grad():
        last_state = states[-1].unsqueeze(0) if states[-1].dim() == 3 else states[-1]
        print("Debug: Shape of last_state for value calculation:", last_state.shape)
        _, next_value = agent.forward(last_state)
        print("Debug: Shape of next_value before squeeze:", next_value.shape)
        

    returns = compute_gae(next_value, rewards, dones, values, gamma)
    returns = torch.tensor(returns, dtype=torch.float).to(device)
    advantages = returns - values

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    policy_loss, value_loss, loss, entropy = calculate_loss(
        states, actions, log_probs, values, advantages,returns,  entropies, agent, entropy_coeff
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return policy_loss, value_loss, loss, entropy.mean()

def calculate_loss(states, actions, log_probs, values, advantages, returns, entropies, agent, entropy_coeff):
    # Calculate policy loss, value loss, and entropy
    # This function is extracted for clarity



    current_agent_policy, current_agent_value = agent(states)
    print("Debug: Shape of current_agent_policy output:", current_agent_policy.shape)  # Debug info
    print("Debug: Shape of current_agent_value output:", current_agent_value.shape)  # Debug info

    old_probs = torch.exp(log_probs)
    new_probs = current_agent_policy.gather(1, actions.unsqueeze(1)).squeeze(1)
    ratio = new_probs / old_probs

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss = 0.5 * (returns - current_agent_value.squeeze()).pow(2).mean()
    entropy = sum(entropies) / len(entropies)

    loss = policy_loss + value_loss - entropy_coeff * entropy

    return policy_loss, value_loss, loss, entropy
