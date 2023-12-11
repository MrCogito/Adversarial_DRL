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

def train_ppo(agent, opponent_agent, env, optimizer, opponent_optimizer, name, epochs, gamma, save_folder, batch_size, device):
    # Initialize wandb (if you're using it)
    wandb.init(project="adversarial_dlr", name=name)
    
    # Paths for saving metrics and model
    model_save_path = lambda epoch, agent_name: os.path.join(save_folder, f'{agent_name}_trained_agent_epoch_{epoch}.pth')
    final_model_save_path = lambda agent_name: os.path.join(save_folder, f'{agent_name}_trained_agent.pth')

    os.makedirs(save_folder, exist_ok=True)

    # Training loop
    for epoch in range(epochs):
        print(f"Starting Epoch {epoch+1}/{epochs}")
        observations, _ = env.reset()
        observations = {k: torch.from_numpy(v).float().to(device) for k, v in observations.items()}
        done = False
        total_reward = 0
        batch_data = {agent_name: [] for agent_name in env.agents}  # Data for each agent
        total_steps = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_loss = 0
        total_entropy = 0

        while total_steps < batch_size:
            actions = {}
            for agent_name, state in observations.items():
                if isinstance(state, np.ndarray):
                    flat_state = torch.from_numpy(state.flatten()).float()
                elif isinstance(state, torch.Tensor):
                    flat_state = state.flatten().float()
                else:
                    raise TypeError("State must be a np.ndarray or torch.Tensor")

                # Move the tensor to the specified device
                flat_state = flat_state.to(device)
                current_agent = agent if agent_name == 'first_0' else opponent_agent
                action, log_prob, value, entropy = current_agent.act(flat_state,device)

                actions[agent_name] = action

                batch_data[agent_name].append((flat_state, action, log_prob, value, entropy))
            step_results = env.step(actions)
            next_observations, rewards, dones = step_results[:3]
            #print(rewards.values())
            total_reward += sum(rewards.values())

            for agent_name in env.agents:
                batch_data[agent_name][-1] += (rewards[agent_name], dones[agent_name])
            #print(f"Epoch {epoch+1}, Step {total_steps+1}: Step completed")
            total_steps += 1
            observations = next_observations

            if done:
                observations, _ = env.reset()
            #print(f"Data collection step: {total_steps+1}")

        # Process batch data for each agent
        for agent_name, data in batch_data.items():
           # print(f"Processing batch data for {agent_name}")
            policy_loss, value_loss, loss, entropy = process_batch_data(agent_name, data, agent, opponent_agent, optimizer, opponent_optimizer, gamma, device)
            total_policy_loss += policy_loss
            total_value_loss += value_loss
            total_loss += loss
            total_entropy += entropy
           # print(f"Completed processing batch data for {agent_name}")
        # Logging and saving
        if epoch % 10 == 0:  # Adjust the frequency as needed
            for agent_name in ['agent', 'opponent_agent']:
                current_agent = agent if agent_name == 'agent' else opponent_agent
                torch.save(current_agent.state_dict(), model_save_path(epoch, agent_name))
            wandb.log({
            'epoch': epoch,
            'average_policy_loss': total_policy_loss / total_steps,
            'average_value_loss': total_value_loss / total_steps,
            'average_total_loss': total_loss / total_steps,
            'average_entropy': total_entropy / total_steps,
            'total_reward': total_reward,
            'average_reward': total_reward / total_steps
        })
       # print(f"Finished Epoch {epoch+1}/{epochs}")
    # Final save after training completion
    for agent_name in ['agent', 'opponent_agent']:
        current_agent = agent if agent_name == 'agent' else opponent_agent
        torch.save(current_agent.state_dict(), final_model_save_path(agent_name))


def process_batch_data(agent_name, data, agent, opponent_agent, optimizer, opponent_optimizer, gamma,device):
    # Unpack data
    #print(f"Processing batch data for {agent_name}")
    total_entropy = 0

    states, actions, log_probs, values, entropies, rewards, dones = zip(*data)

    # Check if states are NumPy arrays and convert them to tensors if they are
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

    # Initialize entropy accumulation
    total_entropy = 0

    # Loop through each step in the batch to calculate entropy
    for entropy in entropies:
        total_entropy += entropy

    # Average entropy over all steps
    average_entropy = total_entropy / len(entropies)


    current_agent = agent if agent_name == 'first_0' else opponent_agent
    current_optimizer = optimizer if agent_name == 'first_0' else opponent_optimizer

    # Compute policy loss and value loss
    current_agent_policy, _ = current_agent(states)
    old_probs = torch.exp(log_probs)
    new_probs = current_agent_policy.gather(1, actions.unsqueeze(1)).squeeze(1)
    ratio = new_probs / old_probs

    # Clipped policy objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # Value loss
    _, current_agent_value = current_agent(states)
    value_loss = 0.5 * (returns - current_agent_value.squeeze()).pow(2).mean()

    # Total loss
    loss = policy_loss + value_loss - 0.01 * torch.stack(entropies).mean()

    current_optimizer.zero_grad()
    loss.backward()
    current_optimizer.step()
    #print(f"Completed processing batch data for {agent_name}")
    return policy_loss.item(), value_loss.item(), loss.item(), average_entropy
