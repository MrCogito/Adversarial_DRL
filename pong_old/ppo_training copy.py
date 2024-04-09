import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

class PPO(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPO, self).__init__()
        self.hidden_size = 200
        
        # Shared layers for both policy and value networks
        self.fc1 = nn.Linear(input_dim, self.hidden_size)
        
        # Policy network
        self.policy = nn.Linear(self.hidden_size, output_dim)
        self.softmax = nn.Softmax(dim=-1)
        
        # Value network
        self.value = nn.Linear(self.hidden_size, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        policy = self.softmax(self.policy(x))
        value = self.value(x)
        return policy, value
    
    def act(self, state, device):
        state = torch.from_numpy(state).float().to(device)
        probs, value = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        entropy = m.entropy()
        return action.item(), log_prob, value, entropy


def calculate_loss(agent, states, actions, old_log_probs, returns, advantages, entropy_coeff=0.01, clip_param=0.2):
    # Forward pass
    new_probs, state_values = agent(states)
    new_log_probs = torch.log(new_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1))
    
    # Entropy Bonus
    entropy_bonus = -(new_probs * torch.log(new_probs)).sum(1).mean()
    
    # Ratio for Clipped Surrogate Objective
    ratios = torch.exp(new_log_probs - old_log_probs.detach())

    # Clipped Surrogate Objective
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1.0 - clip_param, 1.0 + clip_param) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # Value Function Loss
    value_loss = F.mse_loss(state_values.squeeze(), returns)

    # Total Loss
    loss = policy_loss + 0.5 * value_loss - entropy_coeff * entropy_bonus
    
    return policy_loss, value_loss, loss, entropy_bonus



import torch
import torch.optim as optim
from collections import deque

def train_ppo(agent, env, optimizer, device, epochs=100, threshold=21, update_every=10):
    agent.to(device)
    total_rewards = deque(maxlen=100)  # For tracking average rewards over recent episodes
    update_count = 0  # Tracks number of updates (i.e., batches of episodes processed)

    for epoch in range(epochs):
        state, _ = env.reset()
        state = torch.tensor(state['first_0'], dtype=torch.float32).to(device)
        episode_rewards = 0  # Sum of rewards for the current episode

        log_probs = []
        values = []
        states = []
        actions = []
        rewards = []
        dones = []
        advantages = []
        returns = []

        done = False
        while not done:
            action, log_prob, value, entropy = agent.act(state.numpy(), device)
            next_state, reward, done, _ = env.step({'first_0': action})
            next_state = torch.tensor(next_state['first_0'], dtype=torch.float32).to(device)

            # Store outcomes for later processing
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob)
            values.append(value)

            state = next_state
            episode_rewards += reward

            # Check for episode termination condition (e.g., reaching 21 points)
            if done or episode_rewards >= threshold:
                next_value = 0 if done else agent(state)[1].item()
                returns, advs = compute_returns_advantages(rewards, dones, values, next_value, device)
                advantages.extend(advs)
                returns.extend(returns)

                # Reset episode-specific accumulators
                rewards = []
                dones = []
                values = []

        total_rewards.append(episode_rewards)

        if (epoch + 1) % update_every == 0:
            # After 'update_every' episodes, perform an update
            update_count += 1
            optimizer.zero_grad()
            policy_loss, value_loss, loss = calculate_total_loss(agent, states, actions, log_probs, returns, advantages, device)
            loss.backward()
            optimizer.step()

            print(f"Update {update_count}: Average Reward: {sum(total_rewards) / len(total_rewards)}")


def calculate_total_loss(agent, states, actions, log_probs, returns, advantages, device):
    # This function would calculate the total loss using the collected states, actions, etc.
    # Placeholder implementation
    return policy_loss, value_loss, total_loss

def compute_returns_advantages(rewards, dones, values, next_value, device, gamma=0.99, tau=0.95):
    # Compute GAE and returns. This is a placeholder implementation.
    returns = []
    advantages = []
    return returns, advantages

# Initialize your environment, agent, and optimizer here
# env = YourEnvironment()
# agent = PPO(input_dim, output_dim).to(device)
# optimizer = optim.Adam(agent.parameters(), lr=0.001)

# train_ppo(agent, env, optimizer, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))



def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def calculate_loss(agent, states, actions, old_log_probs, values, advantages, returns, entropy_coeff=0.01, clip_param=0.2):
    new_probs, new_value = agent(states)
    new_log_probs = new_probs.gather(1, actions.unsqueeze(-1)).log()

    ratios = torch.exp(new_log_probs - old_log_probs.detach())
    surr1 = ratios * advantages.detach()
    surr2 = torch.clamp(ratios, 1.0 - clip_param, 1.0 + clip_param) * advantages.detach()
    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss = (returns - new_value).pow(2).mean()

    # Compute the entropy bonus
    entropy_bonus = -(new_probs * torch.log(new_probs + 1e-10)).mean()

    loss = policy_loss + 0.5 * value_loss - entropy_coeff * entropy_bonus

    return policy_loss, value_loss, loss, entropy_bonus