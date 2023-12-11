from dtu import Parameters, dtu, GPU
from pettingzoo.atari import pong_v3
from ppo_training import PPOAgent, train_ppo  # Import from ppo_combined.py
import torch.optim as optim
import numpy as np

@dtu
class Defaults(Parameters):
    name: str = "experiment-name"
    instances: int = 1
    GPU: None | GPU = GPU.v32
    time: int = 84600 # 23.5 hours
    #data_folder_name: str = "data_fod"
    epochs: int = 1000
    batch_size: int = 32  
    isServer: bool = True
    gamma: float = 99

    def run(self, name: str, epochs: int, batch_size: int, gamma: float):
        self.train_agent(name=name, epochs=epochs, batch_size=batch_size, gamma=gamma)

    def train_agent(self, name, epochs, batch_size, gamma):
        print("Starting training with PPO Agent")
        env = pong_v3.parallel_env()
        output_dim = env.action_space('first_0').n
        input_dim = np.prod(env.observation_space('first_0').shape)

        agent = PPOAgent(input_dim, output_dim)
        opponent_agent = PPOAgent(input_dim, output_dim)  # Create opponent agent
        optimizer = optim.Adam(agent.parameters(), lr=3e-4)
        opponent_optimizer = optim.Adam(opponent_agent.parameters(), lr=3e-4)  # Separate optimizer for opponent

        save_folder = '/zhome/59/9/198225/Adversarial_DRL/agents/'
        train_ppo(agent=agent, opponent_agent=opponent_agent, env=env, optimizer=optimizer, opponent_optimizer=opponent_optimizer, name=name, epochs=epochs, gamma=gamma, save_folder=save_folder, batch_size=batch_size)

# Start the program
Defaults.start()
