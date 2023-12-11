# from dtu import Parameters, dtu, GPU  # Comment out if these are HPC-specific
from pettingzoo.atari import pong_v3
from ppo_training import PPOAgent, train_ppo  # Import from ppo_combined.py
import torch.optim as optim
import numpy as np

class Defaults:  # Removed the inheritance from Parameters if it's HPC-specific
    def __init__(self):
        self.name = "experiment-name"
        self.instances = 1
        # self.GPU = None  # GPU configuration, remove if not relevant for local setup
        self.time = 84600  # 23.5 hours, adjust if needed
        # self.data_folder_name = "data_fod"  # Uncomment and adjust path if needed
        self.epochs = 1000
        self.batch_size = 32  
        self.isServer = False  # Set to False for local execution
        self.gamma = 99

    def run(self):
        self.train_agent(self.name, self.epochs, self.batch_size, self.gamma)

    def train_agent(self, name, epochs, batch_size, gamma):
        print("Starting training with PPO Agent")
        env = pong_v3.parallel_env()
        output_dim = env.action_space('first_0').n
        input_dim = np.prod(env.observation_space('first_0').shape)

        agent = PPOAgent(input_dim, output_dim)
        opponent_agent = PPOAgent(input_dim, output_dim)  # Create opponent agent
        optimizer = optim.Adam(agent.parameters(), lr=3e-4)
        save_folder = '/path/to/save_folder'
        train_ppo(agent, opponent_agent, env, optimizer, name, epochs, gamma, save_folder, batch_size)

# Start the program
defaults = Defaults()
defaults.run()
