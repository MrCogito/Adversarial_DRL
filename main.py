import sys

# Print the path to the Python interpreter
print("Python interpreter path:", sys.executable)
from pettingzoo.atari import pong_v3
from ppo_training import PPOAgent, train_ppo
import torch.optim as optim
import numpy as np
import torch
from dtu import Parameters, dtu, GPU

@dtu
class Defaults(Parameters):
    name: str = "testname"
    instances: int = 1
    #GPU: None | GPU = GPU.v32
    time: int = 84600 # 23.5 hours
    GPU:  GPU = GPU.v32
    #data_folder_name: str = "data_fod"
    epochs: int = 1000
    batch_size: int = 32  
    isServer: bool = True
    gamma: float = 99
    lr: float = 0.003



    def run(self, name: str, epochs: int, batch_size: int, gamma: float, lr: float):
        entropy_coef=0.15
        self.train_agent(self=self,name=name, epochs=epochs, batch_size=batch_size, gamma=gamma,entropy_coeff=entropy_coef, lr=lr)

    def train_agent(self, name, epochs, batch_size, gamma, entropy_coeff, lr):
        print("Starting training with PPO Agent")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        env = pong_v3.parallel_env()
        output_dim = env.action_space('first_0').n
        input_dim = np.prod(env.observation_space('first_0').shape)

        input_channels = 3  # RGB channels
        input_height = 210  # Height of the Atari frame
        input_width = 160   # Width of the Atari frame
        output_dim = 6      # Number of discrete actions in the Pong game

        agent = PPOAgent(input_channels, input_height, input_width, output_dim).to(device)

        optimizer = optim.Adam(agent.parameters(), lr=lr)

        save_folder = '/zhome/59/9/198225/Desktop/Adversarial_DRL/Adversarial_DRL/agents/'

        train_ppo(agent=agent, env=env, optimizer=optimizer, name=name, epochs=epochs, entropy_coeff=entropy_coeff, gamma=gamma, save_folder=save_folder, batch_size=batch_size, device=device)

# Start the program
Defaults.start()
