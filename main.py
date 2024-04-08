import sys

# Print the path to the Python interpreter
print("Python interpreter path:", sys.executable)
from pettingzoo.atari import pong_v3
from ppo_training import PPOAgent, train_ppo
import torch.optim as optim
import numpy as np
import torch
from dtu import Parameters, dtu, GPU
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1

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



    def train_agent(self, name, epochs, batch_size, gamma, entropy_coeff, lr):
        print("Starting training with PPO Agent")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        env = pong_v3.parallel_env()

        # Apply SuperSuit transformations
        env = color_reduction_v0(env, mode='B')  # Convert to grayscale
        env = resize_v1(env, x_size=80, y_size=80)  # Resize to 80x80
        env = frame_stack_v1(env, 4)  # Stack 4 frames

        output_dim = env.action_space('first_0').n

        # Adjust input dimensions to match the preprocessing
        input_channels = 1  # Grayscale means only 1 channel
        input_height = 80   # Height after resize
        input_width = 80    # Width after resize

        agent = PPOAgent(input_channels, input_height, input_width, output_dim).to(device)

        optimizer = optim.Adam(agent.parameters(), lr=lr)

        save_folder = '/zhome/59/9/198225/Desktop/Adversarial_DRL/Adversarial_DRL/agents/'

        train_ppo(agent=agent, env=env, optimizer=optimizer, name=name, epochs=epochs, entropy_coeff=entropy_coeff, gamma=gamma, save_folder=save_folder, batch_size=batch_size, device=device)

# Start the program
Defaults.start()
