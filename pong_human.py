import time
from pynput import keyboard
from pettingzoo.atari import pong_v3
import random
import torch
from ppo_training import PPOAgent

# Global variable to store the current action and human frame counter
current_action = 0
human_frame_counter = 0
delay = 0.003 # make game slower

def load_model(model_path, input_dim, output_dim, device):
    
    model = PPOAgent(input_dim, output_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model

# Define the step pattern (1 = action allowed, 0 = action not allowed)

step_pattern = [1, 0,]  # This pattern represents: 2 steps, 1 step delay - This is implemented to slow down paddle 

def on_key_press(key):
    global current_action
    if key == keyboard.Key.up:
        current_action = 2  # Move left/up
    elif key == keyboard.Key.down:
        current_action = 3  # Move right/down
    elif key == keyboard.Key.left:
        current_action = 1  # Move right/down

def on_key_release(key):
    global current_action
    # Reset action to NOOP when key is released
    current_action = 0

listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
listener.start()

def main():
    env = pong_v3.env(render_mode="human")
    env.reset()

    global human_frame_counter

    #input_dim = env.observation_space('first_0').shape[0]
    #output_dim = env.action_space('first_0').n
    input_dim = 210 * 160 * 3
    output_dim = env.action_space('first_0').n

    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model = load_model("exp20kTest32-0_agent_trained_agent_epoch_7200.pth", input_dim, output_dim, device)

    for agent in env.agent_iter():
        env.render()
        observation, reward, termination, truncation, info = env.last()

        if agent == 'first_0':
            human_frame_counter += 1
            # Determine action based on the step pattern
            pattern_position = human_frame_counter % len(step_pattern)
            if step_pattern[pattern_position] == 0:
                action = 0  # NOOP
            else:
                action = current_action
        else:
            #action = random.choice(list(range(6)))  # Random action for the second_0 agent
            obs_tensor = torch.from_numpy(observation).float().to(device)
            action, _, _, _ = trained_model.act(obs_tensor, device)
        env.step(action)

        # Introduce a delay to slow down the game
        time.sleep(delay)  # 0.05 seconds delay. Adjust this value as needed.

    env.close()

if __name__ == "__main__":
    main()
