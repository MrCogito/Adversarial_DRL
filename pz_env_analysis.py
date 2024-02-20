from pettingzoo.atari import pong_v3
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to save an observation as an image file
def save_observation(observation, filename):
    plt.figure(figsize=(5, 5))
    plt.imshow(observation, cmap='gray')
    plt.axis('off')
    plt.savefig(filename)
    plt.close()

# Function to inspect and analyze the observation
def inspect_observation(env, steps=5):
    observations, infos = env.reset()
    step_counter = 0

    while True:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        step_counter += 1

        # When we reach the specified step, inspect the observation
        if step_counter == steps:
            # Assuming 'first_0' is still in the game
            if 'first_0' in observations:
                obs = observations['first_0']
                print(f"Observation shape at step {steps}: {obs.shape}")
                print(f"Observation size: {obs.size}")
                print(f"Value range: min {np.min(obs)}, max {np.max(obs)}")

                # Save the observation values and image
                np.savetxt(f"observation_values_step_{steps}.csv", obs.flatten(), delimiter=",")
                save_observation(obs, f"observation_step_{steps}.png")
            break

        if not env.agents:
            break  # End the loop if all agents are done

    env.close()

# Initialize the PettingZoo environment
env = pong_v3.parallel_env()

# Inspect and analyze the observation after 5 steps
inspect_observation(env, 5)
