import numpy as np
from pettingzoo.atari import pong_v3
from supersuit import color_reduction_v0, resize_v1, frame_stack_v1

# Initialize the environment
env_pz = pong_v3.parallel_env()
env_pz = color_reduction_v0(env_pz)
env_pz = resize_v1(env_pz, 84, 84)
# Note: Skipping frame_stack_v1 since we'll manually handle frame skipping and stacking

observations, _ = env_pz.reset()

# Custom frame skipping and max pooling
def custom_frame_skip_and_max_pool(env, actions, skip=4):
    total_rewards = {agent: 0 for agent in env.agents}
    terminated = {agent: False for agent in env.agents}
    truncated = {agent: False for agent in env.agents}
    infos = {agent: {} for agent in env.agents}
    
    final_observations = observations
    for _ in range(skip):
        observations, rewards, terminations, truncations, info = env.step(actions)
        
        # Accumulate rewards
        for agent, reward in rewards.items():
            total_rewards[agent] += reward
        
        # Track terminations and truncations
        for agent in env.agents:
            terminated[agent] |= terminations[agent]
            truncated[agent] |= truncations[agent]
        
        # Use the last observation for max pooling
        final_observations = observations
    
    return final_observations, total_rewards, terminated, truncated, infos

# Example usage
actions = {agent: env_pz.action_space(agent).sample() for agent in env_pz.agents}
obs, rewards, terminated, truncated, infos = custom_frame_skip_and_max_pool(env_pz, actions)

# Note: This example doesn't implement actual max pooling across frames, as it simply uses the last observation.
# For true max pooling, you'd need to compare pixel values across the skipped frames and take the maximum.
