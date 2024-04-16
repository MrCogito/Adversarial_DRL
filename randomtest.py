"""Uses Stable-Baselines3 to train agents in the Connect Four environment using invalid action masking.

For information about invalid action masking in PettingZoo, see https://pettingzoo.farama.org/api/aec/#action-masking
For more information about invalid action masking in SB3, see https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html

Author: Elliot (https://github.com/elliottower)
"""
import glob
import os
import time

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker

import pettingzoo.utils
from pettingzoo.classic import connect_four_v3
import numpy as np




class SB3ActionMaskWrapper(pettingzoo.utils.BaseWrapper):
    """Wrapper to allow PettingZoo environments to be used with SB3 illegal action masking."""

    def reset(self, seed=None, options=None):
        """Gymnasium-like reset function which assigns obs/action spaces to be the same for each agent.

        This is required as SB3 is designed for single-agent RL and doesn't expect obs/action spaces to be functions
        """
        super().reset(seed, options)

        # Strip the action mask out from the observation space
        self.observation_space = super().observation_space(self.possible_agents[0])[
            "observation"
        ]
        self.action_space = super().action_space(self.possible_agents[0])

        # Return initial observation, info (PettingZoo AEC envs do not by default)
        return self.observe(self.agent_selection), {}

    def step(self, action):
        """Gymnasium-like step function, returning observation, reward, termination, truncation, info."""
        super().step(action)
        obs, reward, termination, truncation, info = super().last()
        return obs, -reward, termination, truncation, info

    def observe(self, agent):
        """Return only raw observation, removing action mask."""
        return super().observe(agent)["observation"]

    def action_mask(self):
        """Separate function used in order to access the action mask."""
        return super().observe(self.agent_selection)["action_mask"]


def mask_fn(env):
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.action_mask()

class RandomAgent:
    """Random agent that selects valid actions based on action masks."""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        mask = observation["action_mask"]
        valid_actions = np.where(mask)[0]
        return np.random.choice(valid_actions)

def train_action_mask(env_fn, steps=10000, seed=0):
    env = env_fn.env(render_mode="human")
    env.reset(seed=seed)
    
    # Create two different agents
    learning_agent = "player_1"
    random_agent = RandomAgent(env.action_space(learning_agent))

    model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)
    model.set_random_seed(seed)

    for step in range(steps):
        for agent in env.agent_iter():
            observation, _, termination, truncation, _ = env.last()
            if termination or truncation:
                action = None
            else:
                if agent == learning_agent:
                    action, _ = model.predict(observation, deterministic=True)
                else:
                    action = random_agent.act(observation)

            env.step(action)
            if agent == learning_agent and not (termination or truncation):
                model.learn(total_timesteps=1)  # Perform learning

    model.save("connect_four_model")

    env.close()

def eval_action_mask(env_fn, num_games=100, render_mode=None, **env_kwargs):
    # Evaluate a trained agent vs a random agent
    env = env_fn.env(render_mode=render_mode, **env_kwargs)

    print(
        f"Starting evaluation vs a random agent. Trained agent will play as {env.possible_agents[1]}."
    )

    try:
        latest_policy = max(
            glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = MaskablePPO.load(latest_policy)

    scores = {agent: 0 for agent in env.possible_agents}
    total_rewards = {agent: 0 for agent in env.possible_agents}
    round_rewards = []

    for i in range(num_games):
        env.reset(seed=i)
        env.action_space(env.possible_agents[0]).seed(i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            # Separate observation and action mask
            observation, action_mask = obs.values()

            if termination or truncation:
                # If there is a winner, keep track, otherwise don't change the scores (tie)
                if (
                    env.rewards[env.possible_agents[0]]
                    != env.rewards[env.possible_agents[1]]
                ):
                    winner = max(env.rewards, key=env.rewards.get)
                    scores[winner] += env.rewards[
                        winner
                    ]  # only tracks the largest reward (winner of game)
                # Also track negative and positive rewards (penalizes illegal moves)
                for a in env.possible_agents:
                    total_rewards[a] += env.rewards[a]
                # List of rewards by round, for reference
                round_rewards.append(env.rewards)
                break
            else:
                if agent == env.possible_agents[0]:
                    act = env.action_space(agent).sample(action_mask)
                else:
                    # Note: PettingZoo expects integer actions # TODO: change chess to cast actions to type int?
                    act = int(
                        model.predict(
                            observation, action_masks=action_mask, deterministic=True
                        )[0]
                    )
            env.step(act)
    env.close()

    # Avoid dividing by zero
    if sum(scores.values()) == 0:
        winrate = 0
    else:
        winrate = scores[env.possible_agents[1]] / sum(scores.values())
    print("Rewards by round: ", round_rewards)
    print("Total rewards (incl. negative rewards): ", total_rewards)
    print("Winrate: ", winrate)
    print("Final scores: ", scores)
    return round_rewards, total_rewards, winrate, scores

if __name__ == "__main__":
    env_fn = connect_four_v3

    env_kwargs = {}

    # Evaluation/training hyperparameter notes:
    # 10k steps: Winrate:  0.76, loss order of 1e-03
    # 20k steps: Winrate:  0.86, loss order of 1e-04
    # 40k steps: Winrate:  0.86, loss order of 7e-06


    # Train a model against itself (takes ~20 seconds on a laptop CPU)
    train_action_mask(env_fn, steps=20_480, seed=0, **env_kwargs)

        # Evaluate 100 games against a random agent (winrate should be ~80%)
    eval_action_mask(env_fn, num_games=100, render_mode=None, **env_kwargs)
