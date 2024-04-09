from pettingzoo.atari import pong_v3
import random

num_games = 1
wins_first_0 = 0
wins_second_0 = 0

for game in range(num_games):
    # To not display game
    env =  pong_v3.parallel_env(render_mode="human")
    #env = pong_v3.parallel_env()
    observations, infos = env.reset()

    # Initialize score for only one agent (since the other agent's score is just the negative of this one)
    score_first_0 = 0

    while env.agents:
        actions = {}
        for agent in env.agents:
            if agent == 'first_0':
                actions[agent] = random.choice(list(range(6)))  # Move left
            elif agent == 'second_0':
                actions[agent] = random.choice(list(range(6)))   # Move right
            else:
                raise ValueError(f"Unknown agent: {agent}")

        observations, rewards, terminations, truncations, infos = env.step(actions)
        score_first_0 += rewards['first_0']

    env.close()

    # Adjust scores to reflect the 21-point game format
    if(score_first_0 > 0):
        score_first_0_adjusted = 21 
        score_second_0_adjusted = 21 - score_first_0
    else:
        score_second_0_adjusted = 21 
        score_first_0_adjusted = 21 + score_first_0

    # Update win counters
    if score_first_0_adjusted > score_second_0_adjusted:
        wins_first_0 += 1
    elif score_first_0_adjusted < score_second_0_adjusted:
        wins_second_0 += 1

# Print the number of wins for each agent
print(f"first_0 won {wins_first_0} times.")
print(f"second_0 won {wins_second_0} times.")