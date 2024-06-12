# Description
Deep Reinforcement Learning (DRL) has a wide range of applications, such as enabling autonomous cars ([Pan et al., 2020](https://arxiv.org/abs/2002.00444)), discovering optimized matrix multiplication algorithms ([Alon et al., 2022](https://www.nature.com/articles/s41586-022-05172-4)), and achieving superhuman performance in board games like Go ([Silver et al., 2016](https://www.nature.com/articles/nature16961)). Reinforcement Learning (RL) is distinctive as it doesn’t rely on traditional datasets, thus avoiding associated data biases. It learns by interacting within simulated environments and utilizes self-play techniques to improve. However, recent research indicates that, similar to other deep learning models, RL is susceptible to adversarial attacks. The goal of this project is to implement and evaluate adversarial deep learning algorithms. After reviewing possible adversarial attack methods ([Liang et al., 2023](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9536399)), the Adversarial Policies method was chosen, introduced by [Gleave et al. (2019)](https://arxiv.org/abs/1905.10615) and further refined by [Wu and Xian (2021)](https://www.usenix.org/conference/usenixsecurity21/presentation/wu-xian). This technique was selected as it mirrors real-life scenarios where an adversary lacks direct access to the victim network and can only manipulate the environment through its actions. Another motivation for this choice was its proven efficacy, demonstrated by outperforming the top algorithms in the game of Go ([Wang et al. (2022)](https://arxiv.org/abs/2211.00241)).

# Theory 
Adversarial Policy Attacks exploit vulnerabilities in deep reinforcement learning (DRL) agents in multi-agent RL settings. In this method, the adversarial RL agent acts according to adversarial policy to create natural observations in the environment that are adversarial for the victim. These cause different activations in the victim's policy network, leading the victim to lose by taking actions that seem random and uncoordinated. 
### Assumptions 
1. The adversary is allowed unlimited access to the actions sampled from the victims' policy (gray-box access) but does not have any white-box information such as weights or activations.
2. The victim follows a fixed stochastic policy with static weights, reflecting common practices in deploying RL models to prevent new issues from developing during retraining.

### Training Process

For multi-agent competitive games, the training setup can be modeled as a two-player Markov game. The game $M$ is defined as:

$$M = (S, (A_\alpha, A_\nu), T, (R_\alpha, R_\nu))$$

where:
- $S$: set of states
- $A_\alpha$ and $A_\nu$: action sets for the adversary and the victim
- $T$: joint state transition function
- $R_\alpha$ and $R_\nu$: reward functions for the adversary and the victim

Since the victim's policy is fixed, this reduces to a single-player Markov Decision Process (MDP) for the adversary. The reduced MDP, $M_\alpha$, is defined as:

$$M_\alpha = (S, A_\alpha, T_\alpha, R'_\alpha)$$

where:
- $T_\alpha(s, a_\alpha)$ is derived from the original two-player transition function:

$$T_\alpha(s, a_\alpha) = T(s, a_\alpha, a_\nu)$$

Here, $a_\alpha$ is sampled from the victim's stochastic policy $\pi_\nu(\cdot \mid s)$. $R_\alpha'(s, a_\alpha, s')$ is derived from the two-player reward function:

$$
R_\alpha'(s, a_\alpha, s') = R_\alpha(s, a_\alpha, a_\nu, s')
$$

The goal of the adversary is to find a policy $\pi_\alpha$ that maximizes the sum of discounted rewards:

$$\max_{\pi_\alpha} \sum_{t=0}^{\infty} \gamma^t R_\alpha(s^{(t)}, a^{(t)}_\alpha, s^{(t+1)})     \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \       (1)     $$ 

where $s^{(t+1)}$ is the next state, $a^{(t)}_\alpha$ is the adversary's action, and $\gamma$ is the discount factor.

The MDP dynamics will be unknown to the adversary because it has access only to actions sampled from the victim's policy. Thus, the adversary must solve a reinforcement learning problem to learn a policy that maximizes its rewards.


The study by [Gleave et al. (2019)](https://arxiv.org/abs/1905.10615) demonstrated the effectiveness of adversarial policies in zero-sum robotics games using the [MuJoCo](https://mujoco.org/) environment. During the training, Proximal Policy Optimization was used to maximize Equation 1 described above.  After each episode adversary was given sparse reward - positive for wins and negative for losses and ties. Those trained Adversarial policies reliably won against most victim policies and outperformed the pre-trained [Zoo baseline](https://arxiv.org/abs/1710.03748)  for a majority of environments and victims.

[Wu and Xian (2021)](https://www.usenix.org/conference/usenixsecurity21/presentation/wu-xian). expanded on earlier research by introducing a method where the adversarial policy not only aims to win but also maximizes the deviation in the victim's actions, leading to suboptimal performance. They enhanced the standard loss function, traditionally based only on game outcomes, with a new term comprising two components: the first measures the deviation in the victim's policy, which the adversary tried to maximize; the second measures the variation added to the observations, which the adversary tried to minimize.
In simpler terms, Adversary aimed to  maximize the deviation of victims' actions using minimal effort. This method, evaluated in [MuJoCo](https://mujoco.org/) and  [RoboSchool Pong]( https://openai.com/index/roboschool/), improved win rates and outperformed the standard Adversarial Policy Attack.

# Implementation

### Training Environemnt
While selecting a training environment, several factors were taken into consideration:
1. The environment must be competitive and support multi-agent interaction, allowing control over the policies of multiple agents.
2. The game's dimensionality can not be too small, since this attack method achieve better results in high-dimensional games [Gleave et al. (2019)](https://arxiv.org/abs/1905.10615).
3. The game's dimensionality can not be to big due to time and resource limitations.

This method proved effective in both continuous (MuJoCo) and discrete (Go) action spaces, showing that it was not limited by the nature of the action space—discrete or continuous.

Given those considerations, [Petting-Zoo Connect Four](https://pettingzoo.farama.org/tutorials/sb3/connect_four/) environment has been chosen. 

Game rules from Petting-zoo documentation:
> Connect Four is a 2-player turn based game, where players must connect four of their tokens vertically, horizontally or diagonally. The players drop their respective token in a column of a standing grid, where each token will fall until it reaches the bottom of the column or reaches an existing token. Players cannot place a token in a full column, and the game ends when either a player has made a sequence of 4 tokens, or when all 7 columns have been filled.

State space of (6x7x3), and action space of 7 seemed to be big enough to perform adversarial policy attack. 
Atari pong environment was also considered, but because its observation size (210x160x3) is larger and there were no pre-trained deep-learning agents available, it was not chosen due to time and resource limitations during this project. 


Both agents were using same architecture([DQN with masking](https://docs.agilerl.com/en/latest/api/algorithms/dqn.html))
The architecture is a Convolutional Neural Network with an input layer accepting a 6x7 grid and two hidden layers, each with 64 units followed by an output layer of size 7 corresponding to possible actions in the environment.
All hyperparameters: 
'''
        
        INIT_HP = {
            "POPULATION_SIZE": 1,
            "ALGO": "DQN",  # Algorithm
            "DOUBLE": True,
            "BATCH_SIZE": 256,  # Batch size
            "LR": 0.00005,  # Learning rate
            "GAMMA": 0.99,  # Discount factor
            "MEMORY_SIZE": 100000,  # Max memory buffer size
            "LEARN_STEP": 1,  # Learning frequency
            "N_STEP": 1,  # Step number to calculate td error
            "PER": False,  # Use prioritized experience replay buffer
            "ALPHA": 0.6,  # Prioritized replay buffer parameter
            "TAU": 0.01,  # For soft update of target parameters
            "BETA": 0.4,  # Importance sampling coefficient
            "PRIOR_EPS": 0.000001,  # Minimum priority for sampling
            "NUM_ATOMS": 51,  # Unit number of support
            "V_MIN": 0.0,  # Minimum value of support
            "V_MAX": 200.0,  # Maximum value of support
        }
              NET_CONFIG = {
            "arch": "cnn", 
            "hidden_size": [64, 64],  
            "channel_size": [128],  
            "kernel_size": [4],  
            "stride_size": [1],  
            "normalize": False,  
        }
'''

To maintain a consistent perspective regardless of player turn, observations are flipped and transformed accordingly. This step ensures that the agent always perceives the board from the same point of view, simplifying the learning process.


### Training procedure
1. Train the victim agent using self-play (playing against an older version of itself that was updated every 500 epochs) with 1000 epochs warm-up against the random agent, until the victim agent achieves decent performance. 
2. Save victim agent
3. Train adversarial agents using frozen victim's agent policy as an opponent.

Agent can score:

+1 for win

0 for draw

-1 for lose

### Results 
**Victim**

After 4000 epochs(evaluation on chart is performed every 10 epochs) agent learned simple strategy that allowed him to achieve score on average 0.78 point per game. 
<div align="center">
<img src="https://github.com/MrCogito/Adversarial_DRL/assets/22586533/b8c8e179-fa09-4dc3-a5d5-2d4d76402fdf" width="85%" height="85%">
</div>
After looking at win/draw/lose statistics agent's average score of 0.78 reflects frequent draws rather than losses.
<div align="center">
<img src="https://github.com/MrCogito/Adversarial_DRL/assets/22586533/bc378c73-4ba4-4614-be5b-b01ebcdfe1f6" width="85%" height="85%">
</div>
By analyzing agent games it can be seen that it learned a simple strategy of building vertical towers to win
<div align="center">
<p><b>Victim vs Random (all games are won by victim)</b></p>
<img src="https://github.com/MrCogito/Adversarial_DRL/assets/22586533/5cfbbd18-886e-4a95-a3d0-4eab7d3788ed">
</div>

**Adversary**

Adversarial agent was trained against victim agent from epoch 6000. 
Around epoch 400 it learned how to exploit victim strategy and achieve 98.8% win rate. 
<div align="center">
<img src="https://github.com/MrCogito/Adversarial_DRL/assets/22586533/287416ab-a547-46a1-9761-62263fe4accc" width="85%" height="85%">
</div>
To ensure that this is not because of "lucky" seed, at the same time, Adversary agent was evaluated 
gainst rule-based opponent and it performed much worse that Victim 

<div align="center">
<img src="https://github.com/MrCogito/Adversarial_DRL/assets/22586533/0703d171-034d-47e5-b265-3454b2eb5e58" width="85%" height="85%">
</div>

By analyzing game, we can see that when Victim was 1st to play, Adversary learned how to block vertical win, and if Victim was 2nd Adversay found interesting pattern that eventually led to win. 

<div align="center">
<p><b>Adversary vs Victim (Victim moves 1st)</b></p>
<img src="https://github.com/MrCogito/Adversarial_DRL/assets/22586533/7855d49d-90f9-4179-a512-dc00987e7453">
</div>

<div align="center">
<p><b>Adversary vs Victim (Adversary moves 2nd)</b></p>
<img src="https://github.com/MrCogito/Adversarial_DRL/assets/22586533/48524384-2d10-49aa-932d-916ebbe17596">
</div>

What is also interesting - when playing against a rule-based opponent that was not trying to push vertical win, the Adversary had trouble finding winning patterns and blocking horizontal wins, which resulted in more "random" looking games.
<div align="center">
<p><b>Adversary vs rule-based (Adversary moves 1st)</b></p>

<img src="https://github.com/MrCogito/Adversarial_DRL/assets/22586533/f194848b-fe1e-466e-a58e-91dfb248f954">
</div>

Results are summarized in table below:
| Match       | Average score| Winner  |
|-------------|----------|---------|
| Victim vs rule-based| 78%      | Victim |
| Adversarial vs rule-based| 55.4%      | Adversarial |
| Adversarial vs Victim| 98.8%      | Adversarial |

### Discussion 
Online discussion that included the author of Adversarial Policy Attack, and primary author of KataGo under  [Adversarial Policies Beat Professional-Level Go AIs](https://www.reddit.com/r/MachineLearning/comments/yjryrd/n_adversarial_policies_beat_professionallevel_go/) post give insights why this method works. 
Researchers agreed that AlphaZero and similar self-play RL applications in general does not give superhuman performance. They do so only in the in-distribution subset of game states that are similar to the one explored during self-play.
They added that there are currently no common methods of exploration or adding noise that can ensure that all of the important parts of state space are covered - especially in environments with large state spaces.
Additionally, Adam Gleave argue that pure neural net scaling does seem like it's enough to get good average-case performance on-distribution for many tasks and more attention should be put on neurosymbolic/search-like methods. 

Even though we don’t yet have a way to cover all state spaces completely, there is the possibility of making models more robust to make it harder and more expensive to attack. In the study about the game Go, attackers managed to beat the KataGo model using just about 1% of the computing power needed to train the KataGo model. In this project, it took about 10% of that compute power. Making attack more compute-intensive, might discourage attackers because of the lower cost-benefit ratio. 

Possible defense strategies include:
- Adding more search to models. The disadvantage of this approach is that more computing will be needed during inference.
- Increase the diversity of opponents during self-play training - described in this blog post[Defending against Adversarial Policies in Reinforcement Learning with Alternating Training](https://forum.effectivealtruism.org/posts/YscrJFofd6S8eJGS8/defending-against-adversarial-policies-in-reinforcement)
- Running multiple iterations of attack and fine-tuning on games vs adversarial. In this paper [Wang et al. (2022)](https://arxiv.org/abs/2211.00241) researchers demonstrated that one iteration of fine-tuning does not cover enough game space to make KataGo agent robust enough. An additional drawback of this approach is that too much fine-tuning against adversaries can decrease agent performance against standard opponents.

The main challenge of an Adversarial Policy Attack is that it requires unlimited gray-box access to the victim (Access to actions sampled from victim policy).
For AI systems engaged in games like Go, chess, or poker, it is plausible that an attacker could gain such access. However, in applications involving robotics or autonomous vehicles, obtaining gray-box access may be considerably more challenging, limiting the feasibility of such attacks.

Because of that future improvements in the Adversarial Policy Attack should focus on finding victim's out of distribution states, in which civtim is vulnerable, with a limited number of victim's actions observations. 

- One approach might involve building an approximate victim model of the victim. Then, perform the attack and transfer it to the real victim model. Here ([Wang et al. (2022)](https://arxiv.org/abs/2211.00241)) it was demonstrated that KataGo attack transferred to other Go models. 
- Other approaches can focus on more effective sampling of victims actions to avoid similar scenarios from occurring frequently during the training.

### Summary and project limitations 

During the project, the deep reinforcement learning model was trained and successfully attacked using Adversarial Policy method. 
Because there were no pre-trained agents, the victim model had to be developed from scratch. Constraints on time and resources resulted in a relatively small network size and limited training duration for the victim model. With additional time for model tuning, extended training periods, and an increase in network size, the performance of the victim's network could have been significantly enhanced.

## Additional comments 
- Training code has quick-fixes to [bug in Petting-Zoo](https://github.com/Farama-Foundation/PettingZoo/issues/114) which causes the agent to learn how to lose the game instead of how to win. Example line with fix:
```
cumulative_reward = -cumulative_reward
```
- Not all models are uploaded to github - all files can be found on DTU server /zhome/59/9/198225/Desktop/Adversarial_DRL/Adversarial_DRL
- pong_old folder os there because originally I tried to use the atari-pong environment, but there were no trained agents in petting-zoo environment and agent from other libraries eg. StableBaseline were not compatible with petting-zoo. 
Additionally there were no build-in agent to train model in PZ, so I decided that training atari-pong model from scratch will take too much time. 


