# Adversarial_DRL
Adversarial attacks in deep reinforcement learning 


# Description
Deep Reinforcement Learning (DRL) has a wide range of applications, such as enabling autonomous cars ([Pan et al., 2020](https://arxiv.org/abs/2002.00444)), discovering optimized matrix multiplication algorithms ([Alon et al., 2022](https://www.nature.com/articles/s41586-022-05172-4)), and achieving superhuman performance in board games like Go ([Silver et al., 2016](https://www.nature.com/articles/nature16961)). Reinforcement Learning (RL) is distinctive as it doesn’t rely on traditional datasets, thus avoiding associated data biases. It learns by interacting within simulated environments and utilizes self-play techniques to improve. However, recent research indicates that, similar to other deep learning models, RL is susceptible to adversarial attacks. The goal of this project is to implement and evaluate adversarial deep learning algorithms. After reviewing possible adversarial attack methods ([Liang et al., 2023](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9536399)), the Adversarial Policies method was chosen, introduced by [Gleave et al. (2019)](https://arxiv.org/abs/1905.10615) and further refined by [Wu and Xian (2021)](https://www.usenix.org/conference/usenixsecurity21/presentation/wu-xian). This technique was selected as it mirrors real-life scenarios where an adversary lacks direct access to the victim network and can only manipulate the environment through its actions. Another motivation for this choice was its proven efficacy, demonstrated by outperforming the top algorithms in the game of Go ([Wang et al. (2022)](https://arxiv.org/abs/2211.00241)).

# Theory 
Adversarial Policy Attacks exploit vulnerabilities in deep reinforcement learning (DRL) agents in multi-agent RL settings. In this method, the adversarial RL agent acts according to adversarial policy to create natural observations in the environment that are adversarial for the victim. These cause different activations in the victim's policy network, leading the victim to lose by taking actions that seem random and uncoordinated. 
### Asumptions 
1. The adversary is allowed unlimited black-box access to the actions sampled from victims policy, but do not have any white-box information such as weights or activations.
2. The victim follows fixed stochastic policy with static weights, reflecting common practices in deploying RL models to prevent new issues from developing during retraining.

### Training Process

Here's the updated version following the guidelines:

For the [Connect four environment](https://pettingzoo.farama.org/environments/classic/connect_four/), the training setup is modeled as a two-player Markov game. The game $M$ is defined as:

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

$$\max_{\pi_\alpha} \sum_{t=0}^{\infty} \gamma^t R_\alpha(s^{(t)}, a^{(t)}_\alpha, s^{(t+1)})$$

where $s^{(t+1)}$ is the next state, $a^{(t)}_\alpha$ is the adversary's action, and $\gamma$ is the discount factor.

The MDP dynamics will be unknown to the adversary because the victim's policy is a black-box. Thus, the adversary must solve a reinforcement learning problem to learn a policy that maximizes its rewards.


# Implementation

### Training Environemnt
While selecting a training environment, several factors were taken into consideration:
1. The environment must be competitive and support multi-agent interaction, allowing control over the policies of multiple agents.
2. The game's dimensionality can not be too small, since this attack method achieve better results in high-dimensional games [Gleave et al. (2019)](https://arxiv.org/abs/1905.10615).
3. The game's dimensionality can not be to big due to time and resources limitations.
Given those considerations, [Petting-Zoo Connect Four](https://pettingzoo.farama.org/tutorials/sb3/connect_four/) environment has been chosen.
Both agents were using same architecture([DQN with masking](https://docs.agilerl.com/en/latest/api/algorithms/dqn.html)) and hyperparameters.

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
After 4000 epochs(evaluation on chart is performed every 10 epochs) agent learned simple strategy that allowed him to achieve score on average 0.8 point per game. 
![image](https://github.com/MrCogito/Adversarial_DRL/assets/22586533/b8c8e179-fa09-4dc3-a5d5-2d4d76402fdf)
The agent's average score is only 0.8, which reflects frequent draws rather than losses.
![image](https://github.com/MrCogito/Adversarial_DRL/assets/22586533/bc378c73-4ba4-4614-be5b-b01ebcdfe1f6)
By analyzing agent games it can be seen that it learned a simple strategy of building vertical towers to win
Victim vs Random (all games are won by victim)
![VictimVsRandom](https://github.com/MrCogito/Adversarial_DRL/assets/22586533/5cfbbd18-886e-4a95-a3d0-4eab7d3788ed)

**Adversary**
Adversarial agent was trained against victim agent from epoch 6000. 
Around epoch ~400 it learned how to exploit victim strategy and achieve almost 100% win rate. 
![image](https://github.com/MrCogito/Adversarial_DRL/assets/22586533/287416ab-a547-46a1-9761-62263fe4accc)
To ensure that this is not because of "lucky" seed, at the same time, Adversary agent was evaluated against rule-based opponent and it performed much worse that Victim 
![image](https://github.com/MrCogito/Adversarial_DRL/assets/22586533/0703d171-034d-47e5-b265-3454b2eb5e58)

By analyzing game, we can see that when Victim was 1st to play, Adversary learned how to block vertical win, and if Victim was 2nd Adversay found interesting pattern that eventually led to win. 

Adversary vs Victim (Victim moves 1st)
![Adversary_victim_2](https://github.com/MrCogito/Adversarial_DRL/assets/22586533/48524384-2d10-49aa-932d-916ebbe17596)

Adversary vs Victim (Adversary moves 1st)
![Adversary_victim_1](https://github.com/MrCogito/Adversarial_DRL/assets/22586533/c7204ae8-6b5f-409a-8874-f0d213e546e4)

What is also interesting - when playing against a rule-based opponent that was not trying to push vertical win, the Adversary had trouble finding winning patterns and blocking horizontal wins, which resulted in more "random" games.
Adversary vs rule-based
![Adv_random](https://github.com/MrCogito/Adversarial_DRL/assets/22586533/f194848b-fe1e-466e-a58e-91dfb248f954)

Results are summarized in table below:
| Match       | Win Rate | Winner  |
|-------------|----------|---------|
| Agent 1 vs 2| 50%      | Agent 2 |
| Agent 1 vs 3| 75%      | Agent 1 |
| Agent 2 vs 3| 55%      | Agent 2 |

### Discussion
 -- OOD
 --Future work (train more/stronger agents)
 -- 
### Add 
- finding out of distribution states
- theory for dqn


# Results
- [ ] Add victim policy estimation https://www.usenix.org/conference/usenixsecurity21/presentation/wu-xian 

**Multiagent**
