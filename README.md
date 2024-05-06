# Adversarial_DRL
Adversarial attacks in deep reinforcement learning 


# Description
Deep Reinforcement Learning (DRL) has a wide range of applications, such as enabling autonomous cars ([Pan et al., 2020](https://arxiv.org/abs/2002.00444)), discovering optimized matrix multiplication algorithms ([Alon et al., 2022](https://www.nature.com/articles/s41586-022-05172-4)), and achieving superhuman performance in board games like Go ([Silver et al., 2016](https://www.nature.com/articles/nature16961)). Reinforcement Learning (RL) is distinctive as it doesn’t rely on traditional datasets, thus avoiding associated data biases. It learns by interacting within simulated environments and utilizes self-play techniques to improve. However, recent research indicates that, similar to other deep learning models, RL is susceptible to adversarial attacks. The goal of this project is to implement and evaluate adversarial deep learning algorithms. After reviewing possible adversarial attack methods ([Liang et al., 2023](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9536399)), the Adversarial Policies method was chosen, introduced by [Gleave et al. (2019)](https://arxiv.org/abs/1905.10615) and further refined by [Wu and Xian (2021)](https://www.usenix.org/conference/usenixsecurity21/presentation/wu-xian). This technique was selected as it mirrors real-life scenarios where an adversary lacks direct access to the victim network and can only manipulate the environment through its actions. Another motivation for this choice was its proven efficacy, demonstrated by outperforming the top algorithms in the game of Go ([Wang et al. (2022)](https://arxiv.org/abs/2211.00241)).

# Theory 
Adversarial Policy Attacks exploit vulnerabilities in deep reinforcement learning (DRL) agents in multi-agent RL settings. In this method, the adversarial RL agent acts according to adversarial policy to create natural observations in the environment that are adversarial for the victim. These cause different activations in the victim's policy network, leading the victim to lose by taking actions that seem random and uncoordinated. 
### Asumptions 
1. The adversary is allowed unlimited black-box access to the actions sampled from victims policy, but do not have any white-box information such as weights or activations.
2. The victim follows fixed stochastic policy with static weights, reflecting common practices in deploying RL models to prevent new issues from developing during retraining.
### Training 
Here's how you can rewrite the paragraph using the GitHub-compatible mathematical expressions mentioned:

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



# Results
- [ ] Add victim policy estimation https://www.usenix.org/conference/usenixsecurity21/presentation/wu-xian 

**Multiagent**
