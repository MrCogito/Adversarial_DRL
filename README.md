# Adversarial_DRL
Adversarial attacks in deep reinforcement learning 


# Description
Deep Reinforcement Learning (DRL) has a wide range of applications, such as enabling autonomous cars ([Pan et al., 2020](https://arxiv.org/abs/2002.00444)), discovering optimized matrix multiplication algorithms ([Alon et al., 2022](https://www.nature.com/articles/s41586-022-05172-4)), and achieving superhuman performance in board games like Go ([Silver et al., 2016](https://www.nature.com/articles/nature16961)). Reinforcement Learning (RL) is distinctive as it doesn’t rely on traditional datasets, thus avoiding associated data biases. It learns by interacting within simulated environments and utilizes self-play techniques to improve. However, recent research indicates that, similar to other deep learning models, RL is susceptible to adversarial attacks. The goal of this project is to implement and evaluate adversarial deep learning algorithms. After reviewing possible adversarial attack methods ([Liang et al., 2023](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9536399)), the Adversarial Policies method was chosen, introduced by [Lin et al. (2019)](https://arxiv.org/abs/1905.10615) and further refined by [Wu and Xian (2021)](https://www.usenix.org/conference/usenixsecurity21/presentation/wu-xian). This technique was selected as it mirrors real-life scenarios where an adversary lacks direct access to the victim network and can only manipulate the environment through its actions. Another motivation for this choice was its proven efficacy, demonstrated by outperforming the top algorithms in the game of Go ([Wang et al. (2022)](https://arxiv.org/abs/2211.00241)).

# Implementation


# Results



# Checklist 
- [X] Write description
- [ ] write implementation
- [ ] write resutls
- [X] Connect to HCP
**Code**
- [X] Run 2 random pong agents env 
- [X] Run human vs agent env
- [X] Win vs random agent as human
- [X] setup trainig for PPO agent
- [ ] Train superhuman pong agent using PPO that will bear human
- [ ] Set up training for adversarial agent - formulate it as 1player markow problem
- [ ] Add victim policy estimation https://www.usenix.org/conference/usenixsecurity21/presentation/wu-xian 

**Multiagent**
