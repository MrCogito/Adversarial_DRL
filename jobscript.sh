#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J PPOAgentTrain
#BSUB -n 4
#BSUB -W 10:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o /zhome/59/9/198225/Adversarial_DRL/myjob.out
#BSUB -e /zhome/59/9/198225/Adversarial_DRL/myjob.err
#BSUB -R "span[hosts=1]"

# Activate the virtual environment
source /zhome/59/9/198225/Adversarial_DRL/venvdtu/bin/activate

# <loading of any additional modules, dependencies etc.>
echo "Running PPO Agent Train script..."
python3 /zhome/59/9/198225/Adversarial_DRL/ppo_agent_train.py