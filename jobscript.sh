#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J PPOAgentTrain
#BSUB -n 4
#BSUB -W 10:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o /zhome/59/9/198225/Desktop/Adversarial_DRLa/myjob.out
#BSUB -e /zhome/59/9/198225/Desktop/Adversarial_DRLa/myjob.err
#BSUB -R "span[hosts=1]"

# Activate the virtual environment
source /zhome/59/9/198225/Desktop/Adversarial_DRL/project-env/bin/activate
# <loading of any additional modules, dependencies etc.>
echo "Running scri script..."
python3 /zhome/59/9/198225/Desktop/Adversarial_DRL/Adversarial_DRL/tutorialcode.py

