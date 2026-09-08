import torch
from Biped_Train_DiffPhys import train

# Run Tier 1: Ideal Torque (No Limits)
train(
    actuator_type="ideal_unlimited",  # Tier 1
    num_epochs=300,                   # Total training epochs
    batch_size=128,                   # Parallel environments on GPU
    rollout_steps=60                  # 60 steps = 1.20s continuous horizon
)
