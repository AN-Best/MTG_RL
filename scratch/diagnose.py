import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from SymPy_Skeleton import skeleton_dynamics
from Biped_Simulation import MuscleTorqueGenerator

# Let's inspect the first step values
q = torch.zeros(7)
u = torch.zeros(7)
q_ref = torch.zeros(6)
params = torch.tensor([
    70.0, 10.0, 5.0, 1.5,   # Masses
    0.8, 0.45, 0.45, 0.25, 0.05, 0.2, # Lengths
    0.4, 0.2, 0.2, 0.08,    # COMs
    2.5, 0.1, 0.05, 0.01,   # Inertias
    9.81, 0.95,             # g, h_g
    1e5, 0.1, 0.6, 0.01     # kc, cc, mu, vs
], dtype=torch.float32)

mtg = MuscleTorqueGenerator()
q_j = q[1:7]
u_j = u[1:7]
tau_j = mtg.compute_torques(q_j, u_j, q_ref)
# Note: we need 6 torques to pass to skeleton_dynamics, not 7!
print("Computed joint torques (6):", tau_j)

# Evaluate mass matrix and forcing vector directly from lambdified functions
# to see where NaNs are introduced
from SymPy_Skeleton import M_func, F_func, q_vars, u_vars, torque_vars, parameter_vars

# Convert inputs to torch tensors
def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)

q_tensor = to_tensor(q)
u_tensor = to_tensor(u)
torques_tensor = to_tensor(tau_j)
params_tensor = to_tensor(params)

M_num = M_func(q_tensor, u_tensor, torques_tensor, params_tensor)
F_num = F_func(q_tensor, u_tensor, torques_tensor, params_tensor)

print("M_num elements:")
for i, row in enumerate(M_num):
    for j, val in enumerate(row):
        if torch.isnan(torch.as_tensor(val)):
            print(f"NaN in M_num[{i}][{j}]")

print("F_num elements:")
for i, row in enumerate(F_num):
    val = row[0]
    if torch.isnan(torch.as_tensor(val)):
        print(f"NaN in F_num[{i}]")

dq, du = skeleton_dynamics(q, u, tau_j, params)
print("dq/dt:", dq)
print("du/dt:", du)
