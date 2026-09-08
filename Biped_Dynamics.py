import torch
from SymPy_Skeleton import skeleton_dynamics

# -----------------------------------------------------------------------------
# Standard 28 BSIP Parameters for 10-DoF Kane Dynamics (Nasr et al. 2024)
# -----------------------------------------------------------------------------
def get_default_body_parameters(batch_size=1, device="cuda:0"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    single_params = torch.tensor([
        10.65, 25.0, 8.5, 3.5, 1.0,               # Masses (kg): [pelvis, torso, thigh, shank, foot]
        0.10, 0.4725, 0.45, 0.45, 0.25, 0.07, 0.17,# Lengths (m): [l_p, l_tr, l_t, l_l, l_f, l_heel, l_toe]
        0.04, 0.2739, 0.20, 0.20, 0.05,           # COMs (m): [c_p, c_tr, c_t, c_l, c_f]
        0.10, 0.4377, 0.15, 0.05, 0.01,           # Inertias (kg*m^2): [I_pz, I_trz, I_tz, I_lz, I_fz]
        9.81, 0.90,                               # g, nominal ground level
        1e6, 0.60, 0.8, 0.04                      # Ground contact: [kc, cc, mu, vs]
    ], dtype=torch.float32, device=device)
    return single_params.unsqueeze(0).repeat(batch_size, 1)

# -----------------------------------------------------------------------------
# Differentiable Macro-Step Simulation (Pose Input -> Dynamic Step)
# -----------------------------------------------------------------------------
def simulate_macro_step(q_curr, u_curr, actuator_state, q_des, actuator, params, dt_sim=0.002, dt_control=0.02):
    """
    Integrates 10-DoF Kane equations and internal actuator PD/dynamics across one 50 Hz macro step.
    
    Args:
        q_curr: [batch_size, 10] Current coordinates
        u_curr: [batch_size, 10] Current speeds
        actuator_state: Internal actuator state (muscle activations for MTG, or None)
        q_des: [batch_size, 7] Target joint angles (pose)
        actuator: BaseActuator instance (IdealTorqueUnlimited, IdealTorqueLimited, or MTG)
        params: [batch_size, 28] Kane dynamics parameters
        dt_sim: Internal integration sub-step (0.002s = 500 Hz)
        dt_control: Control macro timestep (0.02s = 50 Hz)
        
    Returns:
        q_next: [batch_size, 10] Next coordinates
        u_next: [batch_size, 10] Next speeds
        actuator_state_next: Updated actuator internal state
        tau_applied: [batch_size, 7] Applied joint torques (N*m)
    """
    num_substeps = int(round(dt_control / dt_sim))
    
    q_t = q_curr
    u_t = u_curr
    state_t = actuator_state
    tau_applied = torch.zeros_like(q_des)
    
    for _ in range(num_substeps):
        # 1. Actuator converts target pose q_des -> joint torques via internal PD/muscle dynamics
        q_j = q_t[:, 3:10]
        u_j = u_t[:, 3:10]
        tau_total, state_t = actuator.compute_torques(q_j, u_j, q_des, state_t, dt_sim=dt_sim)
        tau_applied = tau_total
        
        # 2. Evaluate Analytical Kane Dynamics
        dq, du = skeleton_dynamics(q_t, u_t, tau_total, params)
        
        # Stability acceleration clamping for stiff ground spring contact
        du = torch.clamp(du, -500.0, 500.0)
        
        # 3. Symplectic Euler Integration
        u_t = torch.clamp(u_t + dt_sim * du, -20.0, 20.0)
        q_t = q_t + dt_sim * u_t
        
    return q_t, u_t, state_t, tau_applied
