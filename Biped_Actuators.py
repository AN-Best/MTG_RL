import torch
import torch.nn as nn
import numpy as np

# -----------------------------------------------------------------------------
# Base Actuator Interface (Pose Input -> Torque Output)
# -----------------------------------------------------------------------------
class BaseActuator(nn.Module):
    def __init__(self, device="cuda:0"):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
    def compute_torques(self, q_j, u_j, q_des, state, dt_sim):
        """
        Computes joint torques from target pose q_des via internal PD control.
        
        Args:
            q_j: Joint angles [batch_size, 7] (rad)
            u_j: Joint angular velocities [batch_size, 7] (rad/s)
            q_des: Target joint angles [batch_size, 7] (rad)
            state: Internal actuator state (e.g. muscle activations) or None
            dt_sim: Simulation timestep (s)
            
        Returns:
            tau_total: [batch_size, 7] Net torque applied to joints (N*m)
            next_state: Updated actuator internal state
        """
        raise NotImplementedError

# -----------------------------------------------------------------------------
# Tier 1: Ideal Torque (No Limits) - High Stiffness Direct Tracking
# -----------------------------------------------------------------------------
class IdealTorqueUnlimitedActuator(BaseActuator):
    """
    Tier 1 Actuator: High-gain, critically damped linear PD tracking from target pose q_des.
    Unconstrained control authority without saturation limits.
    """
    def __init__(self, device="cuda:0"):
        super().__init__(device=device)
        # Dedicated High-Stiffness Gains [Lumbar, Hip_L, Knee_L, Ankle_L, Hip_R, Knee_R, Ankle_R]
        self.register_buffer("Kp", torch.tensor([600.0, 600.0, 1000.0, 100.0, 600.0, 1000.0, 100.0], dtype=torch.float32, device=self.device))
        self.register_buffer("Kd", torch.tensor([60.0, 100.0, 60.0, 5.0, 100.0, 60.0, 5.0], dtype=torch.float32, device=self.device))
        self.actuator_type = "ideal_unlimited"
        
    def compute_torques(self, q_j, u_j, q_des, state=None, dt_sim=0.002):
        tau = self.Kp * (q_des - q_j) - self.Kd * u_j
        return tau, None

# -----------------------------------------------------------------------------
# Tier 2: Ideal Torque (Peak Isometric Limits) - Saturation-Scaled Gains
# -----------------------------------------------------------------------------
class IdealTorqueLimitedActuator(BaseActuator):
    """
    Tier 2 Actuator: PD tracking scaled to biological peak isometric limits T_max.
    Gains are sized so that linear control authority spans typical tracking errors (~0.3 rad)
    without immediate chattering saturation.
    """
    def __init__(self, device="cuda:0"):
        super().__init__(device=device)
        # Sized to peak isometric capacities [Lumbar, Hip_L, Knee_L, Ankle_L, Hip_R, Knee_R, Ankle_R]
        self.register_buffer("Kp", torch.tensor([400.0, 400.0, 600.0, 150.0, 400.0, 600.0, 150.0], dtype=torch.float32, device=self.device))
        self.register_buffer("Kd", torch.tensor([40.0, 60.0, 40.0, 8.0, 60.0, 40.0, 8.0], dtype=torch.float32, device=self.device))
        
        # Peak isometric torque capacities (N*m)
        self.register_buffer("T_max", torch.tensor([300.0, 400.0, 500.0, 250.0, 400.0, 500.0, 250.0], dtype=torch.float32, device=self.device))
        
        # 7 Joint limits [Lumbar, Hip_L, Knee_L, Ankle_L, Hip_R, Knee_R, Ankle_R] (rad)
        self.register_buffer("limits_low", torch.tensor([-0.469, -0.5, -2.0, -0.5, -0.5, -2.0, -0.5], dtype=torch.float32, device=self.device))
        self.register_buffer("limits_high", torch.tensor([0.620, 1.5, 0.0, 0.5, 1.5, 0.0, 0.5], dtype=torch.float32, device=self.device))
        self.actuator_type = "ideal_limited"
        
    def compute_torques(self, q_j, u_j, q_des, state=None, dt_sim=0.002):
        q_des_clamped = torch.clamp(q_des, min=self.limits_low, max=self.limits_high)
        tau_pd = self.Kp * (q_des_clamped - q_j) - self.Kd * u_j
        tau_saturated = torch.clamp(tau_pd, -self.T_max, self.T_max)
        return tau_saturated, None

# -----------------------------------------------------------------------------
# Tier 3: Full Musculoskeletal Muscle Torque Generator (MTG) - Sensory-Motor Reflex
# -----------------------------------------------------------------------------
class MTGActuator(BaseActuator):
    """
    Tier 3 Actuator: Full Muscle Torque Generator (MTG) dynamics.
    - Uses sensory-motor reflex gains (K_reflex, D_reflex) to map target pose q_des -> neural excitation u_des in [-1, 1]
    - Intrinsic muscle force-velocity (f_V) and passive curves provide primary physical damping & stiffness
    - 1st-order muscle activation dynamics ODE (tau_act = 20ms, tau_deact = 40ms)
    """
    def __init__(self, device="cuda:0"):
        super().__init__(device=device)
        self.actuator_type = "mtg"
        
        # Sensory-Motor Reflex Gains: Maps angle error (rad) & angular speed (rad/s) -> neural excitation u_des in [-1, 1]
        # K_reflex: ~0.25 - 0.35 rad error produces 100% full muscle excitation
        self.register_buffer("K_reflex", torch.tensor([3.0, 3.5, 3.5, 2.5, 3.5, 3.5, 2.5], dtype=torch.float32, device=self.device))
        # D_reflex: Low derivative feedback since Hill-type force-velocity curve provides intrinsic physical damping
        self.register_buffer("D_reflex", torch.tensor([0.15, 0.20, 0.15, 0.05, 0.20, 0.15, 0.05], dtype=torch.float32, device=self.device))
        
        # Activation time constants (He et al. 1991 / Millard et al. 2013)
        self.tau_act = 0.020
        self.tau_deact = 0.040
        self.a_min = 0.01
        
        # MTG Parameters for 7 Actuated Joints [Lumbar, Hip_L, Knee_L, Ankle_L, Hip_R, Knee_R, Ankle_R]
        # Flexion / Extension peak isometric torques (N*m)
        self.register_buffer("T_max_flex", torch.tensor([300.0, 400.0, 200.0, 80.0, 400.0, 200.0, 80.0], dtype=torch.float32, device=self.device))
        self.register_buffer("T_max_ext", torch.tensor([400.0, 300.0, 500.0, 250.0, 300.0, 500.0, 250.0], dtype=torch.float32, device=self.device))
        
        # Optimal joint angles theta_opt (rad)
        self.register_buffer("q_opt_flex", torch.tensor([0.0, 0.45, -0.60, 0.0, 0.45, -0.60, 0.0], dtype=torch.float32, device=self.device))
        self.register_buffer("q_opt_ext", torch.tensor([0.0, -0.20, -0.80, 0.15, -0.20, -0.80, 0.15], dtype=torch.float32, device=self.device))
        
        # Gaussian width parameters w
        self.register_buffer("w_flex", torch.tensor([0.80, 0.65, 0.90, 0.50, 0.65, 0.90, 0.50], dtype=torch.float32, device=self.device))
        self.register_buffer("w_ext", torch.tensor([0.80, 0.65, 0.90, 0.50, 0.65, 0.90, 0.50], dtype=torch.float32, device=self.device))
        
        # Maximum joint angular velocities omega_max (rad/s)
        self.register_buffer("u_max_flex", torch.tensor([12.0, 15.0, 18.0, 14.0, 15.0, 18.0, 14.0], dtype=torch.float32, device=self.device))
        self.register_buffer("u_max_ext", torch.tensor([12.0, 15.0, 18.0, 14.0, 15.0, 18.0, 14.0], dtype=torch.float32, device=self.device))
        
        # Passive joint stiffness parameters [k1, k2, theta_1, theta_2]
        self.register_buffer("k1_pass", torch.tensor([250.0, 200.0, 300.0, 150.0, 200.0, 300.0, 150.0], dtype=torch.float32, device=self.device))
        self.register_buffer("k2_pass", torch.tensor([250.0, 200.0, 300.0, 150.0, 200.0, 300.0, 150.0], dtype=torch.float32, device=self.device))
        self.register_buffer("q1_pass", torch.tensor([-0.40, -0.40, -1.80, -0.40, -0.40, -1.80, -0.40], dtype=torch.float32, device=self.device))
        self.register_buffer("q2_pass", torch.tensor([0.50, 1.20, 0.05, 0.40, 1.20, 0.05, 0.40], dtype=torch.float32, device=self.device))

    def update_activation(self, u_des, a_curr, dt_sim):
        """1st-order muscle activation dynamics ODE integration."""
        u_clamped = torch.clamp(u_des, -1.0, 1.0)
        da = torch.where(
            torch.abs(u_clamped) >= a_curr,
            (torch.abs(u_clamped) - a_curr) / self.tau_act,
            (torch.abs(u_clamped) - a_curr) / self.tau_deact
        )
        a_next = torch.clamp(a_curr + dt_sim * da, self.a_min, 1.0)
        return a_next

    def compute_torques(self, q_j, u_j, q_des, a_curr=None, dt_sim=0.002):
        # 1. Sensory-Motor Reflex mapping: q_des -> neural excitation u_des in [-1, 1]
        u_des = torch.clamp(self.K_reflex * (q_des - q_j) - self.D_reflex * u_j, -1.0, 1.0)
        
        if a_curr is None:
            a_curr = torch.full_like(u_des, self.a_min)
            
        a_next = self.update_activation(u_des, a_curr, dt_sim)
        
        # 2. Force-Length Relationships f_L(q) (Gaussian)
        f_L_flex = torch.exp(-((q_j - self.q_opt_flex) / self.w_flex) ** 2)
        f_L_ext = torch.exp(-((q_j - self.q_opt_ext) / self.w_ext) ** 2)
        
        # 3. Force-Velocity Relationships f_V(u) (Hill-type curve)
        norm_u_flex = torch.clamp(u_j / self.u_max_flex, -1.0, 1.0)
        norm_u_ext = torch.clamp(-u_j / self.u_max_ext, -1.0, 1.0)
        
        f_V_flex = torch.where(
            norm_u_flex >= 0.0,
            torch.clamp((1.0 - norm_u_flex) / (1.0 + 4.0 * norm_u_flex + 1e-6), 0.0, 1.0),
            torch.clamp(1.0 - 1.4 * norm_u_flex / (1.0 - 4.0 * norm_u_flex + 1e-6), 1.0, 1.6)
        )
        f_V_ext = torch.where(
            norm_u_ext >= 0.0,
            torch.clamp((1.0 - norm_u_ext) / (1.0 + 4.0 * norm_u_ext + 1e-6), 0.0, 1.0),
            torch.clamp(1.0 - 1.4 * norm_u_ext / (1.0 - 4.0 * norm_u_ext + 1e-6), 1.0, 1.6)
        )
        
        # 4. Active Torque T_active = sign(u_des) * a * T_max * f_L * f_V
        tau_flex = a_next * self.T_max_flex * f_L_flex * f_V_flex
        tau_ext = a_next * self.T_max_ext * f_L_ext * f_V_ext
        tau_active = torch.where(u_des >= 0.0, tau_flex, -tau_ext)
        
        # 5. Passive Torque (non-linear joint limit springs)
        tau_passive = - self.k1_pass * torch.relu(self.q1_pass - q_j) ** 2 + self.k2_pass * torch.relu(q_j - self.q2_pass) ** 2
        
        tau_total = tau_active + tau_passive
        return tau_total, a_next

# Factory helper
def create_actuator(actuator_type="ideal_unlimited", device="cuda:0"):
    actuator_type = actuator_type.lower()
    if actuator_type in ["ideal_unlimited", "unlimited", "tier1"]:
        return IdealTorqueUnlimitedActuator(device=device)
    elif actuator_type in ["ideal_limited", "limited", "tier2"]:
        return IdealTorqueLimitedActuator(device=device)
    elif actuator_type in ["mtg", "muscle", "tier3"]:
        return MTGActuator(device=device)
    else:
        raise ValueError(f"Unknown actuator_type '{actuator_type}'. Choose from: 'ideal_unlimited', 'ideal_limited', 'mtg'")
