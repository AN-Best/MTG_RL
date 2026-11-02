"""
Biped Actuators: Mode 1 (Ideal Unlimited PD) & Mode 2 (Hard-Clamped MTG)
========================================================================
- Mode 1 ('ideal_unlimited'): Pure linear PD torque control with zero MTG limits.
- Mode 2 ('ideal_limited'): Instantaneous Hill-type Muscle Torque Generator (MTG)
  hard-clamped at every 0.002s physics sub-step to dynamic biological capacity envelopes:
  tau = clamp(tau_pd, -T_avail^-(q, qdot), +T_avail^+(q, qdot))
"""

import torch
import torch.nn as nn
import numpy as np

class BaseActuator(nn.Module):
    def __init__(self, device="cuda:0"):
        super().__init__()
        self.device = torch.device(device)
        self.actuator_type = "base"

    def compute_torques(self, q_j, u_j, q_des, state=None, dt_sim=0.002, u_des=None):
        raise NotImplementedError

# -----------------------------------------------------------------------------
# Mode 1: Ideal Unlimited PD (Zero mention or computation of MTG)
# -----------------------------------------------------------------------------
class IdealUnlimitedActuator(BaseActuator):
    """
    Mode 1: Pure Ideal PD Joint Controller.
    Computes direct joint torques from tracking errors without any MTG limits:
      tau = Kp * (q_des - q) + Kd * (u_des - u)
    """
    def __init__(self, device="cuda:0"):
        super().__init__(device=device)
        # Sized to human joint compliance [Lumbar, Hip_L, Knee_L, Ankle_L, Hip_R, Knee_R, Ankle_R]
        self.register_buffer("Kp", torch.tensor([400.0, 450.0, 350.0, 300.0, 450.0, 350.0, 300.0], dtype=torch.float32, device=self.device))
        self.register_buffer("Kd", torch.tensor([30.0, 35.0, 25.0, 20.0, 35.0, 25.0, 20.0], dtype=torch.float32, device=self.device))
        self.Kp_np = np.array([400.0, 450.0, 350.0, 300.0, 450.0, 350.0, 300.0], dtype=np.float32)
        self.Kd_np = np.array([30.0, 35.0, 25.0, 20.0, 35.0, 25.0, 20.0], dtype=np.float32)
        self.actuator_type = "ideal_unlimited"

    def compute_torques_np(self, q_j, u_j, q_des, u_des=None):
        if u_des is not None:
            tau = self.Kp_np * (q_des - q_j) + self.Kd_np * (u_des - u_j)
        else:
            tau = self.Kp_np * (q_des - q_j) - self.Kd_np * u_j
        return np.clip(tau, -2000.0, 2000.0)

    def compute_torques(self, q_j, u_j, q_des, state=None, dt_sim=0.002, u_des=None):
        if u_des is not None:
            tau = self.Kp * (q_des - q_j) + self.Kd * (u_des - u_j)
        else:
            tau = self.Kp * (q_des - q_j) - self.Kd * u_j
        tau_safe = torch.clamp(tau, -2000.0, 2000.0)
        return tau_safe, None

# -----------------------------------------------------------------------------
# Mode 2: MTG Clamped Actuator (Activation + Force-Length + Force-Velocity + Passive)
# -----------------------------------------------------------------------------
class MTGClampedActuator(BaseActuator):
    """
    Mode 2: MTG Clamped Actuator with 1st-Order Activation Dynamics & Passive Joint Torque.
    1. Computes passive joint tissue torque:
         tau_passive(q) = k1 * relu(q1 - q)^2 - k2 * relu(q - q2)^2
    2. Compensates passive torque with desired active muscle torque:
         tau_active_des = tau_pd - tau_passive(q)
    3. Evolves 1st-order muscle activation dynamics a(t) (tau_act=20ms, tau_deact=40ms)
       driven by |tau_active_des| / T_LV(q, qdot).
    4. Clamps active muscle torque at every 0.002s sub-step to active biological capacity:
         T_avail_act^+(a, q, qdot) = a(t) * T_max^+ * f_L(q) * f_V^+(qdot)
         T_avail_act^-(a, q, qdot) = a(t) * T_max^- * f_L(q) * f_V^-(qdot)
         tau_active = clamp(tau_active_des, -T_avail_act^-, +T_avail_act^+)
    5. Applies total joint torque to the skeleton:
         tau_total = tau_active + tau_passive(q)
    """
    def __init__(self, device="cuda:0"):
        super().__init__(device=device)
        self.register_buffer("Kp", torch.tensor([400.0, 450.0, 350.0, 300.0, 450.0, 350.0, 300.0], dtype=torch.float32, device=self.device))
        self.register_buffer("Kd", torch.tensor([30.0, 35.0, 25.0, 20.0, 35.0, 25.0, 20.0], dtype=torch.float32, device=self.device))

        # Peak isometric torque capacities (N*m) [Lumbar, Hip_L, Knee_L, Ankle_L, Hip_R, Knee_R, Ankle_R]
        self.register_buffer("T_max_pos", torch.tensor([300.0, 400.0, 500.0, 100.0, 400.0, 500.0, 100.0], dtype=torch.float32, device=self.device))
        self.register_buffer("T_max_neg", torch.tensor([400.0, 300.0, 200.0, 380.0, 300.0, 200.0, 380.0], dtype=torch.float32, device=self.device))
        self.register_buffer("q_opt_pos", torch.tensor([0.0, -0.40, -0.90, 0.35, -0.40, -0.90, 0.35], dtype=torch.float32, device=self.device))
        self.register_buffer("q_opt_neg", torch.tensor([0.0, -0.40, -0.90, 0.35, -0.40, -0.90, 0.35], dtype=torch.float32, device=self.device))
        self.register_buffer("w_pos", torch.tensor([1.20, 1.20, 1.20, 1.10, 1.20, 1.20, 1.10], dtype=torch.float32, device=self.device))
        self.register_buffer("w_neg", torch.tensor([1.20, 1.20, 1.20, 1.10, 1.20, 1.20, 1.10], dtype=torch.float32, device=self.device))
        self.register_buffer("u_max_pos", torch.tensor([12.0, 15.0, 18.0, 16.0, 15.0, 18.0, 16.0], dtype=torch.float32, device=self.device))
        self.register_buffer("u_max_neg", torch.tensor([12.0, 15.0, 18.0, 22.0, 15.0, 18.0, 22.0], dtype=torch.float32, device=self.device))
        
        # Anatomical Joint limits [Lumbar, Hip_L, Knee_L, Ankle_L, Hip_R, Knee_R, Ankle_R] (rad)
        self.register_buffer("limits_low", torch.tensor([-0.469, -1.5, -2.0, -1.2, -1.5, -2.0, -1.2], dtype=torch.float32, device=self.device))
        self.register_buffer("limits_high", torch.tensor([0.620, 0.8, 0.0, 1.2, 0.8, 0.0, 1.2], dtype=torch.float32, device=self.device))

        # Passive joint stiffness parameters [k1, k2, theta_1, theta_2]
        self.register_buffer("k1_pass", torch.tensor([250.0, 200.0, 300.0, 150.0, 200.0, 300.0, 150.0], dtype=torch.float32, device=self.device))
        self.register_buffer("k2_pass", torch.tensor([250.0, 200.0, 300.0, 150.0, 200.0, 300.0, 150.0], dtype=torch.float32, device=self.device))
        self.register_buffer("q1_pass", torch.tensor([-0.40, -0.40, -1.80, -0.40, -0.40, -1.80, -0.40], dtype=torch.float32, device=self.device))
        self.register_buffer("q2_pass", torch.tensor([0.50, 1.20, 0.05, 0.40, 1.20, 0.05, 0.40], dtype=torch.float32, device=self.device))

        # 1st-Order Muscle Activation Dynamics Parameters (He et al. 1991 / Millard et al. 2013)
        self.tau_act = 0.020    # 20 ms activation time constant
        self.tau_deact = 0.040  # 40 ms deactivation time constant
        self.a_min = 0.01       # Minimum baseline physiological muscle tone
        self.a_init = 0.35      # Initial activation state at episode reset

        self.Kp_np = np.array([400.0, 450.0, 350.0, 300.0, 450.0, 350.0, 300.0], dtype=np.float32)
        self.Kd_np = np.array([30.0, 35.0, 25.0, 20.0, 35.0, 25.0, 20.0], dtype=np.float32)
        self.T_max_pos_np = np.array([300.0, 400.0, 500.0, 100.0, 400.0, 500.0, 100.0], dtype=np.float32)
        self.T_max_neg_np = np.array([400.0, 300.0, 200.0, 380.0, 300.0, 200.0, 380.0], dtype=np.float32)
        self.q_opt_np = np.array([0.0, -0.40, -0.90, 0.35, -0.40, -0.90, 0.35], dtype=np.float32)
        self.w_np = np.array([1.20, 1.20, 1.20, 1.10, 1.20, 1.20, 1.10], dtype=np.float32)
        self.u_max_pos_np = np.array([12.0, 15.0, 18.0, 16.0, 15.0, 18.0, 16.0], dtype=np.float32)
        self.u_max_neg_np = np.array([12.0, 15.0, 18.0, 22.0, 15.0, 18.0, 22.0], dtype=np.float32)
        self.k1_pass_np = np.array([250.0, 200.0, 300.0, 150.0, 200.0, 300.0, 150.0], dtype=np.float32)
        self.k2_pass_np = np.array([250.0, 200.0, 300.0, 150.0, 200.0, 300.0, 150.0], dtype=np.float32)
        self.q1_pass_np = np.array([-0.40, -0.40, -1.80, -0.40, -0.40, -1.80, -0.40], dtype=np.float32)
        self.q2_pass_np = np.array([0.50, 1.20, 0.05, 0.40, 1.20, 0.05, 0.40], dtype=np.float32)

        self.a_curr_np = np.full(7, self.a_init, dtype=np.float32)
        self.last_a = self.a_curr_np.copy()
        self.last_T_LV_pos = None
        self.last_T_LV_neg = None
        self.last_T_act_pos = None
        self.last_T_act_neg = None
        self.last_T_pos = None
        self.last_T_neg = None
        self.last_tau_active = None
        self.last_tau_passive = None
        self.actuator_type = "mtg_clamped"

    def reset_state(self):
        """Resets muscle activation state a(t) for a new episode."""
        self.a_curr_np = np.full(7, self.a_init, dtype=np.float32)
        self.last_a = self.a_curr_np.copy()
        self.last_T_LV_pos = None
        self.last_T_LV_neg = None
        self.last_T_act_pos = None
        self.last_T_act_neg = None
        self.last_T_pos = None
        self.last_T_neg = None
        self.last_tau_active = None
        self.last_tau_passive = None

    def compute_passive_np(self, q_j):
        """Computes non-linear passive joint restoring torque tau_passive(q)."""
        return (
            self.k1_pass_np * np.maximum(0.0, self.q1_pass_np - q_j) ** 2
            - self.k2_pass_np * np.maximum(0.0, q_j - self.q2_pass_np) ** 2
        )

    def compute_capacities_np(self, q_j, u_j):
        """Computes full-activation (a=1) Force-Length-Velocity torque capacity T_LV(q, qdot)."""
        f_L = np.exp(-((q_j - self.q_opt_np) / self.w_np) ** 2)
        nu_pos = np.clip(u_j / self.u_max_pos_np, -1.0, 1.0)
        nu_neg = np.clip(-u_j / self.u_max_neg_np, -1.0, 1.0)
        f_V_pos = np.where(
            nu_pos >= 0.0,
            np.clip((1.0 - nu_pos) / (1.0 + 4.0 * nu_pos + 1e-6), 0.0, 1.0),
            np.clip(1.0 - 1.4 * nu_pos / (1.0 - 4.0 * nu_pos + 1e-6), 1.0, 1.6)
        )
        f_V_neg = np.where(
            nu_neg >= 0.0,
            np.clip((1.0 - nu_neg) / (1.0 + 4.0 * nu_neg + 1e-6), 0.0, 1.0),
            np.clip(1.0 - 1.4 * nu_neg / (1.0 - 4.0 * nu_neg + 1e-6), 1.0, 1.6)
        )
        return self.T_max_pos_np * f_L * f_V_pos, self.T_max_neg_np * f_L * f_V_neg

    def step_activation_np(self, tau_active_des, T_LV_pos, T_LV_neg, dt_sim=0.002):
        """Advances 1st-order muscle activation dynamics a(t) by dt_sim and returns a(t)."""
        T_LV_dir = np.where(tau_active_des >= 0.0, T_LV_pos, T_LV_neg)
        u_req = np.clip(np.abs(tau_active_des) / (T_LV_dir + 1e-3), 0.0, 1.0)
        tau_c = np.where(u_req >= self.a_curr_np, self.tau_act, self.tau_deact)
        da_dt = (u_req - self.a_curr_np) / tau_c
        self.a_curr_np = np.clip(self.a_curr_np + dt_sim * da_dt, self.a_min, 1.0)
        self.last_a = self.a_curr_np.copy()
        return self.a_curr_np

    def compute_torques_np(self, q_j, u_j, q_des, u_des=None, dt_sim=0.002):
        if u_des is not None:
            tau_pd = self.Kp_np * (q_des - q_j) + self.Kd_np * (u_des - u_j)
        else:
            tau_pd = self.Kp_np * (q_des - q_j) - self.Kd_np * u_j

        # 1. Passive joint torque
        tau_passive = self.compute_passive_np(q_j)

        # 2. Compensate passive torque with active torque demand
        tau_active_des = tau_pd - tau_passive

        # 3. Kinematic f_L * f_V capacities and 1st-order activation dynamics a(t)
        T_LV_pos, T_LV_neg = self.compute_capacities_np(q_j, u_j)
        a_curr = self.step_activation_np(tau_active_des, T_LV_pos, T_LV_neg, dt_sim=dt_sim)

        # 4. Active torque clamp: tau_active in [-a * T_LV_neg, +a * T_LV_pos]
        T_act_pos = a_curr * T_LV_pos
        T_act_neg = a_curr * T_LV_neg
        tau_active = np.clip(tau_active_des, -T_act_neg, T_act_pos)

        # 5. Total joint torque applied to skeleton: tau_total = tau_active + tau_passive
        tau_total = tau_active + tau_passive

        self.last_T_LV_pos = T_LV_pos
        self.last_T_LV_neg = T_LV_neg
        self.last_T_act_pos = T_act_pos
        self.last_T_act_neg = T_act_neg
        self.last_tau_active = tau_active
        self.last_tau_passive = tau_passive
        # Total torque bounds [ -T_act_neg + tau_passive, +T_act_pos + tau_passive ]
        self.last_T_pos = T_act_pos + tau_passive
        self.last_T_neg = T_act_neg - tau_passive
        return tau_total

    def compute_capacities(self, q_j, u_j):
        """Computes dynamic positive and negative Force-Length-Velocity MTG capacities."""
        f_L_pos = torch.exp(-((q_j - self.q_opt_pos) / self.w_pos) ** 2)
        f_L_neg = torch.exp(-((q_j - self.q_opt_neg) / self.w_neg) ** 2)

        norm_u_pos = torch.clamp(u_j / self.u_max_pos, -1.0, 1.0)
        norm_u_neg = torch.clamp(-u_j / self.u_max_neg, -1.0, 1.0)

        f_V_pos = torch.where(
            norm_u_pos >= 0.0,
            torch.clamp((1.0 - norm_u_pos) / (1.0 + 4.0 * norm_u_pos + 1e-6), 0.0, 1.0),
            torch.clamp(1.0 - 1.4 * norm_u_pos / (1.0 - 4.0 * norm_u_pos + 1e-6), 1.0, 1.6)
        )
        f_V_neg = torch.where(
            norm_u_neg >= 0.0,
            torch.clamp((1.0 - norm_u_neg) / (1.0 + 4.0 * norm_u_neg + 1e-6), 0.0, 1.0),
            torch.clamp(1.0 - 1.4 * norm_u_neg / (1.0 - 4.0 * norm_u_neg + 1e-6), 1.0, 1.6)
        )

        T_LV_pos = self.T_max_pos * f_L_pos * f_V_pos
        T_LV_neg = self.T_max_neg * f_L_neg * f_V_neg
        return T_LV_pos, T_LV_neg

    def compute_torques(self, q_j, u_j, q_des, state=None, dt_sim=0.002, u_des=None):
        if u_des is not None:
            tau_pd = self.Kp * (q_des - q_j) + self.Kd * (u_des - u_j)
        else:
            tau_pd = self.Kp * (q_des - q_j) - self.Kd * u_j

        # 1. Passive joint torque
        tau_passive = (
            self.k1_pass * torch.relu(self.q1_pass - q_j) ** 2
            - self.k2_pass * torch.relu(q_j - self.q2_pass) ** 2
        )

        # 2. Compensate passive torque with active torque demand
        tau_active_des = tau_pd - tau_passive

        # 3. Kinematic f_L * f_V capacities and 1st-order activation dynamics a(t)
        T_LV_pos, T_LV_neg = self.compute_capacities(q_j, u_j)

        if state is None:
            a_curr = torch.full_like(q_j, self.a_init)
        else:
            a_curr = state

        T_LV_dir = torch.where(tau_active_des >= 0.0, T_LV_pos, T_LV_neg)
        u_req = torch.clamp(torch.abs(tau_active_des) / (T_LV_dir + 1e-3), 0.0, 1.0)
        tau_c = torch.where(u_req >= a_curr, self.tau_act, self.tau_deact)
        da_dt = (u_req - a_curr) / tau_c
        a_next = torch.clamp(a_curr + dt_sim * da_dt, self.a_min, 1.0)

        T_act_pos = a_next * T_LV_pos
        T_act_neg = a_next * T_LV_neg

        # 4. Physical MTG Hard-Clamping on Active Torque + Passive Contribution
        tau_active = torch.clamp(tau_active_des, min=-T_act_neg, max=T_act_pos)
        tau_total = tau_active + tau_passive
        return tau_total, a_next

# -----------------------------------------------------------------------------
# Actuator Factory
# -----------------------------------------------------------------------------
def create_actuator(actuator_type="ideal_unlimited", device="cuda:0"):
    actuator_type = str(actuator_type).lower()
    if actuator_type in ["ideal_unlimited", "unlimited", "mode1"]:
        return IdealUnlimitedActuator(device=device)
    elif actuator_type in ["mtg_clamped", "mtg", "clamped", "mode2"]:
        return MTGClampedActuator(device=device)
    else:
        raise ValueError(f"Unknown actuator type '{actuator_type}'. Choose 'ideal_unlimited' or 'mtg_clamped'.")
