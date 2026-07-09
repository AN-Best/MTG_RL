import torch
from SymPy_Skeleton import skeleton_dynamics

# -----------------------------------------------------------------------------
# Muscle Torque Generator (MTG) & PD Controller (Pure PyTorch Version)
# -----------------------------------------------------------------------------
class MuscleTorqueGenerator:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        
        # Peak isometric torque T0 (Nm) - [Hip, Knee, Ankle]
        # Positive direction (Flexion, Extension, Dorsiflexion)
        self.T0_pos = torch.tensor([150.0, 180.0, 80.0], dtype=torch.float32, device=self.device)
        # Negative direction (Extension, Flexion, Plantar Flexion)
        self.T0_neg = torch.tensor([200.0, 120.0, 60.0], dtype=torch.float32, device=self.device)
        
        # Polynomial coefficients for Torque-Angle scaling
        # Positive direction: [p1, p2, p3]
        self.p1_pos = torch.tensor([1.0, 0.3385, 0.9403], dtype=torch.float32, device=self.device)
        self.p2_pos = torch.tensor([-0.0044, -1.066, -0.535], dtype=torch.float32, device=self.device)
        self.p3_pos = torch.tensor([-0.1492, -0.4294, -1.1989], dtype=torch.float32, device=self.device)
        
        # Negative direction: [p1, p2, p3]
        self.p1_neg = torch.tensor([0.4187, 0.5369, 0.955], dtype=torch.float32, device=self.device)
        self.p2_neg = torch.tensor([0.7945, -0.8421, -0.5544], dtype=torch.float32, device=self.device)
        self.p3_neg = torch.tensor([-0.2715, -0.3828, -1.7042], dtype=torch.float32, device=self.device)
        
        # Max velocity omega_max (rad/s)
        self.v_max_pos = torch.tensor([12.00, 28.40, 12.70], dtype=torch.float32, device=self.device)
        self.v_max_neg = torch.tensor([14.10, 18.60, 12.60], dtype=torch.float32, device=self.device)
        
        # PD Gains (Kp, Kd) for Hip, Knee, Ankle
        self.Kp = torch.tensor([250.0, 200.0, 80.0], dtype=torch.float32, device=self.device)
        self.Kd = torch.tensor([25.0, 20.0, 8.0], dtype=torch.float32, device=self.device)
        
        # Hyperbolic scaling factors
        self.Gamma = 3.0
        self.S = 2.0
        self.R = 1.4
        self.tau_act = 0.02 # activation/deactivation time constant (20 ms)

    def get_tv_scaling(self, w_dir, w_max):
        # Double-hyperbolic scaling function (Equation 8)
        # Concentric shortening
        fv_con = (w_max - w_dir) / (w_max + self.Gamma * w_dir)
        # Eccentric lengthening
        fv_ecc = ((1.0 - self.R) * w_max + self.S * w_dir * self.R * (self.Gamma + 1.0)) / \
                 ((1.0 - self.R) * w_max + self.S * w_dir * (self.Gamma + 1.0))
                 
        fv = torch.where(w_dir >= w_max, 0.0,
             torch.where(w_dir > 0, fv_con,
             torch.where(w_dir == 0, 1.0, fv_ecc)))
        return fv

    def get_limits(self, q_joints, u_joints):
        T0_pos = torch.cat([self.T0_pos, self.T0_pos], dim=-1)
        T0_neg = torch.cat([self.T0_neg, self.T0_neg], dim=-1)
        
        p1_pos = torch.cat([self.p1_pos, self.p1_pos], dim=-1)
        p2_pos = torch.cat([self.p2_pos, self.p2_pos], dim=-1)
        p3_pos = torch.cat([self.p3_pos, self.p3_pos], dim=-1)
        
        p1_neg = torch.cat([self.p1_neg, self.p1_neg], dim=-1)
        p2_neg = torch.cat([self.p2_neg, self.p2_neg], dim=-1)
        p3_neg = torch.cat([self.p3_neg, self.p3_neg], dim=-1)
        
        v_max_pos = torch.cat([self.v_max_pos, self.v_max_pos], dim=-1)
        v_max_neg = torch.cat([self.v_max_neg, self.v_max_neg], dim=-1)
        
        # Torque-Angle scaling
        f_theta_pos = torch.clamp(p1_pos + p2_pos * q_joints + p3_pos * q_joints**2, min=0.0)
        f_theta_neg = torch.clamp(p1_neg + p2_neg * q_joints + p3_neg * q_joints**2, min=0.0)
        
        # Torque-Velocity scaling
        fv_pos = self.get_tv_scaling(u_joints, v_max_pos)
        fv_neg = self.get_tv_scaling(-u_joints, v_max_neg)
        
        T_max_pos = T0_pos * f_theta_pos * fv_pos
        T_max_neg = T0_neg * f_theta_neg * fv_neg
        return T_max_pos, T_max_neg

    def get_passive_torques(self, q_joints, u_joints):
        # Anatomical limits
        hip_lim = torch.tensor([-0.5, 1.5], dtype=torch.float32, device=self.device)
        knee_lim = torch.tensor([-2.0, 0.0], dtype=torch.float32, device=self.device)
        ankle_lim = torch.tensor([-0.5, 0.5], dtype=torch.float32, device=self.device)
        
        min_lim = torch.cat([hip_lim[0:1], knee_lim[0:1], ankle_lim[0:1]] * 2, dim=-1)
        max_lim = torch.cat([hip_lim[1:2], knee_lim[1:2], ankle_lim[1:2]] * 2, dim=-1)
        
        k_stiffness = 200.0
        k_exponent = 20.0
        
        b_damping = torch.tensor([2.0, 2.0, 1.0], dtype=torch.float32, device=self.device)
        b_damping = torch.cat([b_damping, b_damping], dim=-1)
        
        passive_stiffness = k_stiffness * torch.exp(-k_exponent * (q_joints - min_lim)) - \
                             k_stiffness * torch.exp(k_exponent * (q_joints - max_lim))
        passive_damping = -b_damping * u_joints
        return passive_stiffness + passive_damping

    def compute_torques(self, q_joints, u_joints, q_ref, tau_prev, dt_sim):
        Kp = torch.cat([self.Kp, self.Kp], dim=-1)
        Kd = torch.cat([self.Kd, self.Kd], dim=-1)
        
        # Desired PD activation command
        tau_des = Kp * (q_ref - q_joints) - Kd * u_joints
        T_max_pos, T_max_neg = self.get_limits(q_joints, u_joints)
        
        # 1. Back-calculate previous activation levels a_prev
        eps = 1e-5
        a_prev = torch.where(tau_prev >= 0,
                             tau_prev / (T_max_pos + eps),
                             tau_prev / (T_max_neg + eps))
        a_prev = torch.clamp(a_prev, -1.0, 1.0)
        
        # 2. Propagate activation bounds over dt_sim using first-order dynamics
        a_max = a_prev + ((1.0 - a_prev) / self.tau_act) * dt_sim
        a_min = a_prev + ((-1.0 - a_prev) / self.tau_act) * dt_sim
        
        a_max = torch.clamp(a_max, -1.0, 1.0)
        a_min = torch.clamp(a_min, -1.0, 1.0)
        
        # 3. Calculate corresponding torque bounds
        tau_max = torch.where(a_max >= 0, a_max * T_max_pos, a_max * T_max_neg)
        tau_min = torch.where(a_min >= 0, a_min * T_max_pos, a_min * T_max_neg)
        
        # 4. Apply dynamic clamp to active torques
        tau_active = torch.max(torch.min(tau_des, tau_max), tau_min)
        tau_passive = self.get_passive_torques(q_joints, u_joints)
        
        return tau_active, tau_active + tau_passive

def rk4_step(q, u, torques_func, params, dt):
    def eval_dynamics(q_curr, u_curr):
        tau_active, tau_j = torques_func(q_curr, u_curr)
        dq, du = skeleton_dynamics(q_curr, u_curr, tau_j, params)
        return dq, du, tau_active

    dq1, du1, ta1 = eval_dynamics(q, u)
    
    q2 = q + 0.5 * dt * dq1
    u2 = u + 0.5 * dt * du1
    dq2, du2, ta2 = eval_dynamics(q2, u2)
    
    q3 = q + 0.5 * dt * dq2
    u3 = u + 0.5 * dt * du2
    dq3, du3, ta3 = eval_dynamics(q3, u3)
    
    q4 = q + dt * dq3
    u4 = u + dt * du3
    dq4, du4, ta4 = eval_dynamics(q4, u4)
    
    q_next = q + (dt / 6.0) * (dq1 + 2.0 * dq2 + 2.0 * dq3 + dq4)
    u_next = u + (dt / 6.0) * (du1 + 2.0 * du2 + 2.0 * du3 + du4)
    tau_active_next = (ta1 + 2.0 * ta2 + 2.0 * ta3 + ta4) / 6.0
    return q_next, u_next, tau_active_next

def step_environment(q, u, q_ref, params, dt_sim=0.002, dt_control=0.1, mtg_controller=None, tau_prev=None, device="cpu"):
    """Integrates biped skeleton dynamics over control timestep using sub-stepping.
    All inputs and outputs are PyTorch Tensors.
    """
    device = torch.device(device)
    if mtg_controller is None:
        mtg_controller = MuscleTorqueGenerator(device=device)
        
    if tau_prev is None:
        if q.ndim > 1:
            tau_prev = torch.zeros((q.shape[0], 6), dtype=torch.float32, device=device)
        else:
            tau_prev = torch.zeros(6, dtype=torch.float32, device=device)
        
    num_steps = int(round(dt_control / dt_sim))
    q_curr = q.to(device=device, dtype=torch.float32)
    u_curr = u.to(device=device, dtype=torch.float32)
    tau_curr = tau_prev.to(device=device, dtype=torch.float32)
    q_ref = q_ref.to(device=device, dtype=torch.float32)
    params = params.to(device=device, dtype=torch.float32)
    
    for _ in range(num_steps):
        def get_torques(q_t, u_t):
            q_j = q_t[..., 3:9]
            u_j = u_t[..., 3:9]
            return mtg_controller.compute_torques(q_j, u_j, q_ref, tau_curr, dt_sim)
            
        q_curr, u_curr, tau_curr = rk4_step(q_curr, u_curr, get_torques, params, dt_sim)
        
    return q_curr, u_curr, tau_curr

if __name__ == '__main__':
    # Simple test of the environment step
    print("Testing Biped Simulation step on GPU...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    q = torch.zeros(9, dtype=torch.float32, device=device)
    q[1] = 0.90 # nominal hip height
    u = torch.zeros(9, dtype=torch.float32, device=device)
    q_ref = torch.zeros(6, dtype=torch.float32, device=device)
    
    params = torch.tensor([
        70.0, 10.0, 5.0, 1.5,   # Masses
        0.8, 0.45, 0.45, 0.25, 0.05, 0.2, # Lengths
        0.4, 0.2, 0.2, 0.08,    # COMs
        2.5, 0.1, 0.05, 0.01,   # Inertias
        9.81, 0.95,             # g, h_g
        5e6, 0.8, 0.6, 0.01     # kc, cc, mu, vs
    ], dtype=torch.float32, device=device)
    
    q_next, u_next, _ = step_environment(q, u, q_ref, params, dt_sim=0.002, dt_control=0.1, device=device)
    
    print("Initial q:", q)
    print("Next q after 0.1s step:", q_next)
    print("Initial u:", u)
    print("Next u after 0.1s step:", u_next)
    print("Simulation step test successful!")
