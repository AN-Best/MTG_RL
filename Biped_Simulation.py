import torch
from SymPy_Skeleton import skeleton_dynamics

# -----------------------------------------------------------------------------
# Muscle Torque Generator (MTG) & PD Controller (Pure PyTorch Version)
# -----------------------------------------------------------------------------
class MuscleTorqueGenerator:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        
        # Peak isometric torque T0 (Nm) - [Hip, Knee, Ankle] (Exact Nasr & McPhee paper values)
        self.T0_pos = torch.tensor([150.0, 180.0, 80.0], dtype=torch.float32, device=self.device)
        # Negative direction (Hip Extension, Knee Flexion, Ankle Plantar Flexion)
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
        
        # PD Gains (Kp, Kd) for Hip, Knee, Ankle (Tuned for precise tracking and foot clearance)
        self.Kp = torch.tensor([400.0, 400.0, 300.0], dtype=torch.float32, device=self.device)
        self.Kd = torch.tensor([40.0, 35.0, 25.0], dtype=torch.float32, device=self.device)
        
        # Hyperbolic scaling factors
        self.Gamma = 3.0
        self.S = 2.0
        self.R = 1.4
        self.tau_act = 0.02 # activation/deactivation time constant (20 ms)

        # Pre-allocated joint limits & passive damping tensors
        hip_lim = torch.tensor([-0.5, 1.5], dtype=torch.float32, device=self.device)
        knee_lim = torch.tensor([-2.0, 0.0], dtype=torch.float32, device=self.device)
        ankle_lim = torch.tensor([-0.5, 0.5], dtype=torch.float32, device=self.device)
        
        self.min_lim = torch.cat([hip_lim[0:1], knee_lim[0:1], ankle_lim[0:1]] * 2, dim=-1)
        self.max_lim = torch.cat([hip_lim[1:2], knee_lim[1:2], ankle_lim[1:2]] * 2, dim=-1)
        
        b_d = torch.tensor([2.0, 2.0, 1.0], dtype=torch.float32, device=self.device)
        self.b_damping = torch.cat([b_d, b_d], dim=-1)
        
        self.Kp_batched = torch.cat([self.Kp, self.Kp], dim=-1)
        self.Kd_batched = torch.cat([self.Kd, self.Kd], dim=-1)

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
        # Quadratic spring stiffness to prevent Inf/NaN explosions
        k_stiffness = 500.0
        k_quadratic = 15.0
        
        # Calculate penetration beyond limits using pre-allocated tensors
        delta_min = torch.clamp(self.min_lim - q_joints, min=0.0)
        delta_max = torch.clamp(q_joints - self.max_lim, min=0.0)
        
        passive_stiffness = k_stiffness * (delta_min + k_quadratic * delta_min**2) - \
                            k_stiffness * (delta_max + k_quadratic * delta_max**2)
        passive_damping = -self.b_damping * u_joints
        return passive_stiffness + passive_damping

    def compute_torques(self, q_joints, u_joints, q_ref, tau_prev, dt_sim):
        # Desired PD activation command
        tau_des = self.Kp_batched * (q_ref - q_joints) - self.Kd_batched * u_joints
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

def symplectic_euler_step(q, u, torques_func, params, dt):
    tau_active, tau_j = torques_func(q, u)
    dq, du = skeleton_dynamics(q, u, tau_j, params)
    
    # Symplectic update: velocity first, then position using new velocity
    u_next = u + dt * du
    q_next = q + dt * u_next
    
    return q_next, u_next, tau_active

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
    
    # Track failed envs
    if q.ndim > 1:
        nan_occurred = torch.zeros(q.shape[0], dtype=torch.bool, device=device)
    else:
        nan_occurred = False
        
    for _ in range(num_steps):
        if q_curr.ndim > 1:
            step_nan = torch.isnan(q_curr).any(dim=-1) | torch.isinf(q_curr).any(dim=-1) | \
                       torch.isnan(u_curr).any(dim=-1) | torch.isinf(u_curr).any(dim=-1)
            nan_occurred |= step_nan
            if step_nan.any():
                q_curr[step_nan] = 0.0
                q_curr[step_nan, 1] = 0.90
                u_curr[step_nan] = 0.0
                tau_curr[step_nan] = 0.0
        else:
            step_nan = torch.isnan(q_curr).any() or torch.isinf(q_curr).any() or \
                       torch.isnan(u_curr).any() or torch.isinf(u_curr).any()
            if step_nan:
                nan_occurred = True
                q_curr = torch.zeros_like(q_curr)
                q_curr[1] = 0.90
                u_curr = torch.zeros_like(u_curr)
                tau_curr = torch.zeros_like(tau_curr)

        def get_torques(q_t, u_t):
            q_j = q_t[..., 3:9]
            u_j = u_t[..., 3:9]
            return mtg_controller.compute_torques(q_j, u_j, q_ref, tau_curr, dt_sim)
            
        q_curr, u_curr, tau_curr = symplectic_euler_step(q_curr, u_curr, get_torques, params, dt_sim)
        
    # Final check and recovery after the loop to clean any NaNs/Infs from the last substep
    if q_curr.ndim > 1:
        step_nan = torch.isnan(q_curr).any(dim=-1) | torch.isinf(q_curr).any(dim=-1) | \
                   torch.isnan(u_curr).any(dim=-1) | torch.isinf(u_curr).any(dim=-1)
        nan_occurred |= step_nan
        if step_nan.any():
            q_curr[step_nan] = 0.0
            q_curr[step_nan, 1] = 0.90
            u_curr[step_nan] = 0.0
            tau_curr[step_nan] = 0.0
    else:
        step_nan = torch.isnan(q_curr).any() or torch.isinf(q_curr).any() or \
                   torch.isnan(u_curr).any() or torch.isinf(u_curr).any()
        if step_nan:
            nan_occurred = True
            q_curr = torch.zeros_like(q_curr)
            q_curr[1] = 0.90
            u_curr = torch.zeros_like(u_curr)
            tau_curr = torch.zeros_like(tau_curr)
            
    return q_curr, u_curr, tau_curr, nan_occurred

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
