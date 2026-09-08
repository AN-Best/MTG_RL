import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from Biped_Actuators import create_actuator
from Biped_Dynamics import get_default_body_parameters, simulate_macro_step

class BipedAMPEnv(gym.Env):
    """
    Vectorized Gymnasium environment for the 10-DoF Pelvis + Lumbar biped skeleton.
    The action space is ALWAYS target pose adjustments delta_q (7 actuated joints).
    Internal PD controllers inside each actuator convert target pose -> joint torques.
    
    Supported Actuator Tiers:
      - 'ideal_unlimited': Unconstrained PD torque actuation
      - 'ideal_limited': Peak isometric torque-saturated actuation
      - 'mtg': Full 1st-order muscle activation dynamics & active/passive torque curves
    """
    metadata = {"render_modes": []}

    def __init__(self, num_envs=1, actuator_type="ideal_unlimited", device="cuda:0", random_start=True):
        super(BipedAMPEnv, self).__init__()
        
        self.num_envs = num_envs
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.random_start = random_start
        self.actuator_type = actuator_type.lower()
        
        # 7 Joint control actions (Lumbar, Hip_L, Knee_L, Ankle_L, Hip_R, Knee_R, Ankle_R)
        self.num_actions = 7
        # 19 observed variables: 9 coordinates (q excluding q_x) and 10 velocities (u)
        self.num_observations = 19
        
        # Actions are target pose offsets in [-0.3, 0.3] rad
        self.action_space = spaces.Box(
            low=np.array([-0.3] * self.num_actions, dtype=np.float32),
            high=np.array([0.3] * self.num_actions, dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_observations,),
            dtype=np.float32
        )
        
        # Initialize Actuator & Body Parameters
        self.actuator = create_actuator(self.actuator_type, device=self.device)
        self.params = get_default_body_parameters(batch_size=self.num_envs, device=self.device)
        
        self.q = torch.zeros((self.num_envs, 10), dtype=torch.float32, device=self.device)
        self.u = torch.zeros((self.num_envs, 10), dtype=torch.float32, device=self.device)
        self.actuator_state = torch.full((self.num_envs, 7), 0.01, device=self.device) if self.actuator_type == "mtg" else None
        
        self.max_episode_steps = 500 # 10 seconds at 50 Hz control rate
        self.episode_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def set_reference_motion(self, q_ref, u_ref):
        self.q_ref = q_ref.to(device=self.device, dtype=torch.float32)
        self.u_ref = u_ref.to(device=self.device, dtype=torch.float32)
        if self.q_ref.ndim == 1:
            self.q_ref = self.q_ref.unsqueeze(0)
        if self.u_ref.ndim == 1:
            self.u_ref = self.u_ref.unsqueeze(0)

    def set_random_start(self, enabled: bool):
        self.random_start = enabled

    def reset(self, seed=None, options=None):
        is_random_start = self.random_start
        if options is not None and "random_start" in options:
            is_random_start = options["random_start"]
            
        if hasattr(self, 'q_ref') and self.q_ref is not None:
            num_samples = self.q_ref.shape[0]
            if is_random_start:
                start_indices = torch.randint(0, num_samples, (self.num_envs,), device=self.device)
                self.episode_steps = start_indices.clone()
                self.q = self.q_ref[start_indices].clone()
                self.u = self.u_ref[start_indices].clone()
                self.q += (torch.rand_like(self.q) - 0.5) * 0.005
                self.u += (torch.rand_like(self.u) - 0.5) * 0.005
            else:
                self.episode_steps.fill_(0)
                self.q = self.q_ref[0:1].repeat(self.num_envs, 1).clone()
                self.u = self.u_ref[0:1].repeat(self.num_envs, 1).clone()
        else:
            self.episode_steps.fill_(0)
            self.q.fill_(0.0)
            self.q[:, 1] = 0.90
            self.u.fill_(0.0)
            
        self.q[:, 0] = 0.0
        if self.actuator_type == "mtg":
            self.actuator_state = torch.full((self.num_envs, 7), 0.01, device=self.device)
            
        obs_tensor = self._get_obs()
        return obs_tensor, {"amp_obs": obs_tensor}

    def _get_obs(self):
        obs = torch.cat([self.q[:, 1:], self.u], dim=-1)
        return torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    def step(self, actions):
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        else:
            actions = actions.to(device=self.device, dtype=torch.float32)
            
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)
            
        # Target pose offset delta_q in [-0.3, 0.3] rad
        delta_q = torch.clamp(actions, min=-0.3, max=0.3)
        
        if hasattr(self, 'q_ref') and self.q_ref is not None:
            phase_idx = self.episode_steps % self.q_ref.shape[0]
            q_target = self.q_ref[phase_idx, 3:].clone()
        else:
            q_target = torch.zeros((self.num_envs, 7), device=self.device)
            
        # Unified Target Pose
        q_des = q_target + delta_q
        
        # Differentiable macro-step simulation (Actuator internally converts q_des -> torques)
        self.q, self.u, self.actuator_state, tau_applied = simulate_macro_step(
            self.q, self.u, self.actuator_state, q_des, self.actuator, self.params,
            dt_sim=0.002, dt_control=0.02
        )
        self.episode_steps += 1
        
        # Multi-objective locomotion reward
        pelvis_angle = self.q[:, 2]
        lumbar_angle = self.q[:, 3]
        torso_angle_world = pelvis_angle + lumbar_angle
        hip_height = self.q[:, 1]
        vel_x = self.u[:, 0]
        
        if hasattr(self, 'q_ref') and self.q_ref is not None:
            phase_idx = (self.episode_steps - 1) % self.q_ref.shape[0]
            q_ref_target = self.q_ref[phase_idx, 3:]
            u_ref_target = self.u_ref[phase_idx, 3:]
            torso_angle_ref = self.q_ref[phase_idx, 2] + self.q_ref[phase_idx, 3]
            
            r_pose = torch.exp(-5.0 * torch.mean((self.q[:, 3:10] - q_ref_target) ** 2, dim=-1))
            r_vel = torch.exp(-0.1 * torch.mean((self.u[:, 3:10] - u_ref_target) ** 2, dim=-1))
        else:
            torso_angle_ref = torch.zeros_like(pelvis_angle)
            r_pose = torch.ones(self.num_envs, device=self.device)
            r_vel = torch.ones(self.num_envs, device=self.device)
            
        r_upright = torch.exp(-15.0 * ((torso_angle_world - torso_angle_ref) ** 2))
        r_speed = torch.exp(-2.0 * ((vel_x - 1.2) ** 2))
        r_energy = torch.exp(-0.01 * torch.mean((tau_applied / 300.0) ** 2, dim=-1))
        
        reward = 0.35 * r_pose + 0.15 * r_vel + 0.25 * r_upright + 0.15 * r_speed + 0.10 * r_energy
        
        # Termination conditions
        terminated = (hip_height < 0.65) | (torch.abs(torso_angle_world - torso_angle_ref) > 0.80) | \
                     (vel_x < -0.50) | (vel_x > 5.0) | torch.isnan(self.q).any(dim=-1)
        truncated = self.episode_steps >= self.max_episode_steps
        
        obs_tensor = self._get_obs()
        return obs_tensor, reward, terminated, truncated, {"amp_obs": obs_tensor}
