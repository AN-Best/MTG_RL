import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from Biped_Simulation import step_environment, MuscleTorqueGenerator

class BipedAMPEnv(gym.Env):
    """A vectorized Gymnasium environment for the 2D biped skeleton,
    running entirely in PyTorch on the GPU/CPU for maximum acceleration.
    """
    metadata = {"render_modes": []}

    def __init__(self, num_envs=1, device="cpu"):
        super(BipedAMPEnv, self).__init__()
        
        self.num_envs = num_envs
        self.device = torch.device(device)
        
        # 7 actions: 6 joint angle residuals + 1 pelvis height residual (dq_y)
        self.num_actions = 7
        # 17 observed variables: 8 positions (q excluding q_x) and 9 velocities (u)
        self.num_observations = 17
        
        # Action space in [-1, 1]: Actions 0-5 = joint residuals (+/- 0.15 rad), Action 6 = pelvis height (+/- 0.05 m)
        self.action_limits_low = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)
        self.action_limits_high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        
        self.action_space = spaces.Box(
            low=self.action_limits_low,
            high=self.action_limits_high,
            dtype=np.float32
        )
        
        # Action limits as PyTorch tensors
        self.action_limits_low_tensor = torch.tensor(self.action_limits_low, dtype=torch.float32, device=self.device)
        self.action_limits_high_tensor = torch.tensor(self.action_limits_high, dtype=torch.float32, device=self.device)
        
        # Observation space: 8 positions and 9 velocities
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_observations,),
            dtype=np.float32
        )
        
        # Initialize the MTG controller on device
        self.mtg_controller = MuscleTorqueGenerator(device=self.device)
        
        # Set up default parameters for skeleton_dynamics (order matches SymPy_Skeleton):
        single_params = torch.tensor([
            45.0, 10.0, 5.0, 1.5,   # Masses (HAT mass reduced to 45kg to match standard subject)
            0.8, 0.45, 0.45, 0.25, 0.05, 0.2, # Lengths
            0.4, 0.2, 0.2, 0.08,    # COMs
            1.6, 0.1, 0.05, 0.01,   # Inertias (Torso inertia scaled to 1.6 kg*m^2)
            9.81, 0.90,             # g, h_g
            1e6, 0.60, 0.8, 0.04    # kc, cc, mu, vs (cc=0.60 realistic Hunt-Crossley damping, vs=0.04 smooth stiction)
        ], dtype=torch.float32, device=self.device)
        
        # Batch parameters if running multiple environments
        if self.num_envs > 1:
            self.params = single_params.unsqueeze(0).repeat(self.num_envs, 1)
        else:
            self.params = single_params
            
        # Allocate states (PyTorch tensors with 9 coordinates & 9 velocities)
        if self.num_envs > 1:
            self.q = torch.zeros((self.num_envs, 9), dtype=torch.float32, device=self.device)
            self.u = torch.zeros((self.num_envs, 9), dtype=torch.float32, device=self.device)
            self.tau_prev = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=self.device)
        else:
            self.q = torch.zeros(9, dtype=torch.float32, device=self.device)
            self.u = torch.zeros(9, dtype=torch.float32, device=self.device)
            self.tau_prev = torch.zeros(6, dtype=torch.float32, device=self.device)
        
        # Track steps — always shape (num_envs,) for consistent tensor ops
        self.max_episode_steps = 500  # 10 seconds at 50 Hz control rate
        self.episode_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def set_reference_motion(self, q_ref, u_ref):
        # Store reference motion coordinates for Reference State Initialization (RSI)
        # We ensure they are 2D tensors of shape (num_samples, 9)
        self.q_ref = q_ref.to(device=self.device, dtype=torch.float32)
        self.u_ref = u_ref.to(device=self.device, dtype=torch.float32)
        if self.q_ref.ndim == 1:
            self.q_ref = self.q_ref.unsqueeze(0)
        if self.u_ref.ndim == 1:
            self.u_ref = self.u_ref.unsqueeze(0)

    def reset(self, seed=None, options=None):
        # Reset episode steps and initialize MTG active torque to nominal stance
        self.episode_steps.fill_(0)
        self.tau_prev.fill_(0.0)
        
        # Reference State Initialization (RSI)
        if hasattr(self, 'q_ref') and self.q_ref is not None:
            # Sample random indices from the reference gait database
            num_samples = self.q_ref.shape[0]
            if self.num_envs > 1:
                indices = torch.randint(0, num_samples, (self.num_envs,), device=self.device)
                self.q = self.q_ref[indices].clone()
                self.u = self.u_ref[indices].clone()
            else:
                idx = torch.randint(0, num_samples, (1,), device=self.device).item()
                self.q = self.q_ref[idx].clone()
                self.u = self.u_ref[idx].clone()
        else:
            # Fallback to nominal standing pose
            if self.num_envs > 1:
                self.q.fill_(0.0)
                self.q[:, 1] = 0.90
                self.u.fill_(0.0)
            else:
                self.q.fill_(0.0)
                self.q[1] = 0.90
                self.u.fill_(0.0)
            
        # Add small random noise to positions/velocities for diversity and exploration
        self.q += (torch.rand_like(self.q) - 0.5) * 0.02
        self.u += (torch.rand_like(self.u) - 0.5) * 0.02
        
        # Ensure horizontal position q_x (index 0) is initialized to exactly 0.0
        if self.num_envs > 1:
            self.q[:, 0] = 0.0
        else:
            self.q[0] = 0.0
            
        obs_tensor = self._get_obs()
        
        info = {"amp_obs": obs_tensor}
        return obs_tensor, info

    def _get_obs(self):
        # Concatenate positions (excluding horizontal position q_x at index 0) and velocities
        if self.num_envs > 1:
            return torch.cat([self.q[:, 1:], self.u], dim=-1)
        else:
            return torch.cat([self.q[1:], self.u], dim=-1)

    def step(self, actions):
        # Ensure actions are a PyTorch tensor on the correct device
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        else:
            actions = actions.to(device=self.device, dtype=torch.float32)
        
        # Handle 1D vs 2D actions depending on vectorized environments
        if self.num_envs == 1:
            if actions.ndim > 1:
                actions = actions.squeeze(0)
        else:
            if actions.ndim == 1:
                actions = actions.unsqueeze(0)
            
        # Clip residual actions to [-1, 1]
        actions = torch.clamp(actions, min=self.action_limits_low_tensor, max=self.action_limits_high_tensor)
        
        # Separate joint angle residuals (actions 0-5) and pelvis height residual (action 6)
        joint_actions = actions[..., :6]
        height_action = actions[..., 6:7]
        
        # Calculate residual target joint angles: q_des = q_ref_human + 0.15 * residual_action
        if hasattr(self, 'q_ref') and self.q_ref is not None:
            num_samples = self.q_ref.shape[0]
            ref_indices = torch.remainder(self.episode_steps, num_samples)
            if self.num_envs > 1:
                q_ref_curr = self.q_ref[ref_indices, 3:9]
            else:
                q_ref_curr = self.q_ref[ref_indices.item(), 3:9]
            q_des = q_ref_curr + 0.15 * joint_actions
        else:
            q_des = joint_actions
            
        # Apply subtle pelvis height offset (+/- 5 cm) from action 6
        if self.num_envs > 1:
            self.q[:, 1:2] = self.q[:, 1:2] + 0.001 * height_action
        else:
            self.q[1] = self.q[1] + 0.001 * height_action.squeeze()
        
        # Step the dynamics simulation (20 substeps of 0.001s = 0.02s macro-step)
        self.q, self.u, self.tau_prev, nan_occurred = step_environment(
            self.q, self.u, q_des, self.params,
            dt_sim=0.001, dt_control=0.02,
            mtg_controller=self.mtg_controller,
            tau_prev=self.tau_prev,
            device=self.device
        )
        
        self.episode_steps += 1
        info = {}
        
         # Calculate rewards: Reward standing upright, staying close to initial height, and moving forward
        if self.num_envs > 1:
            torso_angle = self.q[:, 2]
            hip_height = self.q[:, 1]
            vel_x = self.u[:, 0]
            
            upright_reward = torch.exp(-6.0 * (torso_angle ** 2))
            height_reward = torch.exp(-3.0 * ((hip_height - 0.90) ** 2))
            forward_reward = torch.exp(-((vel_x - 1.0) ** 2) / 0.25)
            control_penalty = 0.01 * torch.sum(actions ** 2, dim=-1)
            
            # 1. Left and right foot/ankle heights
            y_ankle_l = hip_height - 0.45 * torch.cos(torso_angle + self.q[:, 3]) - 0.45 * torch.cos(torso_angle + self.q[:, 3] + self.q[:, 4])
            y_ankle_r = hip_height - 0.45 * torch.cos(torso_angle + self.q[:, 6]) - 0.45 * torch.cos(torso_angle + self.q[:, 6] + self.q[:, 7])
            
            # 2. Relative horizontal velocities of ankles to hip
            theta_lh = torso_angle + self.q[:, 3]
            theta_lk = theta_lh + self.q[:, 4]
            v_rel_l = 0.45 * (self.u[:, 2] + self.u[:, 3]) * torch.cos(theta_lh) + \
                      0.45 * (self.u[:, 2] + self.u[:, 3] + self.u[:, 4]) * torch.cos(theta_lk)
                      
            theta_rh = torso_angle + self.q[:, 6]
            theta_rk = theta_rh + self.q[:, 7]
            v_rel_r = 0.45 * (self.u[:, 2] + self.u[:, 6]) * torch.cos(theta_rh) + \
                      0.45 * (self.u[:, 2] + self.u[:, 6] + self.u[:, 7]) * torch.cos(theta_rk)
            # 3. Compute velocity-thresholded toe-scuffing penalty (using exact OpenSim foot dimensions: 3.5cm sole, 14cm toe)
            theta_la = theta_lk + self.q[:, 5]
            theta_ra = theta_rk + self.q[:, 8]
            y_toe_l = y_ankle_l - 0.035 * torch.cos(theta_la) + 0.14 * torch.sin(theta_la)
            y_toe_r = y_ankle_r - 0.035 * torch.cos(theta_ra) + 0.14 * torch.sin(theta_ra)
            
            # Only penalize if toe is within 1.5cm of ground AND forward swing velocity exceeds 0.5 m/s
            scuff_l = torch.clamp(0.015 - y_toe_l, min=0.0) * torch.clamp(v_rel_l - 0.5, min=0.0)
            scuff_r = torch.clamp(0.015 - y_toe_r, min=0.0) * torch.clamp(v_rel_r - 0.5, min=0.0)
            scuff_penalty = 10.0 * (scuff_l + scuff_r)
            
            # Scale standing rewards by forward progression (must move at least 0.5 m/s for full reward)
            forward_progression = torch.clamp(vel_x / 0.5, min=0.0, max=1.0)
            
            reward = 3.0 * forward_reward + 2.0 * upright_reward + 1.0 * height_reward - control_penalty - scuff_penalty
            reward = torch.clamp(reward, min=-10.0, max=10.0)
            
            velocity_explosion = (torch.abs(vel_x) > 5.0) | (torch.max(torch.abs(self.u), dim=-1)[0] > 50.0)
            terminated = (torch.abs(torso_angle) > 1.22) | (hip_height < 0.60) | velocity_explosion | nan_occurred
            truncated = self.episode_steps >= self.max_episode_steps
        else:
            torso_angle = self.q[2]
            hip_height = self.q[1]
            vel_x = self.u[0]
            
            upright_reward = torch.exp(-6.0 * (torso_angle ** 2))
            height_reward = torch.exp(-3.0 * ((hip_height - 0.90) ** 2))
            forward_reward = torch.exp(-((vel_x - 1.0) ** 2) / 1.0)
            control_penalty = 0.01 * torch.sum(actions ** 2, dim=-1)
            
            # 1. Left and right foot/ankle heights
            y_ankle_l = hip_height - 0.45 * torch.cos(torso_angle + self.q[3]) - 0.45 * torch.cos(torso_angle + self.q[3] + self.q[4])
            y_ankle_r = hip_height - 0.45 * torch.cos(torso_angle + self.q[6]) - 0.45 * torch.cos(torso_angle + self.q[6] + self.q[7])
            
            # 2. Relative horizontal velocities of ankles to hip
            theta_lh = torso_angle + self.q[3]
            theta_lk = theta_lh + self.q[4]
            v_rel_l = 0.45 * (self.u[2] + self.u[3]) * torch.cos(theta_lh) + \
                      0.45 * (self.u[2] + self.u[3] + self.u[4]) * torch.cos(theta_lk)
                      
            theta_rh = torso_angle + self.q[6]
            theta_rk = theta_rh + self.q[7]
            v_rel_r = 0.45 * (self.u[2] + self.u[6]) * torch.cos(theta_rh) + \
                      0.45 * (self.u[2] + self.u[6] + self.u[7]) * torch.cos(theta_rk)
            # 3. Compute velocity-thresholded toe-scuffing penalty (using exact OpenSim foot dimensions: 3.5cm sole, 14cm toe)
            theta_la = theta_lk + self.q[5]
            theta_ra = theta_rk + self.q[8]
            y_toe_l = y_ankle_l - 0.035 * torch.cos(theta_la) + 0.14 * torch.sin(theta_la)
            y_toe_r = y_ankle_r - 0.035 * torch.cos(theta_ra) + 0.14 * torch.sin(theta_ra)
            
            # Only penalize if toe is within 1.5cm of ground AND forward swing velocity exceeds 0.5 m/s
            scuff_l = torch.clamp(0.015 - y_toe_l, min=0.0) * torch.clamp(v_rel_l - 0.5, min=0.0)
            scuff_r = torch.clamp(0.015 - y_toe_r, min=0.0) * torch.clamp(v_rel_r - 0.5, min=0.0)
            scuff_penalty = 10.0 * (scuff_l + scuff_r)
            
            # Scale standing rewards by forward progression
            forward_progression = torch.clamp(vel_x / 0.5, min=0.0, max=1.0)
            
            reward = 1.0 * forward_reward + 2.0 * upright_reward + 1.0 * height_reward - control_penalty - scuff_penalty
            reward = torch.clamp(reward, min=-10.0, max=10.0)
            
            velocity_explosion = (torch.abs(vel_x) > 5.0) | (torch.max(torch.abs(self.u)) > 50.0)
            terminated = ((torch.abs(torso_angle) > 1.22) | (hip_height < 0.60) | velocity_explosion | nan_occurred).unsqueeze(0)
            truncated = (self.episode_steps >= self.max_episode_steps)
        
        obs_tensor = self._get_obs()
        
        # 4. Auto-reset terminated/truncated sub-environments individually
        if self.num_envs > 1:
            reset_indices = torch.where(terminated | truncated)[0]
            if len(reset_indices) > 0:
                # Save final observations before resetting
                info["final_observation"] = obs_tensor.clone()
                info["_final_observation"] = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                info["_final_observation"][reset_indices] = True
                
                # Reset states for these environments using RSI
                if hasattr(self, 'q_ref') and self.q_ref is not None:
                    num_samples = self.q_ref.shape[0]
                    ref_indices = torch.randint(0, num_samples, (len(reset_indices),), device=self.device)
                    self.q[reset_indices] = self.q_ref[ref_indices].clone()
                    self.u[reset_indices] = self.u_ref[ref_indices].clone()
                else:
                    self.q[reset_indices] = torch.zeros(len(reset_indices), 9, dtype=torch.float32, device=self.device)
                    self.q[reset_indices, 1] = 0.90
                    self.u[reset_indices] = 0.0
                
                # Add random noise for exploration
                self.q[reset_indices] += (torch.rand((len(reset_indices), 9), device=self.device) - 0.5) * 0.02
                self.u[reset_indices] += (torch.rand((len(reset_indices), 9), device=self.device) - 0.5) * 0.02
                
                # Reset position offsets and counters
                self.q[reset_indices, 0] = 0.0
                self.episode_steps[reset_indices] = 0
                self.tau_prev[reset_indices] = 0.0
                
                # Recalculate observations so the returned obs contains the fresh reset states
                obs_tensor = self._get_obs()
        
        info["amp_obs"] = obs_tensor
        return obs_tensor, reward, terminated, truncated, info

if __name__ == '__main__':
    print("Testing BipedAMPEnv on GPU...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = BipedAMPEnv(num_envs=4, device=device)
    obs, info = env.reset()
    print("Initial observations shape:", obs.shape)
    
    # Take a step with zero action
    actions = torch.zeros((4, 6), device=device)
    next_obs, reward, term, trunc, info = env.step(actions)
    print("Next observations shape:", next_obs.shape)
    print("Rewards:", reward)
    print("Terminated:", term)
    print("Truncated:", trunc)
    print("BipedAMPEnv test successful!")
