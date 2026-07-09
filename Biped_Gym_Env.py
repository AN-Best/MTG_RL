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
        
        # 6 actuated joints: [left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle]
        self.num_actions = 6
        # 17 observed variables: 8 positions (q excluding q_x) and 9 velocities (u)
        self.num_observations = 17
        
        # Define action space: Target angles for the 6 actuated joints (in radians)
        self.action_limits_low = np.array([-0.5, -2.0, -0.5, -0.5, -2.0, -0.5], dtype=np.float32)
        self.action_limits_high = np.array([1.5, 0.0, 0.5, 1.5, 0.0, 0.5], dtype=np.float32)
        
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
            70.0, 10.0, 5.0, 1.5,   # Masses
            0.8, 0.45, 0.45, 0.25, 0.05, 0.2, # Lengths
            0.4, 0.2, 0.2, 0.08,    # COMs
            2.5, 0.1, 0.05, 0.01,   # Inertias
            9.81, 0.90,             # g, h_g
            5e6, 0.8, 0.6, 0.01     # kc, cc, mu, vs
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
        
        # Track steps
        self.max_episode_steps = 200  # 20 seconds at 10 Hz control rate
        self.episode_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def reset(self, seed=None, options=None):
        # Reset episode steps
        self.episode_steps.fill_(0)
        
        self.q.fill_(0.0)
        self.u.fill_(0.0)
        self.tau_prev.fill_(0.0)
        
        # Initialize hip height (q_y at index 1) to nominal height 0.90 m
        if self.num_envs > 1:
            self.q[:, 1] = 0.90
        else:
            self.q[1] = 0.90
            
        # Small random noise to positions/velocities for diversity on the GPU device
        self.q += (torch.rand_like(self.q) - 0.5) * 0.05
        self.u += (torch.rand_like(self.u) - 0.5) * 0.05
        
        # Ensure q_x (index 0) is initialized to exactly 0.0
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
            
        # Clip actions to joint limits
        actions = torch.clamp(actions, min=self.action_limits_low_tensor, max=self.action_limits_high_tensor)
        
        # Step the dynamics simulation (50 substeps of 0.002s = 0.1s macro-step)
        self.q, self.u, self.tau_prev = step_environment(
            self.q, self.u, actions, self.params,
            dt_sim=0.002, dt_control=0.1,
            mtg_controller=self.mtg_controller,
            tau_prev=self.tau_prev,
            device=self.device
        )
        
        self.episode_steps += 1
        
        # Calculate rewards: Reward standing upright, staying close to initial height, and moving forward
        if self.num_envs > 1:
            torso_angle = self.q[:, 2]
            hip_height = self.q[:, 1]
            vel_x = self.u[:, 0]
            
            upright_reward = torch.exp(-2.0 * (torso_angle ** 2))
            height_reward = torch.exp(-10.0 * ((hip_height - 0.90) ** 2))
            forward_reward = vel_x
            control_penalty = 0.01 * torch.sum(actions ** 2, dim=-1)
            
            reward = upright_reward + height_reward + 0.5 * forward_reward - control_penalty
            
            terminated = (torch.abs(torso_angle) > 0.78) | (hip_height < 0.60)
            truncated = self.episode_steps >= self.max_episode_steps
        else:
            torso_angle = self.q[2]
            hip_height = self.q[1]
            vel_x = self.u[0]
            
            upright_reward = torch.exp(-2.0 * (torso_angle ** 2))
            height_reward = torch.exp(-10.0 * ((hip_height - 0.90) ** 2))
            forward_reward = vel_x
            control_penalty = 0.01 * torch.sum(actions ** 2, dim=-1)
            
            reward = upright_reward + height_reward + 0.5 * forward_reward - control_penalty
            
            terminated = (torch.abs(torso_angle) > 0.78) or (hip_height < 0.60)
            # For num_envs=1, episode_steps has size (1,)
            truncated = self.episode_steps[0] >= self.max_episode_steps
        
        obs_tensor = self._get_obs()
        
        info = {"amp_obs": obs_tensor}
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
