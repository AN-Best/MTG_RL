import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import urllib.request

# Import environment and dynamics
from Biped_Gym_Env import BipedAMPEnv
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.amp import AMP, AMP_CFG
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin

# Define device for agent/networks (GPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------------------------------------------------------
# 1. Neural Network Model Definitions
# -----------------------------------------------------------------------------
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, role="policy", clip_actions=clip_actions,
                               clip_log_std=clip_log_std, min_log_std=min_log_std, max_log_std=max_log_std)
        
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, self.num_actions)
        )
        self.log_std = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        states = inputs.get("states")
        if states is None:
            states = inputs.get("observations")
        return self.net(states), {"log_std": self.log_std}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, role="value")
        
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1)
        )

    def compute(self, inputs, role):
        states = inputs.get("states")
        if states is None:
            states = inputs.get("observations")
        return self.net(states), {}

class Discriminator(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, role="discriminator")
        
        # Discriminator classifies observations (states) as real or generated
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1)
        )

    def compute(self, inputs, role):
        states = inputs.get("states")
        if states is None:
            states = inputs.get("observations")
        return self.net(states), {}

# -----------------------------------------------------------------------------
# 2. Reference Motion Loader
# -----------------------------------------------------------------------------
def load_reference_motion(num_samples=10000, device="cpu"):
    """Loads OpenSim reference walking coordinates and interpolates to build
    a continuous cycle dataset for adversarial motion prior training.
    """
    url = "https://raw.githubusercontent.com/opensim-org/opensim-models/master/Pipelines/Gait2392_Simbody/subject01_walk1_ik.mot"
    local_path = "subject01_walk1_ik.mot"
    
    if not os.path.exists(local_path):
        try:
            print("Downloading reference walking motion from OpenSim repository...")
            urllib.request.urlretrieve(url, local_path)
        except Exception as e:
            print(f"Download failed ({e}). Generating fallback mathematical walking motion.")
            
    if os.path.exists(local_path):
        try:
            # Parse OpenSim .mot file
            with open(local_path, 'r') as f:
                lines = f.readlines()
            
            # Find header end
            data_start = 0
            for idx, line in enumerate(lines):
                if line.strip() == "endheader":
                    data_start = idx + 1
                    break
                    
            col_headers = lines[data_start].strip().split()
            data_lines = [list(map(float, line.strip().split())) for line in lines[data_start+1:] if line.strip()]
            data = np.array(data_lines)
            
            # Map joints
            times = data[:, 0]
            
            # Indices for hip, knee, ankle flexion angles in subject01
            idx_hip_l = col_headers.index("hip_flexion_l")
            idx_knee_l = col_headers.index("knee_angle_l")
            idx_ankle_l = col_headers.index("ankle_angle_l")
            
            idx_hip_r = col_headers.index("hip_flexion_r")
            idx_knee_r = col_headers.index("knee_angle_r")
            idx_ankle_r = col_headers.index("ankle_angle_r")
            
            # Convert degrees to radians
            conv = np.pi / 180.0
            
            # Extract raw coordinates
            q_lh_raw = data[:, idx_hip_l] * conv
            q_lk_raw = -data[:, idx_knee_l] * conv # invert knee flexion sign to match dynamics
            q_la_raw = data[:, idx_ankle_l] * conv
            
            q_rh_raw = data[:, idx_hip_r] * conv
            q_rk_raw = -data[:, idx_knee_r] * conv # invert knee flexion sign
            q_ra_raw = data[:, idx_ankle_r] * conv
            
            q_tr_raw = np.zeros_like(times) # simple vertical torso assumption
            
            q_x_raw = data[:, col_headers.index("pelvis_tx")]
            # Start at x=0
            q_x_raw = q_x_raw - q_x_raw[0]
            q_y_raw = data[:, col_headers.index("pelvis_ty")]
            
            # Interpolate to build continuous cycle dataset of length `num_samples`
            # Gait period (assume subject01_walk1_ik is periodic or we loop it)
            cycle_distance = q_x_raw[-1] - q_x_raw[0]
            cycle_duration = times[-1] - times[0]
            target_t = np.linspace(times[0], times[0] + cycle_duration * (num_samples / 100), num_samples)
            mod_t = times[0] + np.mod(target_t - times[0], cycle_duration)
            cycle_idx = np.floor((target_t - times[0]) / cycle_duration)
            
            q_x = np.interp(mod_t, times, q_x_raw) + cycle_idx * cycle_distance
            q_y = np.interp(mod_t, times, q_y_raw)
            q_tr = np.interp(mod_t, times, q_tr_raw)
            q_lh = np.interp(mod_t, times, q_lh_raw)
            q_lk = np.interp(mod_t, times, q_lk_raw)
            q_la = np.interp(mod_t, times, q_la_raw)
            q_rh = np.interp(mod_t, times, q_rh_raw)
            q_rk = np.interp(mod_t, times, q_rk_raw)
            q_ra = np.interp(mod_t, times, q_ra_raw)
            
            q = np.stack([q_x, q_y, q_tr, q_lh, q_lk, q_la, q_rh, q_rk, q_ra], axis=-1)
            
            # Numerical velocities u = dq/dt
            dt = target_t[1] - target_t[0]
            u = np.zeros_like(q)
            u[1:-1] = (q[2:] - q[:-2]) / (2.0 * dt)
            u[0] = (q[1] - q[0]) / dt
            u[-1] = (q[-1] - q[-2]) / dt
            
            q_tensor = torch.tensor(q, dtype=torch.float32, device=device)
            u_tensor = torch.tensor(u, dtype=torch.float32, device=device)
            
            print(f"Successfully loaded and parsed {len(q)} reference steps from {local_path}!")
            # Note: Observation excludes q_x (index 0), so we slice q_tensor to match BipedAMPEnv observation space!
            # The AMP discriminator compares observations (17D).
            obs_ref = torch.cat([q_tensor[:, 1:], u_tensor], dim=-1)
            return obs_ref
            
        except Exception as e:
            print(f"Error parsing local walk file ({e}). Falling back to mathematical biped gait generation.")

    # FALLBACK GENERATOR: Periodic mathematical gait
    print("Generating fallback mathematical reference motion...")
    t = torch.linspace(0, num_samples * 0.1, num_samples, device=device)
    
    # Translation
    q_x = 1.0 * t
    q_y = 0.90 + 0.03 * torch.cos(4.0 * torch.pi * t)
    
    # Left leg reference kinematics
    q_lh = 0.2 + 0.35 * torch.cos(2.0 * torch.pi * t - 0.5)
    q_lk = -0.55 + 0.55 * torch.cos(2.0 * torch.pi * t - 1.5) - 0.1 * torch.sin(4.0 * torch.pi * t)
    q_la = 0.02 + 0.08 * torch.sin(2.0 * torch.pi * t) + 0.05 * torch.cos(4.0 * torch.pi * t - 0.5)
    
    # Right leg kinematics (50% phase shift)
    t_r = t + 0.5
    q_rh = 0.2 + 0.35 * torch.cos(2.0 * torch.pi * t_r - 0.5)
    q_rk = -0.55 + 0.55 * torch.cos(2.0 * torch.pi * t_r - 1.5) - 0.1 * torch.sin(4.0 * torch.pi * t_r)
    q_ra = 0.02 + 0.08 * torch.sin(2.0 * torch.pi * t_r) + 0.05 * torch.cos(4.0 * torch.pi * t_r - 0.5)
    
    # Torso: slightly oscillates around vertical
    q_tr = 0.02 * torch.sin(2.0 * torch.pi * t)
    
    q = torch.stack([q_x, q_y, q_tr, q_lh, q_lk, q_la, q_rh, q_rk, q_ra], dim=-1)
    
    # Compute analytical velocities u = dq/dt
    u_x = torch.ones_like(t) * 1.0
    u_y = -0.12 * torch.pi * torch.sin(4.0 * torch.pi * t)
    u_lh = -0.35 * 2.0 * torch.pi * torch.sin(2.0 * torch.pi * t - 0.5)
    u_lk = -0.55 * 2.0 * torch.pi * torch.sin(2.0 * torch.pi * t - 1.5) - 0.4 * torch.pi * torch.cos(4.0 * torch.pi * t)
    u_la = 0.08 * 2.0 * torch.pi * torch.cos(2.0 * torch.pi * t) - 0.2 * torch.pi * torch.sin(4.0 * torch.pi * t - 0.5)
    
    u_rh = -0.35 * 2.0 * torch.pi * torch.sin(2.0 * torch.pi * t_r - 0.5)
    u_rk = -0.55 * 2.0 * torch.pi * torch.sin(2.0 * torch.pi * t_r - 1.5) - 0.4 * torch.pi * torch.cos(4.0 * torch.pi * t_r)
    u_ra = 0.08 * 2.0 * torch.pi * torch.cos(2.0 * torch.pi * t_r) - 0.2 * torch.pi * torch.sin(4.0 * torch.pi * t_r - 0.5)
    
    u_tr = 0.02 * 2.0 * torch.pi * torch.cos(2.0 * torch.pi * t)
    
    u = torch.stack([u_x, u_y, u_tr, u_lh, u_lk, u_la, u_rh, u_rk, u_ra], dim=-1)
    
    obs = torch.cat([q[:, 1:], u], dim=-1)
    return obs

# -----------------------------------------------------------------------------
# 3. Setup Environment & Agent
# -----------------------------------------------------------------------------
print("Initializing environment...")
env = BipedAMPEnv(num_envs=2048, device=device)
env = wrap_env(env)

# Define models
models = {
    "policy": Policy(env.observation_space, env.action_space, device),
    "value": Value(env.observation_space, env.action_space, device),
    "discriminator": Discriminator(env.observation_space, env.action_space, device)
}

# Define configuration
cfg = AMP_CFG()
cfg.rollouts = 16
cfg.learning_epochs = 4
cfg.mini_batches = 2
cfg.learning_rate = 5e-5
cfg.state_preprocessor = None  # Disable built-in preprocessor for simplicity

# Configure TensorBoard logging
cfg.experiment.directory = "runs"
cfg.experiment.experiment_name = "biped_amp"
cfg.experiment.write_interval = 1
cfg.experiment.checkpoint_interval = 10

# Define memory buffers (memory uses env.num_envs, reply_buffer uses 1 for flat discriminator samples)
# memory_size = 16 matches cfg.rollouts, saving massive GPU memory at scale
memory = RandomMemory(memory_size=16, device=device, num_envs=env.num_envs)
reply_buffer = RandomMemory(memory_size=200000, device=device, num_envs=1, replacement=True)

# Load reference motion data (Downloads from OpenSim)
print("Loading reference motion dataset...")
ref_obs = load_reference_motion(num_samples=10000, device=device)

# Create and populate motion dataset
motion_dataset = RandomMemory(memory_size=10000, device=device, replacement=True)

# Define the collect_reference_motions callable expected by skrl AMP agent
def collect_reference_motions(num_samples):
    indices = torch.randint(0, len(ref_obs), (num_samples,), device=device)
    return ref_obs[indices]

print("Initializing AMP agent...")
agent = AMP(
    models=models,
    memory=memory,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
    cfg=cfg,
    amp_observation_space=env.observation_space, # Discriminator observes the full state
    motion_dataset=motion_dataset,
    reply_buffer=reply_buffer,
    collect_reference_motions=collect_reference_motions
)

# -----------------------------------------------------------------------------
# 4. Training Loop
# -----------------------------------------------------------------------------
TOTAL_TIMESTEPS     = 30_000_000
ROLLOUT_STEPS       = cfg.rollouts      # steps per env per post_interaction
LOG_INTERVAL        = 10               # update intervals between console prints
CHECKPOINT_INTERVAL = 100             # update intervals between checkpoints

def _fmt(n):
    """Human-readable integer: 1_234_567 -> '1.23M'"""
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)

def print_header():
    print("\n" + "=" * 80)
    print(f"  PPO-AMP Biped Training   |   device: {device}   |   envs: {env.num_envs:,}")
    print(f"  Total timesteps: {_fmt(TOTAL_TIMESTEPS)}   |   Rollout: {ROLLOUT_STEPS} steps/env")
    print("=" * 80)
    print(f"  {'Step':>8}  {'Progress':40}  {'Reward':>8}  {'D-Loss':>8}  {'P-Loss':>8}  {'V-Loss':>8}")
    print("-" * 80)

def print_progress(timestep, total, reward, d_loss, p_loss, v_loss):
    frac  = min(timestep / total, 1.0)
    filled = int(40 * frac)
    bar   = "#" * filled + "-" * (40 - filled)
    def _f(v, fmt): return fmt.format(v) if v is not None else "       -"
    print(
        f"  {_fmt(timestep):>8}  [{bar}]  "
        f"{_f(reward, '{:8.3f}')}  {_f(d_loss, '{:8.4f}')}  "
        f"{_f(p_loss, '{:8.4f}')}  {_f(v_loss, '{:8.4f}')}",
        flush=True
    )

if __name__ == '__main__':
    print_header()

    # skrl AMP needs its internal rollout counter seeded; call pre_interaction once
    # (post_interaction triggers the update when _rollout % rollouts == 0)
    timestep = 0
    update   = 0
    last_reward = last_dloss = last_ploss = last_vloss = None

    states, infos = env.reset()
    # skrl uses keyword-only args — keep a `states` alias for the agent state input
    agent_states = states  # for non-hierarchical agents, states == observations

    while timestep < TOTAL_TIMESTEPS:
        # ---- signal start of rollout to agent ----
        agent.pre_interaction(timestep=timestep, timesteps=TOTAL_TIMESTEPS)

        # ---- rollout collection ----
        for _ in range(ROLLOUT_STEPS):
            with torch.no_grad():
                # act() signature: (observations, states, *, timestep, timesteps)
                actions, agent_outputs = agent.act(
                    states, agent_states,
                    timestep=timestep, timesteps=TOTAL_TIMESTEPS
                )

            next_states, rewards, terminated, truncated, infos = env.step(actions)

            # record_transition uses keyword-only args; states == observations for flat envs
            agent.record_transition(
                observations=states,
                states=agent_states,
                actions=actions,
                rewards=rewards,
                next_observations=next_states,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
                infos=infos,
                timestep=timestep,
                timesteps=TOTAL_TIMESTEPS
            )

            # ---- auto-reset terminated / truncated environments ----
            done = terminated | truncated
            if done.any():
                reset_obs, _ = env.reset()
                # Only overwrite the done envs
                next_states = next_states.clone()
                next_states[done] = reset_obs[done]

            states       = next_states
            agent_states = next_states
            timestep    += env.num_envs

        # ---- policy update ----
        agent.post_interaction(timestep=timestep, timesteps=TOTAL_TIMESTEPS)
        update += 1

        # ---- extract metrics (exact skrl key names from AMP.update) ----
        tracking = getattr(agent, '_tracking_data', {})
        def _last(key):
            vals = tracking.get(key, [])
            if vals:
                v = vals[-1]
                try: return float(v)
                except: pass
            return None
        last_reward = _last("Reward / Total reward (mean)") or last_reward
        last_dloss  = _last("Loss / Discriminator loss")    or last_dloss
        last_ploss  = _last("Loss / Policy loss")           or last_ploss
        last_vloss  = _last("Loss / Value loss")            or last_vloss

        if update % LOG_INTERVAL == 0:
            print_progress(timestep, TOTAL_TIMESTEPS, last_reward, last_dloss, last_ploss, last_vloss)

        if update % CHECKPOINT_INTERVAL == 0:
            ckpt_path = f"runs/biped_amp/checkpoints/agent_{timestep}.pt"
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            agent.save(ckpt_path)
            print(f"  [checkpoint] saved -> {ckpt_path}")

    print("\n" + "=" * 80)
    print("  Training complete!")
    agent.save("runs/biped_amp/checkpoints/agent_final.pt")
    print("  Final checkpoint saved: runs/biped_amp/checkpoints/agent_final.pt")
    print("=" * 80)
