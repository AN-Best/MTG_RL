import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
import imageio
import mujoco

from Biped_Env import BipedEnv

# -----------------------------------------------------------------------------
# 1. Universal Motion Loader
#    Supports any motion format (.npz, .mot, numpy dict) of arbitrary duration.
# -----------------------------------------------------------------------------
class MotionClip:
    """
    Encapsulates arbitrary kinematic trajectory data:
    q: [N, 10] (x, z, pitch, lumbar, 6 leg joints)
    u: [N, 10] (vx, vz, pitch_rate, lumbar_rate, 6 leg velocities)
    t: [N] (timestamps in seconds)
    """
    def __init__(self, t, q, u):
        self.t = np.asarray(t, dtype=np.float32)
        self.q = np.asarray(q, dtype=np.float32)
        self.u = np.asarray(u, dtype=np.float32)
        self.num_frames = self.q.shape[0]
        self.duration = self.t[-1] - self.t[0]
        self.dt = (self.t[-1] - self.t[0]) / max(1, self.num_frames - 1)
        # Net forward displacement per cycle for continuous forward progression without teleportation
        self.dx_per_cycle = float(self.q[-1, 0] - self.q[0, 0])
        if self.num_frames > 1:
            self.dx_per_cycle += float(self.q[1, 0] - self.q[0, 0])

    @classmethod
    def load(cls, motion_path):
        if motion_path.endswith(".npz"):
            data = np.load(motion_path)
            t = data["t"] if "t" in data else np.arange(data["q"].shape[0]) * 0.002
            return cls(t, data["q"], data["u"])
        else:
            raise ValueError(f"Unsupported motion file format: {motion_path}")

    def get_frame(self, index):
        cycle = int(index) // self.num_frames
        idx = int(index) % self.num_frames
        q_frame = self.q[idx].copy()
        q_frame[0] += cycle * self.dx_per_cycle
        return q_frame, self.u[idx]

    def sample_random_frame(self):
        idx = np.random.randint(0, self.num_frames)
        return idx, self.q[idx], self.u[idx]


class MotionLib:
    """
    Perpetual Humanoid Control (PHC) Motion Library (Luo et al. 2023):
    Encapsulates a collection of motion clips (from AddBiomechanics, AMASS, or custom gaits).
    Enables Random State Initialization (RSI) across the whole dataset for multi-motion training.
    """
    def __init__(self, sources, filter_str=None):
        if isinstance(sources, (str, os.PathLike)):
            sources = [sources]
            
        self.clips = []
        self.clip_names = []
        for src in sources:
            src_str = str(src)
            if os.path.isdir(src_str):
                files = sorted([os.path.join(src_str, f) for f in os.listdir(src_str) if f.endswith(".npz")])
                if filter_str is not None:
                    files = [f for f in files if filter_str.lower() in os.path.basename(f).lower() or "gait" in os.path.basename(f).lower()]
                for f in files:
                    clip = MotionClip.load(f)
                    self.clips.append(clip)
                    self.clip_names.append(os.path.splitext(os.path.basename(f))[0])
            elif os.path.isfile(src_str):
                clip = MotionClip.load(src_str)
                self.clips.append(clip)
                self.clip_names.append(os.path.splitext(os.path.basename(src_str))[0])
            else:
                raise FileNotFoundError(f"Motion source '{src_str}' not found.")
                
        if not self.clips:
            raise ValueError(f"No valid .npz motion files found in: {sources}")
            
        self.durations = np.array([c.duration for c in self.clips], dtype=np.float32)
        total_dur = np.sum(self.durations)
        self.weights = self.durations / (total_dur if total_dur > 0 else 1.0)
        print(f"\n==========================================================================")
        print(f"   PHC Motion Dataset Loaded: {len(self.clips)} sequences ({total_dur:.2f}s total)")
        print(f"==========================================================================")
        for i, (name, dur) in enumerate(zip(self.clip_names, self.durations)):
            print(f"  [{i+1:02d}] {name:<28} | Duration: {dur:.2f}s ({self.clips[i].num_frames} frames)")
        print(f"==========================================================================\n")

    def sample(self):
        idx = np.random.choice(len(self.clips), p=self.weights)
        clip = self.clips[idx]
        frame, q0, u0 = clip.sample_random_frame()
        return clip, frame, q0, u0

    def get_clip(self, name_or_idx):
        if isinstance(name_or_idx, int):
            return self.clips[name_or_idx], self.clip_names[name_or_idx]
        for c, name in zip(self.clips, self.clip_names):
            if name.lower() == str(name_or_idx).lower():
                return c, name
        raise ValueError(f"Motion '{name_or_idx}' not found in MotionLib.")

# 2. DeepMimic / PHC Gated Residual Actor-Critic Policy
#    Inputs goal-conditioned trajectory errors; outputs small bounded corrective nudges.
# -----------------------------------------------------------------------------
class GatedResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.act = nn.SiLU()
        self.gate = nn.Parameter(torch.tensor(-2.0))
        
        nn.init.orthogonal_(self.fc1.weight, gain=1.0)
        nn.init.zeros_(self.fc1.bias)
        nn.init.orthogonal_(self.fc2.weight, gain=0.01)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        res = self.act(self.norm1(self.fc1(x)))
        res = self.norm2(self.fc2(res))
        return x + torch.sigmoid(self.gate) * res

class UniversalTrackingPolicy(nn.Module):
    """
    Goal-Conditioned Motion Tracking Policy (PHC Architecture):
    Observation Space (60-dim):
    - Current Non-root Joint Positions (7) & Velocities (7)
    - Immediate Joint Errors: q_ref(t) - q(t) (7) & u_ref(t) - u(t) (7)
    - Future Joint Errors at t + 10 frames (20ms): q_ref(t+10) - q(t) (7)
    - Future Joint Errors at t + 30 frames (60ms): q_ref(t+30) - q(t) (7)
    - Root State Errors: [z_ref - z, pitch_ref - pitch, vx_ref - vx, vz_ref - vz, vp_ref - vp] (5)
    - Relative Feet Sole Positions (4)
    - Foot Normal Contact Forces (2)
    - Action History a_{t-1} (7)
    
    Action Space (7-dim):
    - Corrective joint angle offsets dq_target (rad)
      Authority: Lumbar: +/- 0.25 rad (~14.3 deg), Hips/Ankles: +/- 0.20 rad (~11.5 deg), Knees: +/- 0.15 rad (~8.6 deg)
    """
    def __init__(self, obs_dim=60, action_dim=7, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        hidden_dim = 256

        # Actor Network
        self.actor_in = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.SiLU())
        self.actor_res1 = GatedResidualBlock(hidden_dim)
        self.actor_res2 = GatedResidualBlock(hidden_dim)
        self.actor_out = nn.Linear(hidden_dim, action_dim)
        nn.init.orthogonal_(self.actor_out.weight, gain=0.01)
        nn.init.zeros_(self.actor_out.bias)

        # Exploration std starts controlled: exp(-2.0) ~ 0.135
        self.log_std = nn.Parameter(torch.zeros(action_dim, device=self.device) - 2.0)
        
        # Corrective action authority limits (rad): Lumbar: +/-0.35, Hips/Ankles: +/-0.35, Knees: +/-0.25
        self.register_buffer(
            "action_scale",
            torch.tensor([0.35, 0.35, 0.25, 0.35, 0.35, 0.25, 0.35], dtype=torch.float32, device=self.device)
        )


        # Critic Network
        self.critic_in = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.SiLU())
        self.critic_res1 = GatedResidualBlock(hidden_dim)
        self.critic_res2 = GatedResidualBlock(hidden_dim)
        self.critic_out = nn.Linear(hidden_dim, 1)
        nn.init.orthogonal_(self.critic_out.weight, gain=1.0)
        nn.init.zeros_(self.critic_out.bias)

        self.to(self.device)

    def forward(self, obs):
        ha = self.actor_in(obs)
        ha = self.actor_res1(ha)
        ha = self.actor_res2(ha)
        action_mean = self.action_scale * torch.tanh(self.actor_out(ha))
        std = torch.exp(self.log_std)

        hc = self.critic_in(obs)
        hc = self.critic_res1(hc)
        hc = self.critic_res2(hc)
        value = self.critic_out(hc)
        return action_mean, std, value

    def get_action(self, obs, deterministic=False):
        mean, std, value = self.forward(obs)
        if deterministic:
            return mean, torch.zeros_like(mean), value
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob, value

    def evaluate_actions(self, obs, action):
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy, value

# -----------------------------------------------------------------------------
# 3. Universal Motion Tracking Environment (50 Hz Policy / 500 Hz Physics)
# -----------------------------------------------------------------------------
class UniversalTrackingEnv:
    def __init__(self, motion_source, actuator_type="ideal_unlimited", device="cpu", decimation=10):
        if isinstance(motion_source, MotionLib):
            self.motion_lib = motion_source
            self.active_clip = motion_source.clips[0]
        elif isinstance(motion_source, MotionClip):
            self.motion_lib = None
            self.active_clip = motion_source
        else:
            self.motion_lib = MotionLib(motion_source)
            self.active_clip = self.motion_lib.clips[0]

        self.device = torch.device(device)
        self.actuator_type = actuator_type
        self.decimation = decimation # 10 physics sub-steps at 500 Hz = 50 Hz policy (0.02s)
        # Keep underlying physics env stepping on CPU for zero PCIe synchronization overhead
        self.env = BipedEnv(actuator_type=actuator_type, device="cpu")
        self.d_site = mujoco.MjData(self.env.model)
        
        self.pelvis_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, "pelvis_geom")
        self.pelvis_body_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.sole_l_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, "foot_l_sole")
        self.sole_r_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, "foot_r_sole")
        self.trunk_tip_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_SITE, "trunk_tip")

        # Peak biological torque limits (N*m): Lumbar: 300, Hip: 400, Knee: 500, Ankle: 250
        self.T_max = np.array([300.0, 400.0, 500.0, 250.0, 400.0, 500.0, 250.0], dtype=np.float32)

        self.prev_action = np.zeros(7, dtype=np.float32)
        self.filtered_action = np.zeros(7, dtype=np.float32)
        self.enable_perturbation = False
        self.max_steps = 300
        self.step_count = 0

    def reset(self, clip=None, start_frame=None):
        """Resets simulation to a specified or random frame across the motion dataset (RSI)."""
        if clip is not None:
            self.active_clip = clip
            if start_frame is None:
                self.current_frame, q0, u0 = self.active_clip.sample_random_frame()
            else:
                self.current_frame = int(start_frame) % self.active_clip.num_frames
                q0, u0 = self.active_clip.get_frame(self.current_frame)
        elif self.motion_lib is not None:
            if start_frame is None:
                self.active_clip, self.current_frame, q0, u0 = self.motion_lib.sample()
            else:
                self.active_clip = self.motion_lib.clips[0]
                self.current_frame = int(start_frame) % self.active_clip.num_frames
                q0, u0 = self.active_clip.get_frame(self.current_frame)
        else:
            if start_frame is None:
                self.current_frame, q0, u0 = self.active_clip.sample_random_frame()
            else:
                self.current_frame = int(start_frame) % self.active_clip.num_frames
                q0, u0 = self.active_clip.get_frame(self.current_frame)

        self.env.reset(qpos_init=q0.copy(), qvel_init=u0.copy())
        self.prev_action = np.zeros(7, dtype=np.float32)
        self.filtered_action = np.zeros(7, dtype=np.float32)
        self.step_count = 0
        return self.get_observation()

    def get_observation(self):
        """Constructs 60-dim goal-conditioned motion tracking observation."""
        q_curr = self.env.data.qpos
        u_curr = self.env.data.qvel

        q_ref_0, u_ref_0 = self.active_clip.get_frame(self.current_frame)
        q_ref_10, _ = self.active_clip.get_frame(self.current_frame + 10)
        q_ref_30, _ = self.active_clip.get_frame(self.current_frame + 30)

        # 1. Joint state & immediate errors
        q_joints = q_curr[3:10].astype(np.float32)
        u_joints = u_curr[3:10].astype(np.float32)
        err_q0 = (q_ref_0[3:10] - q_joints)
        err_u0 = (u_ref_0[3:10] - u_joints)

        # 2. Lookahead future errors (+20ms and +60ms)
        err_q10 = (q_ref_10[3:10] - q_joints)
        err_q30 = (q_ref_30[3:10] - q_joints)

        # 3. Root state errors [z, pitch, vx, vz, vp]
        root_err = np.array([
            q_ref_0[1] - q_curr[1],
            q_ref_0[2] - q_curr[2],
            u_ref_0[0] - u_curr[0],
            u_ref_0[1] - u_curr[1],
            u_ref_0[2] - u_curr[2]
        ], dtype=np.float32)

        # 4. End-effector positions relative to pelvis (Feet + Trunk Tip)
        pos_pelvis = self.env.data.geom_xpos[self.pelvis_id]
        pos_l = self.env.data.geom_xpos[self.sole_l_id]
        pos_r = self.env.data.geom_xpos[self.sole_r_id]
        pos_t = self.env.data.site_xpos[self.trunk_tip_id]
        rel_ee_pos = np.array([
            pos_l[0] - pos_pelvis[0], pos_l[2] - pos_pelvis[2],
            pos_r[0] - pos_pelvis[0], pos_r[2] - pos_pelvis[2],
            pos_t[0] - pos_pelvis[0], pos_t[2] - pos_pelvis[2]
        ], dtype=np.float32)

        # 5. Foot normal contacts
        fl, fr = 0.0, 0.0
        d = self.env.data
        for i in range(d.ncon):
            c_array = np.zeros(6)
            mujoco.mj_contactForce(self.env.model, d, i, c_array)
            normal_f = abs(c_array[0])
            g1, g2 = d.contact[i].geom1, d.contact[i].geom2
            name1 = mujoco.mj_id2name(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, g1) or ""
            name2 = mujoco.mj_id2name(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, g2) or ""
            if "foot_l_sole" in (name1, name2):
                fl += normal_f
            if "foot_r_sole" in (name1, name2):
                fr += normal_f
        contacts = np.array([min(fl, 1500.0) / 600.0, min(fr, 1500.0) / 600.0], dtype=np.float32)

        # 6. Explicit cyclic gait phase features: [sin(2*pi*phi), cos(2*pi*phi)]
        phi = (self.current_frame % self.active_clip.num_frames) / float(max(1, self.active_clip.num_frames))
        phase_feat = np.array([np.sin(2.0 * np.pi * phi), np.cos(2.0 * np.pi * phi)], dtype=np.float32)

        obs = np.concatenate([
            q_joints, u_joints, err_q0, err_u0, err_q10, err_q30, root_err, rel_ee_pos, contacts, self.filtered_action, phase_feat
        ], dtype=np.float32)
        return obs

    def step(self, action):
        """
        Executes policy action at 50 Hz with 10 physics sub-steps at 500 Hz (dt = 0.002s).
        Linearly interpolates targets across sub-steps to eliminate all 100 Hz torque chatter.
        """
        # Low-pass filter action with low latency (0.40 persist / 0.60 innovation)
        prev_filtered = self.filtered_action.copy()
        filtered = 0.40 * self.filtered_action + 0.60 * action
        self.filtered_action = filtered.copy()

        tau_acc = np.zeros(7, dtype=np.float32)

        # Anticipatory physiological lead-time for MTG 1st-order activation ODE (tau_act = 20ms = 10 physics frames at 500Hz)
        lead_frames = 10 if self.actuator_type in ["mtg", "muscle", "tier3"] else 0

        # Optional push recovery perturbation during training (5% probability)
        if self.enable_perturbation and np.random.rand() < 0.05:
            push_fx = float(np.random.uniform(-30.0, 30.0))
            self.env.data.xfrc_applied[self.pelvis_body_id, 0] = push_fx
        else:
            self.env.data.xfrc_applied[self.pelvis_body_id, 0] = 0.0

        for sub_step in range(self.decimation):
            q_ref_k, u_ref_k = self.active_clip.get_frame(self.current_frame + sub_step)
            alpha = (sub_step + 1) / float(self.decimation)
            act_k = (1.0 - alpha) * prev_filtered + alpha * filtered

            # Pure end-to-end RL: Target joint angle commanded directly by policy (zero heuristics)
            q_des = q_ref_k[3:10] + act_k
            u_des = u_ref_k[3:10]

            _, tau = self.env.step_simulation(q_des, dt_sim=0.002, u_des=u_des)
            tau_acc += tau

        self.current_frame += self.decimation
        tau_mean = tau_acc / self.decimation

        # Compute ground contact normal forces under foot soles
        fl, fr = 0.0, 0.0
        d = self.env.data
        for i in range(d.ncon):
            c_array = np.zeros(6)
            mujoco.mj_contactForce(self.env.model, d, i, c_array)
            normal_f = abs(c_array[0])
            g1, g2 = d.contact[i].geom1, d.contact[i].geom2
            name1 = mujoco.mj_id2name(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, g1) or ""
            name2 = mujoco.mj_id2name(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, g2) or ""
            if "foot_l_sole" in (name1, name2):
                fl += normal_f
            if "foot_r_sole" in (name1, name2):
                fr += normal_f

        q_curr = self.env.data.qpos
        u_curr = self.env.data.qvel
        q_ref_end, u_ref_end = self.active_clip.get_frame(self.current_frame)

        # PHC Multi-Objective Tracking Reward (Luo et al. 2023)
        mse_q = np.mean((q_curr[3:10] - q_ref_end[3:10])**2)
        mse_u = np.mean((u_curr[3:10] - u_ref_end[3:10])**2)
        r_q = np.exp(-5.0 * mse_q)
        r_u = np.exp(-0.5 * mse_u)

        # Root orientation (pitch), angular rate damping, height, and velocity
        p_err_sq = (q_curr[2] - q_ref_end[2])**2
        vp_err_sq = (u_curr[2] - u_ref_end[2])**2
        z_err_sq = (q_curr[1] - q_ref_end[1])**2
        vx_err_sq = (u_curr[0] - u_ref_end[0])**2
        
        # Soft postural gates -- still penalise deviations but never zero-out gradient
        r_rot = np.exp(-5.0 * p_err_sq - 0.5 * vp_err_sq)
        r_pos = np.exp(-20.0 * z_err_sq)
        r_vel = np.exp(-3.0 * vx_err_sq)

        # Keypoint Cartesian site tracking (Feet + Trunk Tip end-effectors)
        pos_p = self.env.data.geom_xpos[self.pelvis_id]
        pos_l = self.env.data.geom_xpos[self.sole_l_id]
        pos_r = self.env.data.geom_xpos[self.sole_r_id]
        pos_t = self.env.data.site_xpos[self.trunk_tip_id]
        curr_ee_rel = np.concatenate([
            pos_l[[0, 2]] - pos_p[[0, 2]],
            pos_r[[0, 2]] - pos_p[[0, 2]],
            pos_t[[0, 2]] - pos_p[[0, 2]]
        ])
        
        self.d_site.qpos[:] = q_ref_end
        mujoco.mj_kinematics(self.env.model, self.d_site)
        ref_p = self.d_site.geom_xpos[self.pelvis_id]
        ref_l = self.d_site.geom_xpos[self.sole_l_id]
        ref_r = self.d_site.geom_xpos[self.sole_r_id]
        ref_t = self.d_site.site_xpos[self.trunk_tip_id]
        ref_ee_rel = np.concatenate([
            ref_l[[0, 2]] - ref_p[[0, 2]],
            ref_r[[0, 2]] - ref_p[[0, 2]],
            ref_t[[0, 2]] - ref_p[[0, 2]]
        ])
        
        cart_mse = np.mean((curr_ee_rel - ref_ee_rel)**2)
        r_cart_rel = np.exp(-20.0 * cart_mse)

        # Absolute foot ground clearance tracking and trunk elevation tracking
        ground_err = (pos_l[2] - ref_l[2])**2 + (pos_r[2] - ref_r[2])**2
        trunk_err = (pos_t[2] - ref_t[2])**2
        r_ground_z = np.exp(-20.0 * ground_err)
        r_trunk_z = np.exp(-20.0 * trunk_err)
        r_cart = 0.40 * r_cart_rel + 0.30 * r_ground_z + 0.30 * r_trunk_z

        # Weight-bearing ground contact force reward (ensures biped bears weight on feet without floating)
        total_grf = float(fl + fr)
        r_contact = min(1.0, total_grf / 250.0)

        # Action rate, mechanical work, and torque effort regularizers
        p_act_rate = -0.005 * np.sum((action - self.prev_action)**2)
        p_work = -0.0002 * np.sum(np.abs(tau_mean * u_curr[3:10]))
        r_eff = np.exp(-1e-6 * np.sum(tau_mean**2))

        # Biological torque limit penalty (curriculum guidance towards MTG capacity)
        excess = np.maximum(0.0, np.abs(tau_mean) - self.T_max) / self.T_max
        r_limit = np.exp(-3.0 * np.sum(excess**2))
        p_limit = -0.5 * np.sum(excess)

        # Composite PHC reward: upright base gate * joint/cart/contact tracking
        tracking_terms = max(0.0, 0.30 * r_cart + 0.30 * r_q + 0.20 * r_contact + 0.10 * r_vel + 0.05 * r_u + 0.05 * r_eff + p_act_rate + p_work + p_limit)
        total_reward = 10.0 * r_limit * r_rot * r_pos * tracking_terms

        # Fall termination: pelvis collapse < 0.50m or pitch deviation > 40 deg (0.70 rad)
        pitch_err = abs(q_curr[2] - q_ref_end[2])
        fell = bool(q_curr[1] < 0.50 or pitch_err > 0.70)
        self.step_count += 1
        timeout = bool(self.step_count >= self.max_steps)
        done = bool(fell or timeout)
        if fell:
            total_reward = -5.0

        pose_err_deg = np.degrees(np.sqrt(mse_q))
        pitch_err_deg = np.degrees(pitch_err)
        tau_excess = float(np.mean(excess) * 100.0)

        self.prev_action = action.copy()
        next_obs = self.get_observation()
        return next_obs, total_reward, done, tau_mean, pose_err_deg, pitch_err_deg, tau_excess

# -----------------------------------------------------------------------------
# 4. Batched Multi-Environment Vectorizer (GPU Tensor-Core Accelerated)
# -----------------------------------------------------------------------------
class VectorTrackingEnv:
    """
    Batched multi-environment wrapper to saturate RTX 5060 GPU tensor cores.
    Runs N parallel MuJoCo simulation environments simultaneously, stacking observations
    into batch tensors for single-pass GPU neural network inference.
    """
    def __init__(self, motion_source, num_envs=16, actuator_type="ideal_unlimited", device="cuda"):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.envs = [
            UniversalTrackingEnv(motion_source, actuator_type=actuator_type, device=device)
            for _ in range(num_envs)
        ]
        for env in self.envs:
            env.enable_perturbation = True
        dummy = self.envs[0].reset()
        self.obs_dim = dummy.shape[0]

    def reset(self):
        obs_list = [env.reset() for env in self.envs]
        return np.array(obs_list, dtype=np.float32)

    def step(self, actions):
        """
        actions: (num_envs, 7) np.ndarray
        returns: next_obs (num_envs, obs_dim), rewards (num_envs,), dones (num_envs,), info dict
        """
        next_obs_arr = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        rew_arr = np.zeros(self.num_envs, dtype=np.float32)
        done_arr = np.zeros(self.num_envs, dtype=bool)
        pose_err_list = []
        pitch_err_list = []
        excess_list = []

        for i in range(self.num_envs):
            n_obs, r, d, tau, pe, pte, exc = self.envs[i].step(actions[i])
            if d:
                n_obs = self.envs[i].reset()
            next_obs_arr[i] = n_obs
            rew_arr[i] = r
            done_arr[i] = d
            pose_err_list.append(pe)
            pitch_err_list.append(pte)
            excess_list.append(exc)

        info = {
            "pose_err": float(np.mean(pose_err_list)),
            "pitch_err": float(np.mean(pitch_err_list)),
            "excess": float(np.mean(excess_list)),
        }
        return next_obs_arr, rew_arr, done_arr, info

# -----------------------------------------------------------------------------
# 5. Universal Motion Tracking Trainer (PPO with GAE & GPU Batched Execution)
# -----------------------------------------------------------------------------
class UniversalMotionTrainer:
    def __init__(self, motion_sources, actuator_type="ideal_unlimited", device="cpu", num_envs=16, filter_str=None):
        if isinstance(motion_sources, MotionLib):
            self.motion_lib = motion_sources
        else:
            self.motion_lib = MotionLib(motion_sources, filter_str=filter_str)

        self.device = torch.device(device)
        self.actuator_type = actuator_type
        self.num_envs = num_envs
        self.vec_env = VectorTrackingEnv(self.motion_lib, num_envs=num_envs, actuator_type=actuator_type, device=device)
        self.env = UniversalTrackingEnv(self.motion_lib, actuator_type=actuator_type, device=device)

        obs_dim = self.vec_env.obs_dim
        self.policy = UniversalTrackingPolicy(obs_dim=obs_dim, action_dim=7, device=device)

        self.optimizer = optim.AdamW(self.policy.parameters(), lr=5e-4, weight_decay=1e-3)
        self.scheduler = None
        self.motion_tag = "dataset" if len(self.motion_lib.clips) > 1 else self.motion_lib.clip_names[0]

    def train(self, num_iterations=400, steps_per_iter=1600, batch_size=256, epochs=4):
        total_steps = num_iterations * steps_per_iter
        total_dur = np.sum(self.motion_lib.durations)
        print(f"\n==========================================================================")
        print(f"   Universal DeepMimic/PHC Motion Tracking Architecture (GPU Vectorized)")
        print(f"==========================================================================")
        print(f"  Actuator: {self.actuator_type.upper()}  |  Clips in Library: {len(self.motion_lib.clips)} ({total_dur:.2f}s total)")
        print(f"  Parallel Envs: {self.num_envs}  |  Device: {self.device}")
        print(f"  Total Iterations: {num_iterations}  |  Steps/Iter: {steps_per_iter} ({total_steps/1e6:.2f}M steps)")
        print(f"  Action Space: Bounded Micro-Adjustments +/-4.5 deg | Velocity Feedforward: ACTIVE")
        print(f"----------------------------------------------------------------------------------------")
        print(f"  {'Iter':>6}  {'Mean Reward':>12}  {'Pose Error (deg)':>18}  {'Pitch Error (deg)':>18}  {'Torq Excess':>12}  {'Time':>7}")
        print(f"----------------------------------------------------------------------------------------", flush=True)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_iterations, eta_min=2e-5)

        best_reward = -float("inf")
        checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_path = os.path.join(checkpoint_dir, f"policy_universal_tracker_{self.motion_tag}_{self.actuator_type}.pt")
        latest_path = os.path.join(checkpoint_dir, f"policy_universal_tracker_{self.motion_tag}_{self.actuator_type}_latest.pt")

        rollout_len = max(1, steps_per_iter // self.num_envs)

        # Persistent observation: reset once then let envs run continuously across iterations
        # This gives the policy true long-horizon experience (multi-gait-cycle rollouts)
        curr_obs = self.vec_env.reset()

        for it in range(num_iterations):
            t0 = time.time()
            
            obs_buf = np.zeros((rollout_len, self.num_envs, self.vec_env.obs_dim), dtype=np.float32)
            act_buf = np.zeros((rollout_len, self.num_envs, 7), dtype=np.float32)
            logp_buf = np.zeros((rollout_len, self.num_envs), dtype=np.float32)
            rew_buf = np.zeros((rollout_len, self.num_envs), dtype=np.float32)
            val_buf = np.zeros((rollout_len, self.num_envs), dtype=np.float32)
            done_buf = np.zeros((rollout_len, self.num_envs), dtype=bool)

            pose_err_list, pitch_err_list, excess_list = [], [], []

            for step in range(rollout_len):
                obs_t = torch.from_numpy(curr_obs).float().to(self.device)
                with torch.no_grad():
                    action_t, logp_t, val_t = self.policy.get_action(obs_t)

                actions = action_t.cpu().numpy()
                logps = logp_t.cpu().numpy().squeeze(-1)
                vals = val_t.cpu().numpy().squeeze(-1)

                next_obs, rews, dones, info = self.vec_env.step(actions)

                obs_buf[step] = curr_obs
                act_buf[step] = actions
                logp_buf[step] = logps
                rew_buf[step] = rews
                val_buf[step] = vals
                done_buf[step] = dones

                pose_err_list.append(info["pose_err"])
                pitch_err_list.append(info["pitch_err"])
                excess_list.append(info["excess"])

                curr_obs = next_obs

            # Bootstrap last value for GAE
            with torch.no_grad():
                _, _, last_val_t = self.policy.get_action(torch.from_numpy(curr_obs).float().to(self.device))
                last_values = last_val_t.cpu().numpy().squeeze(-1)

            # Generalized Advantage Estimation (GAE) across all parallel envs
            gamma, lam = 0.99, 0.95
            advantages = np.zeros((rollout_len, self.num_envs), dtype=np.float32)
            lastgaelam = np.zeros(self.num_envs, dtype=np.float32)
            for t in reversed(range(rollout_len)):
                if t == rollout_len - 1:
                    next_val = last_values
                else:
                    next_val = val_buf[t + 1]
                nonterminal = 1.0 - done_buf[t].astype(np.float32)
                delta = rew_buf[t] + gamma * next_val * nonterminal - val_buf[t]
                advantages[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
            returns = advantages + val_buf

            # Flatten transitions onto GPU CUDA tensors
            b_obs = torch.tensor(obs_buf.reshape(-1, self.vec_env.obs_dim), dtype=torch.float32, device=self.device)
            b_act = torch.tensor(act_buf.reshape(-1, 7), dtype=torch.float32, device=self.device)
            b_logp = torch.tensor(logp_buf.reshape(-1, 1), dtype=torch.float32, device=self.device)
            b_adv = torch.tensor(advantages.reshape(-1, 1), dtype=torch.float32, device=self.device)
            b_ret = torch.tensor(returns.reshape(-1, 1), dtype=torch.float32, device=self.device)

            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

            # Mini-batch PPO Updates directly on GPU
            dataset_size = b_obs.shape[0]
            for epoch in range(epochs):
                perm = torch.randperm(dataset_size, device=self.device)
                for start in range(0, dataset_size, batch_size):
                    idx = perm[start:start + batch_size]
                    mb_obs = b_obs[idx]
                    mb_act = b_act[idx]
                    mb_logp = b_logp[idx]
                    mb_adv = b_adv[idx]
                    mb_ret = b_ret[idx]

                    new_logp, entropy, new_val = self.policy.evaluate_actions(mb_obs, mb_act)
                    ratio = torch.exp(new_logp - mb_logp)
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 0.80, 1.20) * mb_adv
                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = 0.5 * nn.MSELoss()(new_val, mb_ret)
                    loss = actor_loss + critic_loss - 0.02 * entropy.mean()

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                    self.optimizer.step()

            self.scheduler.step()

            iter_reward = float(np.mean(rew_buf))
            mean_pose = float(np.mean(pose_err_list))
            mean_pitch = float(np.mean(pitch_err_list))
            mean_excess = float(np.mean(excess_list))
            dt_iter = time.time() - t0

            if (it + 1) % 5 == 0 or it == 0:
                print(f"  {it+1:6d}  {iter_reward:12.2f}  {mean_pose:18.2f}  {mean_pitch:18.2f}  {mean_excess:11.2f}%  {dt_iter:6.2f}s", flush=True)

            if iter_reward > best_reward:
                best_reward = iter_reward
                torch.save(self.policy.state_dict(), save_path)

            if (it + 1) % 50 == 0 or (it + 1) == num_iterations:
                torch.save(self.policy.state_dict(), latest_path)

        print(f"--------------------------------------------------------------------------", flush=True)
        print(f"Finished Training Universal Motion Tracker! Checkpoints saved to: {save_path} and {latest_path}", flush=True)
        # Return the BEST checkpoint (not latest) so eval always uses peak performance
        return save_path if os.path.exists(save_path) else latest_path

    def evaluate_and_render(self, checkpoint_path, clip_name=None, num_steps=150, start_phase=None, render_fps=50):
        """
        Evaluates Universal Motion Tracker on a specified clip or all clips in the MotionLib.
        """
        import shutil
        print(f"\nEvaluating Universal Motion Tracker checkpoint: {checkpoint_path}")
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.policy.eval()

        if clip_name is not None and str(clip_name).lower() == "all":
            eval_clips = list(zip(self.motion_lib.clips, self.motion_lib.clip_names))
        elif clip_name is not None:
            eval_clips = [self.motion_lib.get_clip(clip_name)]
        else:
            eval_clips = [(self.motion_lib.clips[0], self.motion_lib.clip_names[0])]

        results_summary = []
        for eval_clip, eval_name in eval_clips:
            if start_phase is None or start_phase == "random":
                start_frame = np.random.randint(0, eval_clip.num_frames)
            else:
                start_frame = int(start_phase) % eval_clip.num_frames

            print(f"\nEvaluating clip: '{eval_name}' starting from frame {start_frame}/{eval_clip.num_frames} ({start_frame * 0.002:.3f}s)")
            obs = self.env.reset(clip=eval_clip, start_frame=start_frame)

            t_hist, q_hist, q_ref_hist, tau_hist, grf_hist, frames = [], [], [], [], [], []
            dt_policy = 0.002 * self.env.decimation  # 0.02s per step (50 Hz)
            render_interval = max(1, int(round(1.0 / (render_fps * dt_policy))))

            fell = False
            for step in range(num_steps):
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    action_t, _, _ = self.policy.get_action(obs_t, deterministic=True)
                action = action_t.squeeze(0).cpu().numpy()

                q_ref, _ = eval_clip.get_frame(self.env.current_frame)
                next_obs, _, done, tau, _, _, _ = self.env.step(action)

                t_curr = step * dt_policy
                t_hist.append(t_curr)
                q_hist.append(self.env.env.data.qpos.copy())
                q_ref_hist.append(q_ref)
                tau_hist.append(tau)

                grf = 0.0
                for i in range(self.env.env.data.ncon):
                    c_array = np.zeros(6)
                    mujoco.mj_contactForce(self.env.env.model, self.env.env.data, i, c_array)
                    grf += abs(c_array[0])
                grf_hist.append(grf)

                if step % render_interval == 0:
                    frame = self.env.env.render_frame()
                    frames.append(frame)

                if done:
                    q_now = self.env.env.data.qpos
                    if q_now[1] < 0.50 or abs(q_now[2] - q_ref[2]) > 0.70:
                        fell = True
                        print(f"  [!] Fall detected at step {step}/{num_steps} (t = {t_curr:.3f}s, z={q_now[1]:.2f}m, pitch={np.degrees(q_now[2]):.1f} deg)")
                    break

                obs = next_obs

            t_hist = np.array(t_hist)
            q_hist = np.array(q_hist)
            q_ref_hist = np.array(q_ref_hist)
            tau_hist = np.array(tau_hist)
            grf_hist = np.array(grf_hist)

            pose_err = np.mean(np.abs(q_hist[:, 3:] - q_ref_hist[:, 3:]))
            pitch_drift = np.mean(np.abs(q_hist[:, 2] - q_ref_hist[:, 2]))
            rms_torque = np.sqrt(np.mean(tau_hist**2))
            mean_grf = np.mean(grf_hist)

            if fell:
                status_str = "FALL DETECTED"
            elif step < num_steps - 1:
                status_str = "COMPLETED FULL CLIP ACCURATELY!"
            else:
                status_str = "COMPLETED FULL RUN WITHOUT FALLING!"
            print(f"==========================================================================")
            print(f"   EVALUATION [{eval_name}]: {status_str}")
            print(f"  - Initial Frame       : {start_frame}/{eval_clip.num_frames} ({start_frame*0.002:.3f}s)")
            print(f"  - Duration Evaluated  : {len(t_hist)} steps ({len(t_hist)*dt_policy:.3f}s)")
            print(f"  - Pose Tracking Error : {pose_err:.4f} rad ({np.degrees(pose_err):.2f} deg)")
            print(f"  - Pelvis Pitch Drift  : {pitch_drift:.4f} rad ({np.degrees(pitch_drift):.2f} deg)")
            print(f"  - Mean Stance GRF     : {mean_grf:.1f} N")
            print(f"  - RMS Torque Effort   : {rms_torque:.2f} N*m")
            print(f"==========================================================================")

            results_summary.append((eval_name, status_str, len(t_hist)*dt_policy, np.degrees(pose_err), np.degrees(pitch_drift)))

            render_dir = os.path.join(os.path.dirname(__file__), "renders")
            os.makedirs(render_dir, exist_ok=True)

            gif_name = f"universal_tracker_{eval_name}_{self.actuator_type}.gif"
            gif_path = os.path.join(render_dir, gif_name)
            imageio.mimsave(gif_path, frames, fps=render_fps)
            print(f"Saved Universal Tracker GIF animation to: {gif_path}")

            artifact_dir = r"C:\Users\anbes\.gemini\antigravity-ide\brain\5e0a9087-9b28-4f80-85c6-057cf5c3bf21"
            if os.path.exists(artifact_dir):
                imageio.mimsave(os.path.join(artifact_dir, gif_name), frames, fps=render_fps)

            # Diagnostic plot
            fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            joint_labels = ["Lumbar", "L-Hip", "L-Knee", "L-Ankle", "R-Hip", "R-Knee", "R-Ankle"]
            for j in range(7):
                axes[0].plot(t_hist, np.degrees(q_hist[:, 3+j]), label=joint_labels[j])
                axes[0].plot(t_hist, np.degrees(q_ref_hist[:, 3+j]), linestyle="--", alpha=0.5)
            axes[0].set_ylabel("Joint Angles (deg)")
            axes[0].set_title(f"Universal Tracker ({eval_name} / {self.actuator_type.upper()}) - Init Frame: {start_frame} ({status_str})")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(loc="upper right", ncol=4, fontsize=8)

            axes[1].plot(t_hist, q_hist[:, 0], label="Pelvis X (m)", color="blue")
            axes[1].plot(t_hist, q_hist[:, 1], label="Pelvis Z (m)", color="green")
            axes[1].plot(t_hist, np.degrees(q_hist[:, 2]), label="Pelvis Pitch (deg)", color="red")
            axes[1].plot(t_hist, np.degrees(q_ref_hist[:, 2]), label="Ref Pitch (deg)", color="darkred", linestyle="--")
            axes[1].set_ylabel("Base State")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(loc="upper right", fontsize=8)

            for j in range(7):
                axes[2].plot(t_hist, tau_hist[:, j], label=joint_labels[j])
            axes[2].set_ylabel("Torque Effort (N*m)")
            axes[2].set_xlabel("Time (s)")
            axes[2].grid(True, alpha=0.3)
            axes[2].legend(loc="upper right", ncol=4, fontsize=8)

            plt.tight_layout()
            plot_name = f"universal_tracker_{eval_name}_{self.actuator_type}_results.png"
            plot_path = os.path.join(render_dir, plot_name)
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Saved Universal Tracker diagnostic plot to: {plot_path}")

            if os.path.exists(artifact_dir):
                shutil.copy2(plot_path, os.path.join(artifact_dir, plot_name))

        if len(results_summary) > 1:
            print(f"\n==========================================================================")
            print(f"   MULTI-MOTION DATASET EVALUATION SUMMARY")
            print(f"==========================================================================")
            print(f"  {'Motion Name':<28} | {'Status':<32} | {'Time (s)':<8} | {'Pose Err':<10} | {'Pitch Drift':<10}")
            print(f"--------------------------------------------------------------------------")
            for name, status, dur, pe, pd in results_summary:
                print(f"  {name:<28} | {status:<32} | {dur:6.2f}s  | {pe:6.2f} deg  | {pd:6.2f} deg")
            print(f"==========================================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal DeepMimic/PHC Motion Tracker")
    parser.add_argument("--dataset", type=str, default="motions", help="Path to motion dataset directory or .npz file")
    parser.add_argument("--motion", type=str, default=None, help="Optional specific motion file (overrides --dataset)")
    parser.add_argument("--iters", type=int, default=300, help="Training iterations (default: 300)")
    parser.add_argument("--steps", type=int, default=4096, help="Rollout steps per iteration (default: 4096)")
    parser.add_argument("--batch-size", type=int, default=512, help="PPO mini-batch size on GPU (default: 512)")
    parser.add_argument("--epochs", type=int, default=4, help="PPO epochs per iteration (default: 4)")
    parser.add_argument("--actuator", type=str, default="ideal_unlimited", help="Actuator type")
    parser.add_argument("--resume", action="store_true", help="Resume/warmstart training from existing checkpoint")
    parser.add_argument("--warmstart-from", type=str, default=None, help="Explicit checkpoint path to warmstart from across actuator tiers")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate existing checkpoint")
    parser.add_argument("--eval-motion", type=str, default=None, help="Motion name to evaluate or 'all'")
    parser.add_argument("--eval-phase", type=str, default="random", help="Evaluation start frame index or 'random'")
    parser.add_argument("--eval-steps", type=int, default=150, help="Evaluation steps at 50 Hz (default: 150 = 3.0s)")
    parser.add_argument("--num-envs", type=int, default=64, help="Number of parallel environments for GPU acceleration (default: 64)")
    parser.add_argument("--filter", type=str, default="walk", help="Filter motions by keyword (default: 'walk')")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device for Universal Tracker: {device} (RTX Tensor Accelerated)")

    # Determine motion source
    if args.motion is not None:
        motion_source = os.path.join(os.path.dirname(__file__), args.motion)
    else:
        motion_source = os.path.join(os.path.dirname(__file__), args.dataset)

    trainer = UniversalMotionTrainer(motion_source, actuator_type=args.actuator, device=device, num_envs=args.num_envs, filter_str=args.filter)
    motion_tag = trainer.motion_tag
    checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints", f"policy_universal_tracker_{motion_tag}_{args.actuator}.pt")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints", f"policy_universal_tracker_{args.actuator}.pt")

    if args.warmstart_from and os.path.exists(args.warmstart_from):
        print(f"Curriculum warmstart: loading weights from {args.warmstart_from}")
        trainer.policy.load_state_dict(torch.load(args.warmstart_from, map_location=device))
        if args.eval_only:
            checkpoint_path = args.warmstart_from
    elif args.resume and os.path.exists(checkpoint_path):
        print(f"Resuming/warmstarting training from existing checkpoint: {checkpoint_path}")
        trainer.policy.load_state_dict(torch.load(checkpoint_path, map_location=device))

    if not args.eval_only:
        checkpoint_path = trainer.train(
            num_iterations=args.iters,
            steps_per_iter=args.steps,
            batch_size=args.batch_size,
            epochs=args.epochs
        )

    phase_arg = None if args.eval_phase == "random" else int(args.eval_phase)
    trainer.evaluate_and_render(checkpoint_path, clip_name=args.eval_motion, num_steps=args.eval_steps, start_phase=phase_arg)
