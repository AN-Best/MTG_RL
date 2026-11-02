"""
Stable Motion Tracker: Mode 1 (Ideal Unlimited PD) & Mode 2 (Curriculum MTG)
=============================================================================
- Mode 1 ('ideal_unlimited'): Pure ideal PD tracking. Zero MTG calculations or mentions.
- Mode 2 ('ideal_limited'): Two-phase curriculum:
    Phase 1: 'soft' MTG penalty in reward (shapes policy toward biological feasibility)
    Phase 2: 'hard' instantaneous physical clamping in the actuator at every 0.002s sub-step.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mujoco

sys.path.insert(0, os.path.dirname(__file__))
from Biped_Env import BipedEnv

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------------------
# Reference Motion Library
# -----------------------------------------------------------------------------
class MotionClip:
    def __init__(self, t, q, u):
        self.t = t
        self.q = q
        self.u = u
        self.dt = float(t[1] - t[0]) if len(t) > 1 else 0.002
        self.num_frames = len(t)
        self.duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0
        self.dx_per_cycle = float(self.q[-1, 0] - self.q[0, 0])
        if self.num_frames > 1:
            self.dx_per_cycle += float(self.q[1, 0] - self.q[0, 0])

    @classmethod
    def load(cls, npz_path):
        data = np.load(npz_path)
        return cls(t=data["t"], q=data["q"], u=data["u"])

    def get_frame(self, frame_idx):
        n_cycles = int(frame_idx) // self.num_frames
        idx = int(frame_idx) % self.num_frames
        q = self.q[idx].copy()
        u = self.u[idx].copy()
        q[0] += n_cycles * self.dx_per_cycle
        return q, u

    def sample_random_frame(self):
        idx = np.random.randint(0, self.num_frames)
        return idx, self.q[idx].copy(), self.u[idx].copy()

# -----------------------------------------------------------------------------
# Gated Residual Architecture and Gaussian Actor-Critic
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

class StableTrackingPolicy(nn.Module):
    def __init__(self, obs_dim=64, action_dim=7, hidden_dim=256, device="cuda"):
        super().__init__()
        self.device = torch.device(device)

        # Actor
        self.actor_in = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.SiLU())
        self.actor_res1 = GatedResidualBlock(hidden_dim)
        self.actor_res2 = GatedResidualBlock(hidden_dim)
        self.actor_out = nn.Linear(hidden_dim, action_dim)
        nn.init.orthogonal_(self.actor_out.weight, gain=0.02)
        nn.init.zeros_(self.actor_out.bias)

        # Standard deviation parameter: start with small exploration ~0.10 rad (~5.7 deg)
        self.log_std = nn.Parameter(torch.zeros(action_dim, device=self.device) - 2.2)

        # Corrective action authority (rad): Lumbar 0.25, Hips 0.30, Knees 0.25, Ankles 0.35
        self.register_buffer(
            "action_scale",
            torch.tensor([0.25, 0.30, 0.25, 0.35, 0.30, 0.25, 0.35], dtype=torch.float32, device=self.device)
        )

        # Critic
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
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        elif obs.device != self.device:
            obs = obs.to(self.device)

        mean, std, value = self.forward(obs)
        if deterministic:
            return mean, torch.zeros_like(mean), value
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob, value

    def evaluate_actions(self, obs, action):
        mean, std, value = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy, value

# -----------------------------------------------------------------------------
# Tracking Environment: Clean Separation Between Mode 1 and Mode 2
# -----------------------------------------------------------------------------
class StableTrackingEnv:
    def __init__(self, clip, mode="mode1", mtg_phase="hard", decimation=10, max_steps=300,
                 enable_perturbations=False, pert_prob=0.06, pert_force_range=(-150.0, 150.0)):
        """
        mode:
          - "mode1" (or "ideal_unlimited"): Pure ideal PD controller. Zero MTG mention or calculation.
          - "mode2" (or "ideal_limited"): MTG-constrained tracking.
              mtg_phase="soft": Curriculum Phase 1 (unlimited physical actuator + soft MTG penalty in reward)
              mtg_phase="hard": Curriculum Phase 2 (physical hard clamping in actuator at every sub-step)
        """
        self.clip = clip
        self.mode = "mode1" if mode in ["mode1", "ideal_unlimited", "unlimited"] else "mode2"
        self.mtg_phase = mtg_phase
        self.decimation = decimation
        self.max_steps = max_steps
        self.enable_perturbations = enable_perturbations
        self.pert_prob = pert_prob
        self.pert_force_range = pert_force_range
        self.pert_remaining_steps = 0
        self.pert_force = 0.0

        # Select underlying physical actuator
        if self.mode == "mode1":
            actuator_type = "ideal_unlimited"
        else:
            # Mode 2: In soft phase we use unconstrained physics; in hard phase we use mtg_clamped physics
            actuator_type = "ideal_unlimited" if self.mtg_phase == "soft" else "mtg_clamped"

        self.env = BipedEnv(actuator_type=actuator_type, device="cpu")
        self.d_site = mujoco.MjData(self.env.model)

        self.torso_body_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.pelvis_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, "pelvis_geom")
        self.sole_l_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, "foot_l_sole")
        self.sole_r_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, "foot_r_sole")
        self.trunk_tip_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_SITE, "trunk_tip")

        self.prev_action = np.zeros(7, dtype=np.float32)
        self.filtered_action = np.zeros(7, dtype=np.float32)
        self.step_count = 0
        self.current_frame = 0

        # MTG Parameters instantiated ONLY for Mode 2
        if self.mode == "mode2":
            self.T_max_pos = np.array([300.0, 400.0, 500.0, 100.0, 400.0, 500.0, 100.0], dtype=np.float32)
            self.T_max_neg = np.array([400.0, 300.0, 200.0, 380.0, 300.0, 200.0, 380.0], dtype=np.float32)
            self.q_opt = np.array([0.0, -0.40, -0.90, 0.35, -0.40, -0.90, 0.35], dtype=np.float32)
            self.w = np.array([1.20, 1.20, 1.20, 1.10, 1.20, 1.20, 1.10], dtype=np.float32)
            self.u_max_pos = np.array([12.0, 15.0, 18.0, 16.0, 15.0, 18.0, 16.0], dtype=np.float32)
            self.u_max_neg = np.array([12.0, 15.0, 18.0, 22.0, 15.0, 18.0, 22.0], dtype=np.float32)
            self.k1_pass = np.array([250.0, 200.0, 300.0, 150.0, 200.0, 300.0, 150.0], dtype=np.float32)
            self.k2_pass = np.array([250.0, 200.0, 300.0, 150.0, 200.0, 300.0, 150.0], dtype=np.float32)
            self.q1_pass = np.array([-0.40, -0.40, -1.80, -0.40, -0.40, -1.80, -0.40], dtype=np.float32)
            self.q2_pass = np.array([0.50, 1.20, 0.05, 0.40, 1.20, 0.05, 0.40], dtype=np.float32)
            self.tau_act = 0.020
            self.tau_deact = 0.040
            self.a_min = 0.01
            self.a_init = 0.35
            self.a_soft = np.full(7, self.a_init, dtype=np.float32)

    def set_mtg_phase(self, phase):
        """Switches Mode 2 between 'soft' reward shaping and 'hard' physical clamping."""
        if self.mode == "mode2":
            self.mtg_phase = phase
            new_actuator = "ideal_unlimited" if phase == "soft" else "mtg_clamped"
            from Biped_Actuators import create_actuator
            self.env.actuator = create_actuator(new_actuator, device=self.env.device)
            self.env.actuator_type = new_actuator

    def reset(self, start_frame=None):
        if start_frame is None:
            self.current_frame, q0, u0 = self.clip.sample_random_frame()
        else:
            self.current_frame = int(start_frame) % self.clip.num_frames
            q0, u0 = self.clip.get_frame(self.current_frame)

        self.pert_remaining_steps = 0
        self.pert_force = 0.0
        self.env.data.xfrc_applied[:] = 0.0

        self.env.reset(qpos_init=q0.copy(), qvel_init=u0.copy())
        if self.mode == "mode2":
            self.a_soft = np.full(7, self.a_init, dtype=np.float32)
        self.prev_action = np.zeros(7, dtype=np.float32)
        self.filtered_action = np.zeros(7, dtype=np.float32)
        self.step_count = 0
        return self.get_obs()

    def get_obs(self):
        q_curr = self.env.data.qpos
        u_curr = self.env.data.qvel

        q_ref_0, u_ref_0 = self.clip.get_frame(self.current_frame)
        q_ref_10, _ = self.clip.get_frame(self.current_frame + 10)
        q_ref_30, _ = self.clip.get_frame(self.current_frame + 30)

        q_joints = q_curr[3:10].astype(np.float32)
        u_joints = u_curr[3:10].astype(np.float32)
        err_q0 = (q_ref_0[3:10] - q_joints)
        err_u0 = (u_ref_0[3:10] - u_joints)
        err_q10 = (q_ref_10[3:10] - q_joints)
        err_q30 = (q_ref_30[3:10] - q_joints)

        root_err = np.array([
            q_ref_0[1] - q_curr[1],
            q_ref_0[2] - q_curr[2],
            u_ref_0[0] - u_curr[0],
            u_ref_0[1] - u_curr[1],
            u_ref_0[2] - u_curr[2]
        ], dtype=np.float32)

        pos_pelvis = self.env.data.geom_xpos[self.pelvis_id]
        pos_l = self.env.data.geom_xpos[self.sole_l_id]
        pos_r = self.env.data.geom_xpos[self.sole_r_id]
        pos_t = self.env.data.site_xpos[self.trunk_tip_id]
        rel_ee_pos = np.array([
            pos_l[0] - pos_pelvis[0], pos_l[2] - pos_pelvis[2],
            pos_r[0] - pos_pelvis[0], pos_r[2] - pos_pelvis[2],
            pos_t[0] - pos_pelvis[0], pos_t[2] - pos_pelvis[2]
        ], dtype=np.float32)

        fl, fr = 0.0, 0.0
        d = self.env.data
        for i in range(d.ncon):
            c_array = np.zeros(6)
            mujoco.mj_contactForce(self.env.model, d, i, c_array)
            normal_f = abs(c_array[0])
            g1, g2 = d.contact[i].geom1, d.contact[i].geom2
            name1 = mujoco.mj_id2name(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, g1) or ""
            name2 = mujoco.mj_id2name(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, g2) or ""
            if "foot_l_sole" in (name1, name2): fl += normal_f
            if "foot_r_sole" in (name1, name2): fr += normal_f
        contacts = np.array([min(fl, 1500.0) / 600.0, min(fr, 1500.0) / 600.0], dtype=np.float32)

        phi = (self.current_frame % self.clip.num_frames) / float(max(1, self.clip.num_frames))
        phase_feat = np.array([np.sin(2.0 * np.pi * phi), np.cos(2.0 * np.pi * phi)], dtype=np.float32)

        return np.concatenate([
            q_joints, u_joints, err_q0, err_u0, err_q10, err_q30, root_err, rel_ee_pos, contacts, self.filtered_action, phase_feat
        ], dtype=np.float32)

    def step(self, action):
        # 1. Optional Perturbation Impulse
        if self.enable_perturbations:
            if self.pert_remaining_steps > 0:
                self.env.data.xfrc_applied[self.torso_body_id, 0] = self.pert_force
                self.pert_remaining_steps -= 1
            else:
                self.env.data.xfrc_applied[self.torso_body_id, 0] = 0.0
                if np.random.rand() < self.pert_prob:
                    self.pert_force = np.random.uniform(self.pert_force_range[0], self.pert_force_range[1])
                    self.pert_remaining_steps = np.random.randint(3, 7)

        prev_filtered = self.filtered_action.copy()
        filtered = 0.40 * self.filtered_action + 0.60 * action
        self.filtered_action = filtered.copy()

        tau_acc = np.zeros(7, dtype=np.float32)
        T_pos_acc = np.zeros(7, dtype=np.float32)
        T_neg_acc = np.zeros(7, dtype=np.float32)
        excess_norm_acc = np.zeros(7, dtype=np.float32)

        for sub_step in range(self.decimation):
            q_ref_k, u_ref_k = self.clip.get_frame(self.current_frame + sub_step)
            alpha = (sub_step + 1) / float(self.decimation)
            act_k = (1.0 - alpha) * prev_filtered + alpha * filtered

            q_des = q_ref_k[3:10] + act_k
            u_des = u_ref_k[3:10]

            if self.mode == "mode2" and self.mtg_phase == "soft":
                q_jk = self.env.data.qpos[3:10].astype(np.float32)
                u_jk = self.env.data.qvel[3:10].astype(np.float32)
                f_Lk = np.exp(-((q_jk - self.q_opt) / self.w)**2)
                nu_pos = np.clip(u_jk / self.u_max_pos, -1.0, 1.0)
                nu_neg = np.clip(-u_jk / self.u_max_neg, -1.0, 1.0)
                f_Vp = np.where(nu_pos >= 0.0,
                                np.clip((1.0 - nu_pos) / (1.0 + 4.0 * nu_pos + 1e-6), 0.0, 1.0),
                                np.clip(1.0 - 1.4 * nu_pos / (1.0 - 4.0 * nu_pos + 1e-6), 1.0, 1.6))
                f_Vn = np.where(nu_neg >= 0.0,
                                np.clip((1.0 - nu_neg) / (1.0 + 4.0 * nu_neg + 1e-6), 0.0, 1.0),
                                np.clip(1.0 - 1.4 * nu_neg / (1.0 - 4.0 * nu_neg + 1e-6), 1.0, 1.6))
                T_LV_p = self.T_max_pos * f_Lk * f_Vp
                T_LV_n = self.T_max_neg * f_Lk * f_Vn
                tau_pass_k = (
                    self.k1_pass * np.maximum(0.0, self.q1_pass - q_jk)**2
                    - self.k2_pass * np.maximum(0.0, q_jk - self.q2_pass)**2
                )

            _, tau = self.env.step_simulation(q_des, dt_sim=0.002, u_des=u_des)
            tau_acc += tau

            if self.mode == "mode2":
                if self.mtg_phase == "soft":
                    tau_act_des = tau - tau_pass_k
                    T_LV_dir = np.where(tau_act_des >= 0.0, T_LV_p, T_LV_n)
                    u_req = np.clip(np.abs(tau_act_des) / (T_LV_dir + 1e-3), 0.0, 1.0)
                    tau_c = np.where(u_req >= self.a_soft, self.tau_act, self.tau_deact)
                    self.a_soft = np.clip(self.a_soft + 0.002 * (u_req - self.a_soft) / tau_c, self.a_min, 1.0)
                    Tp_k = self.a_soft * T_LV_p + tau_pass_k
                    Tn_k = self.a_soft * T_LV_n - tau_pass_k
                else:
                    Tp_k = self.env.actuator.last_T_pos
                    Tn_k = self.env.actuator.last_T_neg
                T_pos_acc += Tp_k
                T_neg_acc += Tn_k
                if self.mtg_phase == "soft":
                    ex_p = np.maximum(0.0, tau - Tp_k) / self.T_max_pos
                    ex_n = np.maximum(0.0, -tau - Tn_k) / self.T_max_neg
                    excess_norm_acc += np.where(tau - tau_pass_k >= 0.0, ex_p, ex_n)

        self.current_frame += self.decimation
        tau_mean = tau_acc / self.decimation

        q_curr = self.env.data.qpos
        u_curr = self.env.data.qvel
        q_ref_end, u_ref_end = self.clip.get_frame(self.current_frame)

        # 2. Tracking Rewards (Shared by both Mode 1 and Mode 2)
        mse_q = np.mean((q_curr[3:10] - q_ref_end[3:10])**2)
        mse_u = np.mean((u_curr[3:10] - u_ref_end[3:10])**2)
        r_q = np.exp(-30.0 * mse_q)
        r_u = np.exp(-0.2 * mse_u)

        p_err_sq = (q_curr[2] - q_ref_end[2])**2
        vp_err_sq = (u_curr[2] - u_ref_end[2])**2
        z_err_sq = (q_curr[1] - q_ref_end[1])**2
        x_err_sq = (q_curr[0] - q_ref_end[0])**2
        vx_err_sq = (u_curr[0] - u_ref_end[0])**2
        
        r_rot = np.exp(-25.0 * p_err_sq - 1.0 * vp_err_sq)
        r_pos_z = np.exp(-20.0 * z_err_sq)
        r_pos_x = np.exp(-5.0 * x_err_sq)
        r_vel = np.exp(-10.0 * vx_err_sq)

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
        r_cart = np.exp(-15.0 * cart_mse)

        p_act = -0.01 * np.sum((action - self.prev_action)**2)
        alive_bonus = 0.10

        # 3. Mode-Specific Constraints
        info = {
            "pose_err": float(np.degrees(np.sqrt(mse_q))),
            "pitch_err": float(np.degrees(abs(q_curr[2] - q_ref_end[2]))),
            "tau": tau_mean.copy()
        }

        if self.mode == "mode1":
            # Mode 1: Pure PD tracking. Zero MTG computation or penalty.
            reward = (
                alive_bonus
                + 0.30 * r_q
                + 0.25 * r_rot
                + 0.15 * r_vel
                + 0.10 * r_pos_x
                + 0.10 * r_pos_z
                + 0.10 * r_cart
                + p_act
            )
            info["mtg_excess"] = 0.0
        else:
            # Mode 2: Sub-step synchronized biological MTG capacities & excess
            T_avail_pos = T_pos_acc / self.decimation
            T_avail_neg = T_neg_acc / self.decimation
            mtg_excess = excess_norm_acc / self.decimation
            
            # In Phase 1 ('soft'), penalize excess in reward gracefully bounded
            if self.mtg_phase == "soft":
                p_mtg_excess = -0.50 * float(np.sum(mtg_excess**2))
            else:
                p_mtg_excess = 0.0

            reward = (
                alive_bonus
                + 0.30 * r_q
                + 0.25 * r_rot
                + 0.15 * r_vel
                + 0.10 * r_pos_x
                + 0.10 * r_pos_z
                + 0.10 * r_cart
                + p_act
                + p_mtg_excess
            )
            info["mtg_excess"] = float(np.sum(mtg_excess))
            info["T_avail_pos"] = T_avail_pos.copy()
            info["T_avail_neg"] = T_avail_neg.copy()

        # Termination: fall detection
        pitch_err = abs(q_curr[2] - q_ref_end[2])
        fell = bool(q_curr[1] < 0.45 or pitch_err > 0.70)
        self.step_count += 1
        done = bool(fell or self.step_count >= self.max_steps)
        if fell:
            reward = 0.0

        info["fell"] = fell
        self.prev_action = action.copy()
        next_obs = self.get_obs()
        return next_obs, reward, done, info

# -----------------------------------------------------------------------------
# Vectorized Multi-Environment
# -----------------------------------------------------------------------------
class VectorStableEnv:
    def __init__(self, clip, num_envs=64, mode="mode1", mtg_phase="hard", enable_perturbations=False, pert_prob=0.04, pert_force_range=(-120.0, 120.0)):
        self.num_envs = num_envs
        self.envs = [
            StableTrackingEnv(
                clip, mode=mode, mtg_phase=mtg_phase,
                enable_perturbations=enable_perturbations,
                pert_prob=pert_prob, pert_force_range=pert_force_range
            )
            for _ in range(num_envs)
        ]
        dummy = self.envs[0].reset()
        self.obs_dim = dummy.shape[0]

    def set_mtg_phase(self, phase):
        for e in self.envs:
            e.set_mtg_phase(phase)

    def reset(self):
        return np.array([e.reset() for e in self.envs], dtype=np.float32)

    def step(self, actions):
        next_obs_arr = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        rew_arr = np.zeros(self.num_envs, dtype=np.float32)
        done_arr = np.zeros(self.num_envs, dtype=bool)
        pe_list, pte_list, ex_list = [], [], []

        for i in range(self.num_envs):
            n_obs, r, d, info = self.envs[i].step(actions[i])
            if d:
                n_obs = self.envs[i].reset()
            next_obs_arr[i] = n_obs
            rew_arr[i] = r
            done_arr[i] = d
            pe_list.append(info["pose_err"])
            pte_list.append(info["pitch_err"])
            ex_list.append(info.get("mtg_excess", 0.0))

        return next_obs_arr, rew_arr, done_arr, {
            "pose_err": float(np.mean(pe_list)),
            "pitch_err": float(np.mean(pte_list)),
            "mtg_excess": float(np.mean(ex_list))
        }
