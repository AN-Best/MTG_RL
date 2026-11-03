"""
Unified Musculoskeletal Biped Imitation & Motion Generation Suite (500 Hz Verified)
===================================================================================
Runs four experiments comparing:
  - Mode 1 (Ideal Unlimited PD): Unconstrained linear PD joint controller.
  - Mode 2 (MTG Clamped at 500 Hz): Hill-type Muscle Torque Generator with:
      1. Passive joint tissue torque:
           tau_passive(q) = k1 * relu(q1 - q)^2 - k2 * relu(q - q2)^2
      2. Active torque compensation for passive torque:
           tau_active_des = tau_pd - tau_passive(q)
      3. 1st-order muscle activation dynamics a(t) (tau_act=20ms, tau_deact=40ms)
      4. Instantaneous 500 Hz active torque clamping + passive torque application:
           tau_active = clamp(tau_active_des, -a(t)*T_LV_neg(q, u), +a(t)*T_LV_pos(q, u))
           tau_total  = tau_active + tau_passive(q)

Experiments:
  1. Experiment 1: Walking Imitation (~1.25 m/s, Arnold2013Walking_subject01_walk2.npz)
  2. Experiment 2: Slow Running Imitation (2.0 m/s, Hamner2013_Run20_cycle.npz)
  3. Experiment 3: Fast Running Imitation (5.0 m/s, Hamner2013_Run50_cycle.npz)
  4. Experiment 4: Maximum Height Jump (Pure Motion Generation via RL, No Reference Trial)
"""

import os
import sys

env_path = r"C:\Users\anbes\anaconda3\envs\MTGModel"
t_lib = os.path.join(env_path, r"Lib\site-packages\torch\lib")
l_bin = os.path.join(env_path, r"Library\bin")
if os.path.exists(t_lib):
    os.add_dll_directory(t_lib)
if os.path.exists(l_bin):
    os.add_dll_directory(l_bin)
os.environ["PATH"] = t_lib + ";" + l_bin + ";" + os.environ.get("PATH", "")

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import imageio
import mujoco

sys.path.insert(0, os.path.dirname(__file__))
from Stable_Motion_Tracker import StableTrackingPolicy, VectorStableEnv, StableTrackingEnv, MotionClip

device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------------------------------------------------------
# PPO Training Loop
# -----------------------------------------------------------------------------
def train_ppo_loop(vec_env, policy, num_iters=40, num_envs=64, steps_per_iter=4096, lr=2.0e-4, label=""):
    optimizer = optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, num_iters), eta_min=3e-5)

    rollout_len = steps_per_iter // num_envs
    curr_obs = vec_env.reset()
    t0 = time.time()

    print(f"\n[{label}] Iter | Mean Reward | Pose Err (deg) | Pitch Err (deg) | MTG Excess | Elapsed")
    print(f"[{label}] -----+-------------+----------------+-----------------+------------+--------")

    for it in range(num_iters):
        obs_buf = np.zeros((rollout_len, num_envs, vec_env.obs_dim), dtype=np.float32)
        act_buf = np.zeros((rollout_len, num_envs, 7), dtype=np.float32)
        rew_buf = np.zeros((rollout_len, num_envs), dtype=np.float32)
        done_buf = np.zeros((rollout_len, num_envs), dtype=bool)
        logp_buf = np.zeros((rollout_len, num_envs), dtype=np.float32)
        val_buf = np.zeros((rollout_len, num_envs), dtype=np.float32)

        pe_list, pte_list, ex_list = [], [], []

        for t in range(rollout_len):
            obs_buf[t] = curr_obs
            with torch.no_grad():
                obs_t = torch.from_numpy(curr_obs).to(device)
                action_t, logp_t, val_t = policy.get_action(obs_t)

            action_np = action_t.cpu().numpy()
            logp_np = logp_t.cpu().numpy().squeeze(-1)
            val_np = val_t.cpu().numpy().squeeze(-1)

            next_obs, rewards, dones, infos = vec_env.step(action_np)

            act_buf[t] = action_np
            rew_buf[t] = rewards
            done_buf[t] = dones
            logp_buf[t] = logp_np
            val_buf[t] = val_np

            pe_list.append(infos.get("pose_err", 0.0))
            pte_list.append(infos.get("pitch_err", 0.0))
            ex_list.append(infos.get("mtg_excess", 0.0))

            curr_obs = next_obs

        with torch.no_grad():
            last_obs_t = torch.from_numpy(curr_obs).to(device)
            _, _, next_val_t = policy.get_action(last_obs_t)
            next_val = next_val_t.cpu().numpy().squeeze(-1)

        gamma, lam = 0.98, 0.95
        adv_buf = np.zeros_like(rew_buf)
        last_gae = 0.0
        for t in reversed(range(rollout_len)):
            if t == rollout_len - 1:
                next_non_terminal = 1.0 - done_buf[t].astype(np.float32)
                next_values = next_val
            else:
                next_non_terminal = 1.0 - done_buf[t + 1].astype(np.float32)
                next_values = val_buf[t + 1]
            delta = rew_buf[t] + gamma * next_values * next_non_terminal - val_buf[t]
            adv_buf[t] = last_gae = delta + gamma * lam * next_non_terminal * last_gae

        ret_buf = adv_buf + val_buf

        b_obs = torch.tensor(obs_buf.reshape(-1, vec_env.obs_dim), dtype=torch.float32, device=device)
        b_act = torch.tensor(act_buf.reshape(-1, 7), dtype=torch.float32, device=device)
        b_logp = torch.tensor(logp_buf.reshape(-1), dtype=torch.float32, device=device)
        b_ret = torch.tensor(ret_buf.reshape(-1), dtype=torch.float32, device=device)
        b_adv = torch.tensor(adv_buf.reshape(-1), dtype=torch.float32, device=device)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        batch_size = b_obs.shape[0]
        minibatch_size = 256
        for _ in range(4):
            idx = np.random.permutation(batch_size)
            for s in range(0, batch_size, minibatch_size):
                mb = idx[s:s + minibatch_size]
                new_logp, ent, new_val = policy.evaluate_actions(b_obs[mb], b_act[mb])
                new_logp = new_logp.squeeze(-1)
                new_val = new_val.squeeze(-1)

                ratio = torch.exp(new_logp - b_logp[mb])
                surr1 = ratio * b_adv[mb]
                surr2 = torch.clamp(ratio, 0.8, 1.2) * b_adv[mb]
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = 0.5 * nn.MSELoss()(new_val, b_ret[mb])
                loss = actor_loss + 0.5 * critic_loss - 0.005 * ent.mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()

        scheduler.step()

        mean_rew = np.mean(rew_buf)
        mean_pe = np.mean(pe_list)
        mean_pte = np.mean(pte_list)
        mean_ex = np.mean(ex_list)
        elapsed = time.time() - t0

        if (it + 1) % 5 == 0 or it == 0 or it == num_iters - 1:
            print(f"[{label}]  {it+1:4d} |  {mean_rew:10.2f} |   {mean_pe:12.2f} |    {mean_pte:12.2f} |   {mean_ex:8.3f} | {elapsed:5.1f}s", flush=True)

    return policy


# -----------------------------------------------------------------------------
# Experiment 3 (5.0 m/s Sprint) Calibrated Clip & Tracking Environment
# -----------------------------------------------------------------------------
def load_calibrated_run50_clip(npz_path="motions/Hamner2013_Run50_cycle.npz"):
    raw = np.load(npz_path)
    t = raw["t"].copy()
    q = raw["q"].copy()
    dt = float(t[1] - t[0])
    N = len(t)

    q[:, 5] = np.clip(q[:, 5], -1.95, -0.05)
    q[:, 8] = np.clip(q[:, 8], -1.95, -0.05)

    mj_model = mujoco.MjModel.from_xml_path("biped_10dof.xml")
    mj_data = mujoco.MjData(mj_model)
    foot_l_body = mj_model.body("foot_l").id
    foot_r_body = mj_model.body("foot_r").id

    footL_z = np.zeros(N)
    footR_z = np.zeros(N)
    for i in range(N):
        mj_data.qpos[:] = q[i]
        mujoco.mj_forward(mj_model, mj_data)
        fl_mat = mj_data.xmat[foot_l_body].reshape(3, 3)
        fr_mat = mj_data.xmat[foot_r_body].reshape(3, 3)
        hl = (mj_data.xpos[foot_l_body] + fl_mat @ np.array([-0.07, 0, 0]))[2] - 0.02
        tl = (mj_data.xpos[foot_l_body] + fl_mat @ np.array([0.17, 0, 0]))[2] - 0.02
        hr = (mj_data.xpos[foot_r_body] + fr_mat @ np.array([-0.07, 0, 0]))[2] - 0.02
        tr = (mj_data.xpos[foot_r_body] + fr_mat @ np.array([0.17, 0, 0]))[2] - 0.02
        footL_z[i] = min(hl, tl)
        footR_z[i] = min(hr, tr)

    min_foot_z = np.minimum(footL_z, footR_z)
    z_shift = np.minimum(min_foot_z - 0.001, 0.032)
    pad = 15
    z_shift_smooth = np.convolve(np.pad(z_shift, pad, mode="wrap"), np.ones(2 * pad + 1) / (2 * pad + 1), mode="valid")
    q[:, 1] -= z_shift_smooth

    u = np.gradient(q, dt, axis=0)
    u[-1, 1:] = u[0, 1:]
    return MotionClip(t=t, q=q, u=u)


class SprintTrackingEnv(StableTrackingEnv):
    def step(self, action):
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

        mse_q = np.mean((q_curr[3:10] - q_ref_end[3:10])**2)
        mse_u = np.mean((u_curr[3:10] - u_ref_end[3:10])**2)
        r_q = np.exp(-30.0 * mse_q)
        r_u = np.exp(-0.25 * mse_u)

        p_err_sq = (q_curr[2] - q_ref_end[2])**2
        vp_err_sq = (u_curr[2] - u_ref_end[2])**2
        z_err_sq = (q_curr[1] - q_ref_end[1])**2
        vz_err_sq = (u_curr[1] - u_ref_end[1])**2
        x_err_sq = (q_curr[0] - q_ref_end[0])**2
        vx_err_sq = (u_curr[0] - u_ref_end[0])**2

        r_rot = np.exp(-30.0 * p_err_sq - 1.0 * vp_err_sq)
        r_pos_z = np.exp(-80.0 * z_err_sq - 2.0 * vz_err_sq)
        r_pos_x = np.exp(-5.0 * x_err_sq)
        r_vel = np.exp(-6.0 * vx_err_sq)

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

        info = {
            "pose_err": float(np.degrees(np.sqrt(mse_q))),
            "pitch_err": float(np.degrees(abs(q_curr[2] - q_ref_end[2]))),
            "tau": tau_mean.copy()
        }

        if self.mode == "mode1":
            reward = (
                alive_bonus
                + 0.25 * r_q + 0.05 * r_u + 0.20 * r_rot
                + 0.20 * r_vel + 0.05 * r_pos_x + 0.15 * r_pos_z
                + 0.10 * r_cart + p_act
            )
            info["mtg_excess"] = 0.0
        else:
            T_avail_pos = T_pos_acc / self.decimation
            T_avail_neg = T_neg_acc / self.decimation
            mtg_excess = excess_norm_acc / self.decimation
            p_mtg_excess = -0.35 * float(np.sum(mtg_excess**2)) if self.mtg_phase == "soft" else 0.0
            reward = (
                alive_bonus
                + 0.25 * r_q + 0.05 * r_u + 0.20 * r_rot
                + 0.20 * r_vel + 0.05 * r_pos_x + 0.15 * r_pos_z
                + 0.10 * r_cart + p_act + p_mtg_excess
            )
            info["mtg_excess"] = float(np.sum(mtg_excess))
            info["T_avail_pos"] = T_avail_pos.copy()
            info["T_avail_neg"] = T_avail_neg.copy()

        pitch_err = abs(q_curr[2] - q_ref_end[2])
        z_err = abs(q_curr[1] - q_ref_end[1])
        fell = bool(
            q_curr[1] < 0.72 or q_curr[1] > 0.99 or z_err > 0.085 or pitch_err > 0.45 or u_curr[0] < 3.0
        )
        self.step_count += 1
        done = bool(fell or self.step_count >= self.max_steps)
        if fell:
            reward = 0.0

        info["fell"] = fell
        self.prev_action = action.copy()
        next_obs = self.get_obs()
        return next_obs, reward, done, info


def make_sprint_policy(obs_dim, device, warm_start_path=None):
    policy = StableTrackingPolicy(obs_dim=obs_dim, action_dim=7, device=device)
    old_scale = policy.action_scale.clone()
    new_scale = torch.tensor([0.35, 0.55, 0.65, 0.65, 0.55, 0.65, 0.65], dtype=torch.float32, device=policy.device)

    if warm_start_path and os.path.exists(warm_start_path):
        state_dict = torch.load(warm_start_path, map_location=device)
        if "action_scale" in state_dict:
            old_scale = state_dict["action_scale"].to(policy.device)
        policy.load_state_dict(state_dict)
        ratio = (old_scale / new_scale).to(policy.device)
        with torch.no_grad():
            policy.actor_out.weight.mul_(ratio.unsqueeze(1))
            policy.actor_out.bias.mul_(ratio)

    policy.action_scale.copy_(new_scale)
    return policy


# -----------------------------------------------------------------------------
# Unified 500 Hz Evaluation, Torque + Limits + Excess Plotter, and GIF Renderer
# -----------------------------------------------------------------------------
def evaluate_500hz_experiment(
    exp_title,
    ref_speed,
    clip,
    policy1,
    policy2,
    gif_path,
    plot_path,
    max_steps=200,
    env_cls=StableTrackingEnv
):
    print("\n" + "=" * 78)
    print(f"   500 Hz EVALUATION & RENDERING: {exp_title}")
    print("=" * 78)
    os.makedirs("renders", exist_ok=True)

    env1 = env_cls(clip, mode="mode1", max_steps=max_steps, enable_perturbations=False)
    env2 = env_cls(clip, mode="mode2", mtg_phase="hard", max_steps=max_steps, enable_perturbations=False)

    obs1 = env1.reset(start_frame=0)
    obs2 = env2.reset(start_frame=0)

    overlay_model = mujoco.MjModel.from_xml_path("biped_overlay.xml")
    overlay_data = mujoco.MjData(overlay_model)
    renderer = mujoco.Renderer(overlay_model, height=540, width=960)
    camera = mujoco.MjvCamera()
    camera.distance = 4.2
    camera.elevation = -10.0
    camera.azimuth = 70.0

    t_500 = []
    tau1_500 = []
    tau2_pd_500 = []
    tau2_clamped_500 = []
    tau2_passive_500 = []
    T_pos2_500 = []
    T_neg2_500 = []
    T_act_pos2_500 = []
    T_act_neg2_500 = []
    T_LV_pos2_500 = []
    T_LV_neg2_500 = []
    vx1_500, vx2_500 = [], []
    ex1_500, ex2_500 = [], []

    a1_shadow = np.full(7, env2.env.actuator.a_init, dtype=np.float32)

    frames = []
    fell1, fell2 = False, False
    step1_survived, step2_survived = 0, 0
    substep_idx = 0

    for step in range(max_steps):
        with torch.no_grad():
            a1, _, _ = policy1.get_action(torch.tensor(obs1, dtype=torch.float32, device=device).unsqueeze(0), deterministic=True)
            a2, _, _ = policy2.get_action(torch.tensor(obs2, dtype=torch.float32, device=device).unsqueeze(0), deterministic=True)
        act1 = a1.squeeze(0).cpu().numpy()
        act2 = a2.squeeze(0).cpu().numpy()

        prev_f1 = env1.filtered_action.copy()
        f1 = 0.40 * env1.filtered_action + 0.60 * act1
        env1.filtered_action = f1.copy()

        prev_f2 = env2.filtered_action.copy()
        f2 = 0.40 * env2.filtered_action + 0.60 * act2
        env2.filtered_action = f2.copy()

        for sub_step in range(env1.decimation):
            alpha = (sub_step + 1) / float(env1.decimation)

            # --- Mode 1 Sub-step ---
            if not fell1:
                q_ref_1, u_ref_1 = clip.get_frame(env1.current_frame + sub_step)
                act_k1 = (1.0 - alpha) * prev_f1 + alpha * f1
                q_des1 = q_ref_1[3:10] + act_k1
                u_des1 = u_ref_1[3:10]

                q_j1 = env1.env.data.qpos[3:10].astype(np.float32)
                u_j1 = env1.env.data.qvel[3:10].astype(np.float32)
                T_LV_p1, T_LV_n1 = env2.env.actuator.compute_capacities_np(q_j1, u_j1)
                tau_pass1 = env2.env.actuator.compute_passive_np(q_j1)
                _, tau1_k = env1.env.step_simulation(q_des1, dt_sim=0.002, u_des=u_des1)
                tau_act_des1 = tau1_k - tau_pass1
                T_LV_dir1 = np.where(tau_act_des1 >= 0.0, T_LV_p1, T_LV_n1)
                u_req1 = np.clip(np.abs(tau_act_des1) / (T_LV_dir1 + 1e-3), 0.0, 1.0)
                tau_c1 = np.where(u_req1 >= a1_shadow, env2.env.actuator.tau_act, env2.env.actuator.tau_deact)
                a1_shadow = np.clip(a1_shadow + 0.002 * (u_req1 - a1_shadow) / tau_c1, env2.env.actuator.a_min, 1.0)
                Tp1 = a1_shadow * T_LV_p1 + tau_pass1
                Tn1 = a1_shadow * T_LV_n1 - tau_pass1
                ex1_k = float(np.sum(np.maximum(0.0, tau1_k - Tp1) + np.maximum(0.0, -tau1_k - Tn1)))
                vx1_k = float(env1.env.data.qvel[0])
            else:
                tau1_k = np.full(7, np.nan, dtype=np.float32)
                ex1_k = np.nan
                vx1_k = np.nan

            # --- Mode 2 Sub-step ---
            if not fell2:
                q_ref_2, u_ref_2 = clip.get_frame(env2.current_frame + sub_step)
                act_k2 = (1.0 - alpha) * prev_f2 + alpha * f2
                q_des2 = q_ref_2[3:10] + act_k2
                u_des2 = u_ref_2[3:10]

                q_j2 = env2.env.data.qpos[3:10].astype(np.float32)
                u_j2 = env2.env.data.qvel[3:10].astype(np.float32)
                tau2_pd_k = env2.env.actuator.Kp_np * (q_des2 - q_j2) + env2.env.actuator.Kd_np * (u_des2 - u_j2)

                _, tau2_k = env2.env.step_simulation(q_des2, dt_sim=0.002, u_des=u_des2)
                Tp2 = env2.env.actuator.last_T_pos
                Tn2 = env2.env.actuator.last_T_neg
                T_act_p2 = env2.env.actuator.last_T_act_pos
                T_act_n2 = env2.env.actuator.last_T_act_neg
                tau2_pass_k = env2.env.actuator.last_tau_passive
                T_LV_p2 = env2.env.actuator.last_T_LV_pos
                T_LV_n2 = env2.env.actuator.last_T_LV_neg
                ex2_k = float(np.sum(np.maximum(0.0, tau2_k - Tp2) + np.maximum(0.0, -tau2_k - Tn2)))
                vx2_k = float(env2.env.data.qvel[0])
            else:
                tau2_pd_k = np.full(7, np.nan, dtype=np.float32)
                tau2_k = np.full(7, np.nan, dtype=np.float32)
                tau2_pass_k = np.full(7, np.nan, dtype=np.float32)
                Tp2 = np.full(7, np.nan, dtype=np.float32)
                Tn2 = np.full(7, np.nan, dtype=np.float32)
                T_act_p2 = np.full(7, np.nan, dtype=np.float32)
                T_act_n2 = np.full(7, np.nan, dtype=np.float32)
                T_LV_p2 = np.full(7, np.nan, dtype=np.float32)
                T_LV_n2 = np.full(7, np.nan, dtype=np.float32)
                ex2_k = np.nan
                vx2_k = np.nan

            t_500.append(substep_idx * 0.002)
            tau1_500.append(tau1_k.copy())
            tau2_pd_500.append(tau2_pd_k.copy())
            tau2_clamped_500.append(tau2_k.copy())
            tau2_passive_500.append(tau2_pass_k.copy())
            T_pos2_500.append(Tp2.copy())
            T_neg2_500.append(Tn2.copy())
            T_act_pos2_500.append(T_act_p2.copy())
            T_act_neg2_500.append(T_act_n2.copy())
            T_LV_pos2_500.append(T_LV_p2.copy())
            T_LV_neg2_500.append(T_LV_n2.copy())
            vx1_500.append(vx1_k)
            vx2_500.append(vx2_k)
            ex1_500.append(ex1_k)
            ex2_500.append(ex2_k)
            substep_idx += 1

        if not fell1:
            env1.current_frame += env1.decimation
            env1.step_count += 1
            env1.prev_action = act1.copy()
            obs1 = env1.get_obs()
            step1_survived = step + 1
            q1 = env1.env.data.qpos
            q_ref_end1, _ = clip.get_frame(env1.current_frame)
            if q1[1] < 0.50 or abs(q1[2] - q_ref_end1[2]) > 0.65:
                fell1 = True

        if not fell2:
            env2.current_frame += env2.decimation
            env2.step_count += 1
            env2.prev_action = act2.copy()
            obs2 = env2.get_obs()
            step2_survived = step + 1
            q2 = env2.env.data.qpos
            q_ref_end2, _ = clip.get_frame(env2.current_frame)
            if q2[1] < 0.50 or abs(q2[2] - q_ref_end2[2]) > 0.65:
                fell2 = True

        q1 = env1.env.data.qpos
        q2 = env2.env.data.qpos
        overlay_data.qpos[:10] = q1[:10]
        overlay_data.qpos[10:20] = q2[:10]
        mujoco.mj_forward(overlay_model, overlay_data)

        camera.lookat[0] = q1[0] + 0.25 if not fell1 else max(q1[0], q2[0]) + 0.25
        camera.lookat[1] = 0.0
        camera.lookat[2] = 0.70

        renderer.update_scene(overlay_data, camera)
        rgb = renderer.render()
        pil_img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_img)

        v1_disp = env1.env.data.qvel[0] if not fell1 else 0.0
        v2_disp = env2.env.data.qvel[0] if not fell2 else 0.0
        s1_str = f"{q1[0]:.2f}m ({v1_disp:.2f}m/s)" if not fell1 else f"{q1[0]:.2f}m [FELL]"
        s2_str = f"{q2[0]:.2f}m ({v2_disp:.2f}m/s)" if not fell2 else f"{q2[0]:.2f}m [FELL]"
        text_str = f"{exp_title} | Mode 1 Unlimited (Ghost): {s1_str} | Mode 2 MTG Clamped (Solid): {s2_str}"
        draw.rectangle([(15, 15), (945, 52)], fill=(15, 23, 42, 220), outline=(56, 189, 248), width=2)
        draw.text((25, 25), text_str, fill=(240, 249, 255))
        frames.append(np.array(pil_img))

        if fell1 and fell2:
            break

    imageio.mimsave(gif_path, frames, fps=30, loop=0)

    t_arr = np.array(t_500)
    tau1_arr = np.array(tau1_500)
    tau2_pd_arr = np.array(tau2_pd_500)
    tau2_clamped_arr = np.array(tau2_clamped_500)
    tau2_pass_arr = np.array(tau2_passive_500)
    Tp2_arr = np.array(T_pos2_500)
    Tn2_arr = np.array(T_neg2_500)
    T_act_p2_arr = np.array(T_act_pos2_500)
    T_act_n2_arr = np.array(T_act_neg2_500)
    T_LV_p2_arr = np.array(T_LV_pos2_500)
    T_LV_n2_arr = np.array(T_LV_neg2_500)
    ex1_arr = np.array(ex1_500)
    ex2_arr = np.array(ex2_500)

    valid2 = ~np.isnan(ex2_arr)
    clamped_mask = np.abs(tau2_pd_arr[valid2] - tau2_clamped_arr[valid2]) > 1e-3
    clamp_pct_per_joint = clamped_mask.mean(axis=0) * 100.0
    joints = ["Lumbar", "Hip (L)", "Knee (L)", "Ankle (L)", "Hip (R)", "Knee (R)", "Ankle (R)"]

    print(f"\n[DONE] Saved Overlay GIF: {gif_path}")
    print(f"  Mode 1 (Unlimited PD): Steps = {step1_survived}/{max_steps} ({step1_survived*0.02:.2f}s) | Fell = {fell1} | Final X = {env1.env.data.qpos[0]:.2f}m | Mean Vx = {np.nanmean(vx1_500):.2f} m/s | Mean Excess = {np.nanmean(ex1_arr):.2f} N*m | Max Excess = {np.nanmax(ex1_arr):.2f} N*m")
    print(f"  Mode 2 (MTG Clamped) : Steps = {step2_survived}/{max_steps} ({step2_survived*0.02:.2f}s) | Fell = {fell2} | Final X = {env2.env.data.qpos[0]:.2f}m | Mean Vx = {np.nanmean(vx2_500):.2f} m/s | Mean Excess = {np.nanmean(ex2_arr):.6f} N*m | Max Excess = {np.nanmax(ex2_arr):.6f} N*m")
    print("  Active Clamping Frequency per Joint in Mode 2 (% of 500 Hz sub-steps where |tau_act_des| > a(t)*T_LV):")
    for j_idx, j_name in enumerate(joints):
        max_clip = np.nanmax(np.abs(tau2_pd_arr[:, j_idx] - tau2_clamped_arr[:, j_idx]))
        max_pass = np.nanmax(np.abs(tau2_pass_arr[:, j_idx]))
        print(f"    - {j_name:<10}: {clamp_pct_per_joint[j_idx]:5.1f}% of sub-steps clamped (Max clipped: {max_clip:6.1f} N*m | Max |tau_pass|: {max_pass:5.1f} N*m)")

    fig, axes = plt.subplots(4, 2, figsize=(16, 13.5), sharex=True)
    axes = axes.flatten()

    for j in range(7):
        ax = axes[j]
        ax.plot(t_arr, T_LV_p2_arr[:, j], color="gray", linestyle="--", lw=0.8, alpha=0.45, zorder=1, label="Full-Activation Capacity T_max*f_L*f_V (a=1)")
        ax.plot(t_arr, -T_LV_n2_arr[:, j], color="gray", linestyle="--", lw=0.8, alpha=0.45, zorder=1)
        ax.fill_between(t_arr, -T_act_n2_arr[:, j], T_act_p2_arr[:, j], color="lightgray", alpha=0.60, label="Active Clamp Envelope [±a(t)*T_max*f_L*f_V]")
        ax.plot(t_arr, Tp2_arr[:, j], "k:", lw=1.2, alpha=0.9, zorder=5, label="Total MTG Limit [±a(t)*T_LV + tau_passive]")
        ax.plot(t_arr, -Tn2_arr[:, j], "k:", lw=1.2, alpha=0.9, zorder=5)

        ax.plot(t_arr, tau1_arr[:, j], "b--", lw=1.2, alpha=0.70, zorder=2, label="Mode 1 Applied Torque (Unlimited PD)")
        ax.plot(t_arr, tau2_pd_arr[:, j], color="darkorange", linestyle=":", lw=1.1, alpha=0.75, zorder=3, label="Mode 2 Unclamped PD Demand")
        ax.plot(t_arr, tau2_pass_arr[:, j], color="seagreen", linestyle="-", lw=1.2, alpha=0.85, zorder=3, label="Mode 2 Passive Torque tau_passive(q)")
        ax.plot(t_arr, tau2_clamped_arr[:, j], "r-", lw=1.8, zorder=4, label="Mode 2 Total Torque [tau_active + tau_passive]")

        ax.set_title(
            f"{joints[j]} Torque — {exp_title} (Clamped {clamp_pct_per_joint[j]:.1f}% of sub-steps)",
            fontsize=10.5, fontweight="bold"
        )
        ax.set_ylabel("Torque (N*m)")
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(loc="upper right", fontsize=6.5)

    ax8 = axes[7]
    ax8_twin = ax8.twinx()
    ax8.plot(t_arr, vx1_500, "b--", lw=1.6, label="Mode 1 Speed (m/s)")
    ax8.plot(t_arr, vx2_500, "r-", lw=1.8, label="Mode 2 Speed (m/s)")
    ax8.axhline(ref_speed, color="green", linestyle=":", lw=1.5, label=f"Reference Trial ({ref_speed:.2f} m/s)")
    ax8.set_ylabel("Forward Velocity (m/s)", color="black")
    ax8.set_ylim(0.0, max(3.0, ref_speed + 1.5))

    ax8_twin.plot(t_arr, ex1_arr, color="darkorange", linestyle="-", lw=1.4, alpha=0.8, label="Mode 1 Exceeded Torque (N*m)")
    ax8_twin.plot(t_arr, ex2_arr, color="darkred", linestyle="-", lw=2.0, label="Mode 2 Exceeded Torque (= 0.00 N*m)")
    ax8_twin.set_ylabel("500 Hz Exceeded Torque (N*m)", color="darkorange")

    lines1, labels1 = ax8.get_legend_handles_labels()
    lines2, labels2 = ax8_twin.get_legend_handles_labels()
    ax8.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=7.5)

    ax8.set_title(f"{exp_title}: Forward Speed & 500 Hz Exceeded Torque", fontsize=10.5, fontweight="bold")
    ax8.set_xlabel("Time (s)")
    ax8.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[DONE] Saved Torque & Limits Plot: {plot_path}")


# -----------------------------------------------------------------------------
# Experiment 4: Maximum Height Vertical Jump (Pure Motion Generation via RL)
# -----------------------------------------------------------------------------
class MaxJumpEnv:
    """
    Pure Motion Generation Environment for Maximum Height Vertical Jump
    with physiological joint range protection (injury-free countermovement & takeoff).
    Simulates from quiet standing (z = 0.918 m) through countermovement squat,
    explosive triple-extension takeoff, and airborne flight until the maximum
    apex height (max_z) is reached. Zero reference clip or imitation.
    """
    def __init__(self, clip=None, mode="mode1", mtg_phase="hard", decimation=10, max_steps=60,
                 enable_perturbations=False):
        from Biped_Env import BipedEnv
        self.clip = clip
        self.mode = "mode1" if mode in ["mode1", "ideal_unlimited", "unlimited"] else "mode2"
        self.mtg_phase = mtg_phase
        self.decimation = decimation
        self.max_steps = max_steps

        if self.mode == "mode1":
            actuator_type = "ideal_unlimited"
        else:
            actuator_type = "ideal_unlimited" if self.mtg_phase == "soft" else "mtg_clamped"

        self.env = BipedEnv(actuator_type=actuator_type, device="cpu")
        self.pelvis_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, "pelvis_geom")
        self.sole_l_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, "foot_l_sole")
        self.sole_r_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, "foot_r_sole")
        self.trunk_tip_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_SITE, "trunk_tip")
        self.foot_l_body = self.env.model.body("foot_l").id
        self.foot_r_body = self.env.model.body("foot_r").id

        self.z_stand = 0.9178
        self.qpos_stand = np.array(
            [0.0, 0.9178, -0.0048, 0.0324, -0.0477, -0.0726, 0.0250, -0.0477, -0.0726, 0.0250],
            dtype=np.float64
        )
        # Balanced, non-injurious athletic ready crouch (z ~ 0.78m, within safe physiological ROM)
        self.q_nom = np.array([0.24, -0.58, -0.88, 0.45, -0.58, -0.88, 0.45], dtype=np.float32)

        # Safe non-injurious physiological joint angle limits [Lumbar, Hip_L, Knee_L, Ankle_L, Hip_R, Knee_R, Ankle_R]
        # Hips: [-57 deg, +15 deg], Knees: [-82 deg, -4 deg], Ankles: [-35 deg, +28 deg]
        self.q_safe_min = np.array([-0.25, -1.00, -1.43, -0.60, -1.00, -1.43, -0.60], dtype=np.float32)
        self.q_safe_max = np.array([ 0.40,  0.25, -0.07,  0.48,  0.25, -0.07,  0.48], dtype=np.float32)

        self.prev_action = np.zeros(7, dtype=np.float32)
        self.filtered_action = np.zeros(7, dtype=np.float32)
        self.step_count = 0
        self.current_frame = 0
        self.max_z = self.z_stand
        self.min_z = self.z_stand
        self.max_foot_z = 0.0
        self.max_vz = 0.0
        self.has_taken_off = False
        self.apex_reached = False
        self.post_apex_steps = 0

        # Passive tissue parameters
        self.k1_pass = np.array([250.0, 200.0, 300.0, 150.0, 200.0, 300.0, 150.0], dtype=np.float32)
        self.k2_pass = np.array([250.0, 200.0, 300.0, 150.0, 200.0, 300.0, 150.0], dtype=np.float32)
        self.q1_pass = np.array([-0.40, -0.40, -1.80, -0.40, -0.40, -1.80, -0.40], dtype=np.float32)
        self.q2_pass = np.array([0.50, 1.20, 0.05, 0.40, 1.20, 0.05, 0.40], dtype=np.float32)

        if self.mode == "mode2":
            self.T_max_pos = np.array([300.0, 400.0, 500.0, 100.0, 400.0, 500.0, 100.0], dtype=np.float32)
            self.T_max_neg = np.array([400.0, 300.0, 200.0, 380.0, 300.0, 200.0, 380.0], dtype=np.float32)
            self.q_opt = np.array([0.0, -0.40, -0.90, 0.35, -0.40, -0.90, 0.35], dtype=np.float32)
            self.w = np.array([1.20, 1.20, 1.20, 1.10, 1.20, 1.20, 1.10], dtype=np.float32)
            self.u_max_pos = np.array([12.0, 15.0, 18.0, 16.0, 15.0, 18.0, 16.0], dtype=np.float32)
            self.u_max_neg = np.array([12.0, 15.0, 18.0, 22.0, 15.0, 18.0, 22.0], dtype=np.float32)
            self.tau_act = 0.020
            self.tau_deact = 0.040
            self.a_min = 0.01
            self.a_init = 0.35
            self.a_soft = np.full(7, self.a_init, dtype=np.float32)

    def reset(self, start_frame=None):
        q0 = self.qpos_stand.copy()
        u0 = np.zeros(10, dtype=np.float64)
        if start_frame is None:
            q0[2] += np.random.uniform(-0.008, 0.008)
            q0[3:10] += np.random.uniform(-0.012, 0.012, size=7)

        self.env.reset(qpos_init=q0, qvel_init=u0)
        if self.mode == "mode2":
            self.a_soft = np.full(7, self.a_init, dtype=np.float32)

        init_act = (q0[3:10].astype(np.float32) - self.q_nom)
        self.prev_action = init_act.copy()
        self.filtered_action = init_act.copy()
        self.step_count = 0
        self.current_frame = 0
        self.max_z = self.z_stand
        self.min_z = self.z_stand
        self.max_foot_z = 0.0
        self.max_vz = 0.0
        self.has_taken_off = False
        self.apex_reached = False
        self.post_apex_steps = 0
        return self.get_obs()

    def _get_contacts_and_foot_z(self):
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

        fl_mat = d.xmat[self.foot_l_body].reshape(3, 3)
        fr_mat = d.xmat[self.foot_r_body].reshape(3, 3)
        hl = (d.xpos[self.foot_l_body] + fl_mat @ np.array([-0.07, 0, 0]))[2] - 0.02
        tl = (d.xpos[self.foot_l_body] + fl_mat @ np.array([0.17, 0, 0]))[2] - 0.02
        hr = (d.xpos[self.foot_r_body] + fr_mat @ np.array([-0.07, 0, 0]))[2] - 0.02
        tr = (d.xpos[self.foot_r_body] + fr_mat @ np.array([0.17, 0, 0]))[2] - 0.02
        min_foot_z = float(min(hl, tl, hr, tr))
        return fl, fr, min_foot_z

    def get_obs(self):
        q_curr = self.env.data.qpos
        u_curr = self.env.data.qvel

        q_joints = q_curr[3:10].astype(np.float32)
        u_joints = u_curr[3:10].astype(np.float32)
        q_rel_nom = q_joints - self.q_nom

        root_state = np.array([
            q_curr[0],
            q_curr[1] - self.z_stand,
            q_curr[2],
            u_curr[0] * 0.2,
            u_curr[1] * 0.2,
            u_curr[2] * 0.2
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

        fl, fr, min_foot_z = self._get_contacts_and_foot_z()
        contacts = np.array([min(fl, 2500.0) / 800.0, min(fr, 2500.0) / 800.0], dtype=np.float32)

        t_norm = self.step_count / float(max(1, self.max_steps))
        phase_feat = np.array([
            t_norm,
            np.sin(2.0 * np.pi * t_norm),
            np.cos(2.0 * np.pi * t_norm),
            self.max_z - self.z_stand,
            self.z_stand - self.min_z,
            max(0.0, min_foot_z),
            float(self.has_taken_off)
        ], dtype=np.float32)

        return np.concatenate([
            q_joints,
            u_joints * 0.1,
            q_rel_nom,
            root_state,
            rel_ee_pos,
            contacts,
            self.filtered_action,
            self.prev_action,
            phase_feat,
            np.zeros(8, dtype=np.float32)
        ], dtype=np.float32)

    def _compute_safe_q_des(self, raw_q_des, q_jk):
        """
        Applies physiological joint-stop cushioning so the PD target can command strong
        extension torque during mid-stroke push-off while never hyperextending or
        slamming joints into injurious anatomical limits.
        """
        # Prevent deep flexion target below safe squat bounds
        q_des = np.maximum(raw_q_des, np.array([-0.28, -1.02, -1.42, -0.85, -1.02, -1.42, -0.85], dtype=np.float32))
        # Prevent excessive lumbar/hip/ankle dorsiflexion target
        q_des = np.minimum(q_des, np.array([0.40, 0.65, 0.85, 0.50, 0.65, 0.85, 0.50], dtype=np.float32))

        # Soft knee hyper-extension protection: as physical knee angle approaches -0.05 rad (-3 deg),
        # smoothly taper max knee q_des from +0.85 down to -0.05 rad so knees never lock or slam into 0.0 rad!
        for k_idx in [2, 5]:
            if q_jk[k_idx] > -0.25:
                frac = np.clip((q_jk[k_idx] - (-0.25)) / 0.20, 0.0, 1.0)
                k_ceil = (1.0 - frac) * 0.85 + frac * (-0.05)
                q_des[k_idx] = min(q_des[k_idx], k_ceil)

        # Once airborne, hold a natural extended flight posture within safe bounds
        if self.has_taken_off:
            q_des = np.clip(q_des, self.q_safe_min, self.q_safe_max)
        return q_des

    def step(self, action):
        prev_filtered = self.filtered_action.copy()
        filtered = 0.25 * self.filtered_action + 0.75 * action
        self.filtered_action = filtered.copy()

        tau_acc = np.zeros(7, dtype=np.float32)
        T_pos_acc = np.zeros(7, dtype=np.float32)
        T_neg_acc = np.zeros(7, dtype=np.float32)
        excess_norm_acc = np.zeros(7, dtype=np.float32)

        prev_max_z = self.max_z
        step_max_z = float(self.env.data.qpos[1])
        step_max_vz = float(self.env.data.qvel[1])

        for sub_step in range(self.decimation):
            alpha = (sub_step + 1) / float(self.decimation)
            act_k = (1.0 - alpha) * prev_filtered + alpha * filtered
            q_jk = self.env.data.qpos[3:10].astype(np.float32)
            q_des = self._compute_safe_q_des(self.q_nom + act_k, q_jk)

            if self.mode == "mode2" and self.mtg_phase == "soft":
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

            _, tau = self.env.step_simulation(q_des, dt_sim=0.002, u_des=None)
            tau_acc += tau
            z_sub = float(self.env.data.qpos[1])
            vz_sub = float(self.env.data.qvel[1])
            if z_sub > step_max_z:
                step_max_z = z_sub
            if vz_sub > step_max_vz:
                step_max_vz = vz_sub
            if not self.has_taken_off and z_sub < self.min_z:
                self.min_z = z_sub

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
        q_curr = self.env.data.qpos
        u_curr = self.env.data.qvel
        z_curr = float(q_curr[1])
        vz_curr = float(u_curr[1])
        fl, fr, min_foot_z = self._get_contacts_and_foot_z()
        airborne = (fl < 1.0 and fr < 1.0)

        if step_max_z > self.max_z:
            self.max_z = step_max_z
        if step_max_vz > self.max_vz:
            self.max_vz = step_max_vz
        if min_foot_z > self.max_foot_z:
            self.max_foot_z = min_foot_z

        if airborne and z_curr > self.z_stand + 0.02:
            self.has_taken_off = True
        if self.has_taken_off and vz_curr <= 0.0:
            self.apex_reached = True
            self.post_apex_steps += 1

        # Pure Max-Height Reward + Joint Limit Injury Prevention (Terminating at Apex Height)
        delta_max_z = max(0.0, self.max_z - prev_max_z)
        r_new_peak = 55.0 * delta_max_z

        peak_gain = max(0.0, self.max_z - self.z_stand)
        r_peak_hold = 4.5 * peak_gain + 14.0 * (peak_gain ** 2)

        vz_up = max(0.0, vz_curr)
        z_ballistic = z_curr + (vz_up ** 2) / (2.0 * 9.81)
        crouch_stroke = max(0.0, min(0.24, self.z_stand - self.min_z))
        if self.step_count < 14 and not self.has_taken_off:
            r_launch = 0.5 * crouch_stroke * np.exp(-6.0 * (q_curr[2] ** 2))
        else:
            r_launch = (
                3.5 * max(0.0, z_ballistic - self.z_stand)
                + 1.8 * max(0.0, z_ballistic - self.min_z) * float(vz_curr > 0.1)
                + 2.0 * max(0.0, z_curr - self.z_stand)
            )

        q_j = q_curr[3:10].astype(np.float32)
        over_low = np.maximum(0.0, self.q_safe_min - q_j)
        over_high = np.maximum(0.0, q_j - self.q_safe_max)
        p_joint_injury = -2.5 * float(np.sum(over_low ** 2 + over_high ** 2))

        upright_mult = float(np.exp(-6.0 * (q_curr[2] ** 2) - 2.5 * (q_curr[0] ** 2)))
        sym_err = np.sum((q_curr[4:7] - q_curr[7:10]) ** 2) + np.sum((action[1:4] - action[4:7]) ** 2)
        p_sym = -0.15 * sym_err
        p_act = -0.005 * np.sum((action - self.prev_action) ** 2)

        reward = float(
            (r_new_peak + r_peak_hold + r_launch) * (0.4 + 0.6 * upright_mult)
            + 0.05 * upright_mult + p_sym + p_act + p_joint_injury
        )

        if self.apex_reached and self.post_apex_steps == 2:
            reward += float((45.0 * peak_gain + 90.0 * (peak_gain ** 2)) * (0.4 + 0.6 * upright_mult))

        if self.mode == "mode2":
            mtg_excess = excess_norm_acc / self.decimation
            if self.mtg_phase == "soft":
                reward -= 0.40 * float(np.sum(mtg_excess ** 2))
            mtg_ex_val = float(np.sum(mtg_excess))
        else:
            mtg_ex_val = 0.0

        # Prevent deep unphysiological squat collapse (< 0.62m) or excessive torso pitch (> 0.55 rad)
        fell = bool(z_curr < 0.62 or abs(q_curr[2]) > 0.55 or abs(q_curr[0]) > 0.75)
        self.step_count += 1
        done = bool(fell or (self.apex_reached and self.post_apex_steps >= 2) or self.step_count >= self.max_steps)
        if fell:
            reward = -1.5

        info = {
            "max_z": float(self.max_z),
            "jump_height": float(self.max_z - self.z_stand),
            "max_foot_z": float(self.max_foot_z),
            "pose_err": float((self.max_z - self.z_stand) * 100.0),
            "pitch_err": float(np.degrees(abs(q_curr[2]))),
            "mtg_excess": mtg_ex_val,
            "fell": fell,
            "apex_reached": self.apex_reached
        }
        self.prev_action = action.copy()
        return self.get_obs(), reward, done, info


class VectorJumpEnv:
    def __init__(self, num_envs=64, mode="mode1", mtg_phase="hard", max_steps=60):
        self.num_envs = num_envs
        self.envs = [MaxJumpEnv(mode=mode, mtg_phase=mtg_phase, max_steps=max_steps) for _ in range(num_envs)]
        dummy = self.envs[0].reset()
        self.obs_dim = dummy.shape[0]

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
                pe_list.append(info["pose_err"])
                n_obs = self.envs[i].reset()
            next_obs_arr[i] = n_obs
            rew_arr[i] = r
            done_arr[i] = d
            pte_list.append(info["pitch_err"])
            ex_list.append(info.get("mtg_excess", 0.0))

        return next_obs_arr, rew_arr, done_arr, {
            "pose_err": float(np.mean(pe_list)) if pe_list else 0.0,
            "pitch_err": float(np.mean(pte_list)),
            "mtg_excess": float(np.mean(ex_list))
        }


def make_jump_policy(obs_dim=64, device=device, warm_start_path=None):
    policy = StableTrackingPolicy(obs_dim=obs_dim, action_dim=7, device=device)
    new_scale = torch.tensor([0.35, 1.15, 1.55, 1.25, 1.15, 1.55, 1.25], dtype=torch.float32, device=policy.device)
    policy.action_scale.copy_(new_scale)
    with torch.no_grad():
        policy.log_std.fill_(-1.1)
    if warm_start_path and os.path.exists(warm_start_path):
        policy.load_state_dict(torch.load(warm_start_path, map_location=device))
    return policy


def evaluate_500hz_jump_experiment(
    exp_title,
    policy1,
    policy2,
    gif_path,
    plot_path,
    max_steps=60
):
    print("\n" + "=" * 78)
    print(f"   500 Hz EVALUATION & RENDERING: {exp_title}")
    print("=" * 78)
    os.makedirs("renders", exist_ok=True)

    env1 = MaxJumpEnv(mode="mode1", max_steps=max_steps)
    env2 = MaxJumpEnv(mode="mode2", mtg_phase="hard", max_steps=max_steps)

    obs1 = env1.reset(start_frame=0)
    obs2 = env2.reset(start_frame=0)

    overlay_model = mujoco.MjModel.from_xml_path("biped_overlay.xml")
    overlay_data = mujoco.MjData(overlay_model)
    renderer = mujoco.Renderer(overlay_model, height=540, width=960)
    camera = mujoco.MjvCamera()
    camera.distance = 3.8
    camera.elevation = -8.0
    camera.azimuth = 85.0

    t_500 = []
    tau1_500 = []
    tau2_pd_500 = []
    tau2_clamped_500 = []
    tau2_passive_500 = []
    T_pos2_500 = []
    T_neg2_500 = []
    T_act_pos2_500 = []
    T_act_neg2_500 = []
    T_LV_pos2_500 = []
    T_LV_neg2_500 = []
    z1_500, z2_500 = [], []
    ex1_500, ex2_500 = [], []

    a1_shadow = np.full(7, env2.env.actuator.a_init, dtype=np.float32)

    frames = []
    done1, done2 = False, False
    fell1, fell2 = False, False
    step1_survived, step2_survived = 0, 0
    substep_idx = 0
    q1_apex_pose = None
    q2_apex_pose = None

    for step in range(max_steps):
        with torch.no_grad():
            a1, _, _ = policy1.get_action(torch.tensor(obs1, dtype=torch.float32, device=device).unsqueeze(0), deterministic=True)
            a2, _, _ = policy2.get_action(torch.tensor(obs2, dtype=torch.float32, device=device).unsqueeze(0), deterministic=True)
        act1 = a1.squeeze(0).cpu().numpy()
        act2 = a2.squeeze(0).cpu().numpy()

        prev_f1 = env1.filtered_action.copy()
        f1 = 0.25 * env1.filtered_action + 0.75 * act1
        if not done1:
            env1.filtered_action = f1.copy()

        prev_f2 = env2.filtered_action.copy()
        f2 = 0.25 * env2.filtered_action + 0.75 * act2
        if not done2:
            env2.filtered_action = f2.copy()

        for sub_step in range(env1.decimation):
            alpha = (sub_step + 1) / float(env1.decimation)

            # --- Mode 1 Sub-step ---
            if not done1:
                act_k1 = (1.0 - alpha) * prev_f1 + alpha * f1
                q_j1 = env1.env.data.qpos[3:10].astype(np.float32)
                u_j1 = env1.env.data.qvel[3:10].astype(np.float32)
                q_des1 = env1._compute_safe_q_des(env1.q_nom + act_k1, q_j1)

                T_LV_p1, T_LV_n1 = env2.env.actuator.compute_capacities_np(q_j1, u_j1)
                tau_pass1 = env2.env.actuator.compute_passive_np(q_j1)
                _, tau1_k = env1.env.step_simulation(q_des1, dt_sim=0.002, u_des=None)
                tau_act_des1 = tau1_k - tau_pass1
                T_LV_dir1 = np.where(tau_act_des1 >= 0.0, T_LV_p1, T_LV_n1)
                u_req1 = np.clip(np.abs(tau_act_des1) / (T_LV_dir1 + 1e-3), 0.0, 1.0)
                tau_c1 = np.where(u_req1 >= a1_shadow, env2.env.actuator.tau_act, env2.env.actuator.tau_deact)
                a1_shadow = np.clip(a1_shadow + 0.002 * (u_req1 - a1_shadow) / tau_c1, env2.env.actuator.a_min, 1.0)
                Tp1 = a1_shadow * T_LV_p1 + tau_pass1
                Tn1 = a1_shadow * T_LV_n1 - tau_pass1
                ex1_k = float(np.sum(np.maximum(0.0, tau1_k - Tp1) + np.maximum(0.0, -tau1_k - Tn1)))
                z1_k = float(env1.env.data.qpos[1])
                vz1_k = float(env1.env.data.qvel[1])
                if z1_k > env1.max_z:
                    env1.max_z = z1_k
                if vz1_k > env1.max_vz:
                    env1.max_vz = vz1_k
                if not env1.has_taken_off and z1_k < env1.min_z:
                    env1.min_z = z1_k
            else:
                tau1_k = np.full(7, np.nan, dtype=np.float32)
                ex1_k = np.nan
                z1_k = float(env1.max_z)

            # --- Mode 2 Sub-step ---
            if not done2:
                act_k2 = (1.0 - alpha) * prev_f2 + alpha * f2
                q_j2 = env2.env.data.qpos[3:10].astype(np.float32)
                u_j2 = env2.env.data.qvel[3:10].astype(np.float32)
                q_des2 = env2._compute_safe_q_des(env2.q_nom + act_k2, q_j2)
                tau2_pd_k = env2.env.actuator.Kp_np * (q_des2 - q_j2) - env2.env.actuator.Kd_np * u_j2

                _, tau2_k = env2.env.step_simulation(q_des2, dt_sim=0.002, u_des=None)
                Tp2 = env2.env.actuator.last_T_pos
                Tn2 = env2.env.actuator.last_T_neg
                T_act_p2 = env2.env.actuator.last_T_act_pos
                T_act_n2 = env2.env.actuator.last_T_act_neg
                tau2_pass_k = env2.env.actuator.last_tau_passive
                T_LV_p2 = env2.env.actuator.last_T_LV_pos
                T_LV_n2 = env2.env.actuator.last_T_LV_neg
                ex2_k = float(np.sum(np.maximum(0.0, tau2_k - Tp2) + np.maximum(0.0, -tau2_k - Tn2)))
                z2_k = float(env2.env.data.qpos[1])
                vz2_k = float(env2.env.data.qvel[1])
                if z2_k > env2.max_z:
                    env2.max_z = z2_k
                if vz2_k > env2.max_vz:
                    env2.max_vz = vz2_k
                if not env2.has_taken_off and z2_k < env2.min_z:
                    env2.min_z = z2_k
            else:
                tau2_pd_k = np.full(7, np.nan, dtype=np.float32)
                tau2_k = np.full(7, np.nan, dtype=np.float32)
                tau2_pass_k = np.full(7, np.nan, dtype=np.float32)
                Tp2 = np.full(7, np.nan, dtype=np.float32)
                Tn2 = np.full(7, np.nan, dtype=np.float32)
                T_act_p2 = np.full(7, np.nan, dtype=np.float32)
                T_act_n2 = np.full(7, np.nan, dtype=np.float32)
                T_LV_p2 = np.full(7, np.nan, dtype=np.float32)
                T_LV_n2 = np.full(7, np.nan, dtype=np.float32)
                ex2_k = np.nan
                z2_k = float(env2.max_z)

            t_500.append(substep_idx * 0.002)
            tau1_500.append(tau1_k.copy())
            tau2_pd_500.append(tau2_pd_k.copy())
            tau2_clamped_500.append(tau2_k.copy())
            tau2_passive_500.append(tau2_pass_k.copy())
            T_pos2_500.append(Tp2.copy())
            T_neg2_500.append(Tn2.copy())
            T_act_pos2_500.append(T_act_p2.copy())
            T_act_neg2_500.append(T_act_n2.copy())
            T_LV_pos2_500.append(T_LV_p2.copy())
            T_LV_neg2_500.append(T_LV_n2.copy())
            z1_500.append(z1_k)
            z2_500.append(z2_k)
            ex1_500.append(ex1_k)
            ex2_500.append(ex2_k)
            substep_idx += 1

        if not done1:
            env1.current_frame += env1.decimation
            env1.step_count += 1
            env1.prev_action = act1.copy()
            fl1, fr1, fz1 = env1._get_contacts_and_foot_z()
            if fz1 > env1.max_foot_z:
                env1.max_foot_z = fz1
            if fl1 < 1.0 and fr1 < 1.0 and env1.env.data.qpos[1] > env1.z_stand + 0.02:
                env1.has_taken_off = True
            if env1.has_taken_off and env1.env.data.qvel[1] <= 0.0:
                env1.apex_reached = True
                env1.post_apex_steps += 1
            obs1 = env1.get_obs()
            step1_survived = step + 1
            q1 = env1.env.data.qpos
            if q1[1] < 0.62 or abs(q1[2]) > 0.55 or abs(q1[0]) > 0.75:
                fell1 = True
                done1 = True
            elif env1.apex_reached and env1.post_apex_steps >= 2:
                q1_apex_pose = q1[:10].copy()
                done1 = True

        if not done2:
            env2.current_frame += env2.decimation
            env2.step_count += 1
            env2.prev_action = act2.copy()
            fl2, fr2, fz2 = env2._get_contacts_and_foot_z()
            if fz2 > env2.max_foot_z:
                env2.max_foot_z = fz2
            if fl2 < 1.0 and fr2 < 1.0 and env2.env.data.qpos[1] > env2.z_stand + 0.02:
                env2.has_taken_off = True
            if env2.has_taken_off and env2.env.data.qvel[1] <= 0.0:
                env2.apex_reached = True
                env2.post_apex_steps += 1
            obs2 = env2.get_obs()
            step2_survived = step + 1
            q2 = env2.env.data.qpos
            if q2[1] < 0.62 or abs(q2[2]) > 0.55 or abs(q2[0]) > 0.75:
                fell2 = True
                done2 = True
            elif env2.apex_reached and env2.post_apex_steps >= 2:
                q2_apex_pose = q2[:10].copy()
                done2 = True

        q1_vis = q1_apex_pose if q1_apex_pose is not None else env1.env.data.qpos[:10]
        q2_vis = q2_apex_pose if q2_apex_pose is not None else env2.env.data.qpos[:10]
        overlay_data.qpos[:10] = q1_vis
        overlay_data.qpos[10:20] = q2_vis
        mujoco.mj_forward(overlay_model, overlay_data)

        camera.lookat[0] = 0.0
        camera.lookat[1] = 0.0
        camera.lookat[2] = 0.90

        renderer.update_scene(overlay_data, camera)
        rgb = renderer.render()
        pil_img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_img)

        dz1_cm = (env1.max_z - env1.z_stand) * 100.0
        dz2_cm = (env2.max_z - env2.z_stand) * 100.0
        s1_str = f"z={q1_vis[1]:.2f}m (Max +{dz1_cm:.1f}cm)"
        s2_str = f"z={q2_vis[1]:.2f}m (Max +{dz2_cm:.1f}cm)"
        text_str = f"{exp_title} | Mode 1 Unlimited (Ghost): {s1_str} | Mode 2 MTG Clamped (Solid): {s2_str}"
        draw.rectangle([(15, 15), (945, 52)], fill=(15, 23, 42, 220), outline=(56, 189, 248), width=2)
        draw.text((25, 25), text_str, fill=(240, 249, 255))
        frames.append(np.array(pil_img))

        if done1 and done2:
            # Hold final apex comparison frame for 12 frames (0.4s) so the peak difference is clearly visible
            for _ in range(12):
                frames.append(np.array(pil_img))
            break

    imageio.mimsave(gif_path, frames, fps=25, loop=0)

    t_arr = np.array(t_500)
    tau1_arr = np.array(tau1_500)
    tau2_pd_arr = np.array(tau2_pd_500)
    tau2_clamped_arr = np.array(tau2_clamped_500)
    tau2_pass_arr = np.array(tau2_passive_500)
    Tp2_arr = np.array(T_pos2_500)
    Tn2_arr = np.array(T_neg2_500)
    T_act_p2_arr = np.array(T_act_pos2_500)
    T_act_n2_arr = np.array(T_act_neg2_500)
    T_LV_p2_arr = np.array(T_LV_pos2_500)
    T_LV_n2_arr = np.array(T_LV_neg2_500)
    ex1_arr = np.array(ex1_500)
    ex2_arr = np.array(ex2_500)

    valid2 = ~np.isnan(ex2_arr)
    clamped_mask = np.abs(tau2_pd_arr[valid2] - tau2_clamped_arr[valid2]) > 1e-3
    clamp_pct_per_joint = clamped_mask.mean(axis=0) * 100.0
    joints = ["Lumbar", "Hip (L)", "Knee (L)", "Ankle (L)", "Hip (R)", "Knee (R)", "Ankle (R)"]

    dz1_cm = (env1.max_z - env1.z_stand) * 100.0
    dz2_cm = (env2.max_z - env2.z_stand) * 100.0

    print(f"\n[DONE] Saved Overlay GIF: {gif_path}")
    print(f"  Mode 1 (Unlimited PD): Steps to Apex = {step1_survived} ({step1_survived*0.02:.2f}s) | Fell = {fell1} | Crouch min_z = {env1.min_z:.2f}m | Peak z = {env1.max_z:.2f}m (+{dz1_cm:.1f} cm) | Foot Clear = {env1.max_foot_z*100:.1f} cm | Peak Vz = {env1.max_vz:.2f} m/s | Mean Excess = {np.nanmean(ex1_arr):.2f} N*m | Max Excess = {np.nanmax(ex1_arr):.2f} N*m")
    print(f"  Mode 2 (MTG Clamped) : Steps to Apex = {step2_survived} ({step2_survived*0.02:.2f}s) | Fell = {fell2} | Crouch min_z = {env2.min_z:.2f}m | Peak z = {env2.max_z:.2f}m (+{dz2_cm:.1f} cm) | Foot Clear = {env2.max_foot_z*100:.1f} cm | Peak Vz = {env2.max_vz:.2f} m/s | Mean Excess = {np.nanmean(ex2_arr):.6f} N*m | Max Excess = {np.nanmax(ex2_arr):.6f} N*m")
    print("  Active Clamping Frequency per Joint in Mode 2 (% of 500 Hz sub-steps where |tau_act_des| > a(t)*T_LV):")
    for j_idx, j_name in enumerate(joints):
        max_clip = np.nanmax(np.abs(tau2_pd_arr[:, j_idx] - tau2_clamped_arr[:, j_idx]))
        max_pass = np.nanmax(np.abs(tau2_pass_arr[:, j_idx]))
        print(f"    - {j_name:<10}: {clamp_pct_per_joint[j_idx]:5.1f}% of sub-steps clamped (Max clipped: {max_clip:6.1f} N*m | Max |tau_pass|: {max_pass:5.1f} N*m)")

    fig, axes = plt.subplots(4, 2, figsize=(16, 13.5), sharex=True)
    axes = axes.flatten()

    for j in range(7):
        ax = axes[j]
        ax.plot(t_arr, T_LV_p2_arr[:, j], color="gray", linestyle="--", lw=0.8, alpha=0.45, zorder=1, label="Full-Activation Capacity T_max*f_L*f_V (a=1)")
        ax.plot(t_arr, -T_LV_n2_arr[:, j], color="gray", linestyle="--", lw=0.8, alpha=0.45, zorder=1)
        ax.fill_between(t_arr, -T_act_n2_arr[:, j], T_act_p2_arr[:, j], color="lightgray", alpha=0.60, label="Active Clamp Envelope [±a(t)*T_max*f_L*f_V]")
        ax.plot(t_arr, Tp2_arr[:, j], "k:", lw=1.2, alpha=0.9, zorder=5, label="Total MTG Limit [±a(t)*T_LV + tau_passive]")
        ax.plot(t_arr, -Tn2_arr[:, j], "k:", lw=1.2, alpha=0.9, zorder=5)

        ax.plot(t_arr, tau1_arr[:, j], "b--", lw=1.2, alpha=0.70, zorder=2, label="Mode 1 Applied Torque (Unlimited PD)")
        ax.plot(t_arr, tau2_pd_arr[:, j], color="darkorange", linestyle=":", lw=1.1, alpha=0.75, zorder=3, label="Mode 2 Unclamped PD Demand")
        ax.plot(t_arr, tau2_pass_arr[:, j], color="seagreen", linestyle="-", lw=1.2, alpha=0.85, zorder=3, label="Mode 2 Passive Torque tau_passive(q)")
        ax.plot(t_arr, tau2_clamped_arr[:, j], "r-", lw=1.8, zorder=4, label="Mode 2 Total Torque [tau_active + tau_passive]")

        ax.set_title(
            f"{joints[j]} Torque — {exp_title} (Clamped {clamp_pct_per_joint[j]:.1f}% of sub-steps)",
            fontsize=10.5, fontweight="bold"
        )
        ax.set_ylabel("Torque (N*m)")
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(loc="upper right", fontsize=6.5)

    ax8 = axes[7]
    ax8_twin = ax8.twinx()
    ax8.plot(t_arr, z1_500, "b--", lw=1.8, label=f"Mode 1 Pelvis Height (Peak {env1.max_z:.2f}m, +{dz1_cm:.1f}cm)")
    ax8.plot(t_arr, z2_500, "r-", lw=2.0, label=f"Mode 2 Pelvis Height (Peak {env2.max_z:.2f}m, +{dz2_cm:.1f}cm)")
    ax8.axhline(env1.z_stand, color="green", linestyle=":", lw=1.5, label=f"Standing Height ({env1.z_stand:.2f}m)")
    ax8.set_ylabel("Pelvis Height z (m)", color="black")
    ax8.set_ylim(0.55, max(1.45, env1.max_z + 0.15))

    ax8_twin.plot(t_arr, ex1_arr, color="darkorange", linestyle="-", lw=1.4, alpha=0.8, label="Mode 1 Exceeded Torque (N*m)")
    ax8_twin.plot(t_arr, ex2_arr, color="darkred", linestyle="-", lw=2.0, label="Mode 2 Exceeded Torque (= 0.00 N*m)")
    ax8_twin.set_ylabel("500 Hz Exceeded Torque (N*m)", color="darkorange")

    lines1, labels1 = ax8.get_legend_handles_labels()
    lines2, labels2 = ax8_twin.get_legend_handles_labels()
    ax8.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7.0)

    ax8.set_title(f"{exp_title}: Pelvis Trajectory z(t) & 500 Hz Exceeded Torque", fontsize=10.5, fontweight="bold")
    ax8.set_xlabel("Time (s)")
    ax8.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[DONE] Saved Torque & Limits Plot: {plot_path}")


# -----------------------------------------------------------------------------
# Main Suite Runner
# -----------------------------------------------------------------------------
def main():
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("renders", exist_ok=True)

    # =========================================================================
    # EXPERIMENT 1: WALKING IMITATION (~1.25 m/s)
    # =========================================================================
    print("\n" + "#" * 78)
    print("   EXPERIMENT 1: WALKING IMITATION (Arnold2013Walking_subject01_walk2.npz)")
    print("#" * 78)
    clip_walk = MotionClip.load("motions/Arnold2013Walking_subject01_walk2.npz")
    walk_ref_vx = float(np.mean(clip_walk.u[:, 0]))

    ckpt_walk_m1 = "checkpoints/exp1_walk_mode1.pt"
    ckpt_walk_m2 = "checkpoints/exp1_walk_mode2.pt"

    policy_walk_m1 = StableTrackingPolicy(obs_dim=64, action_dim=7, device=device)
    if os.path.exists(ckpt_walk_m1):
        policy_walk_m1.load_state_dict(torch.load(ckpt_walk_m1, map_location=device))
        print(f"  [Loaded Exp 1 Mode 1 from {ckpt_walk_m1}]")
    else:
        env_w1 = VectorStableEnv(clip_walk, num_envs=64, mode="mode1")
        policy_walk_m1 = train_ppo_loop(env_w1, policy_walk_m1, num_iters=40, lr=2.5e-4, label="Exp1-Walk-M1")
        torch.save(policy_walk_m1.state_dict(), ckpt_walk_m1)

    policy_walk_m2 = StableTrackingPolicy(obs_dim=64, action_dim=7, device=device)
    if os.path.exists(ckpt_walk_m2):
        policy_walk_m2.load_state_dict(torch.load(ckpt_walk_m2, map_location=device))
        print(f"  [Loaded Exp 1 Mode 2 from {ckpt_walk_m2}]")
    else:
        policy_walk_m2.load_state_dict(policy_walk_m1.state_dict())
        print("  --- Exp 1 Mode 2 Phase 1: Soft MTG Penalty (15 iters) ---")
        env_w2_soft = VectorStableEnv(clip_walk, num_envs=64, mode="mode2", mtg_phase="soft")
        policy_walk_m2 = train_ppo_loop(env_w2_soft, policy_walk_m2, num_iters=15, lr=1.5e-4, label="Exp1-Walk-M2-Soft")
        print("  --- Exp 1 Mode 2 Phase 2: Hard 500 Hz MTG Clamping (25 iters) ---")
        env_w2_hard = VectorStableEnv(clip_walk, num_envs=64, mode="mode2", mtg_phase="hard")
        policy_walk_m2 = train_ppo_loop(env_w2_hard, policy_walk_m2, num_iters=25, lr=1.0e-4, label="Exp1-Walk-M2-Hard")
        torch.save(policy_walk_m2.state_dict(), ckpt_walk_m2)

    evaluate_500hz_experiment(
        exp_title="EXP 1: WALKING (1.25 m/s)",
        ref_speed=walk_ref_vx,
        clip=clip_walk,
        policy1=policy_walk_m1,
        policy2=policy_walk_m2,
        gif_path="renders/exp1_walking_overlay.gif",
        plot_path="renders/exp1_walking_torque_limits.png",
        max_steps=180,
        env_cls=StableTrackingEnv
    )

    # =========================================================================
    # EXPERIMENT 2: SLOW RUNNING IMITATION (2.0 m/s)
    # =========================================================================
    print("\n" + "#" * 78)
    print("   EXPERIMENT 2: SLOW RUNNING (2.0 m/s, Hamner2013_Run20_cycle.npz)")
    print("#" * 78)
    clip_run20 = MotionClip.load("motions/Hamner2013_Run20_cycle.npz")
    run20_ref_vx = float(np.mean(clip_run20.u[:, 0]))

    ckpt_run20_m1 = "checkpoints/exp2_run20_mode1.pt"
    ckpt_run20_m2 = "checkpoints/exp2_run20_mode2.pt"

    policy_run20_m1 = StableTrackingPolicy(obs_dim=64, action_dim=7, device=device)
    if os.path.exists(ckpt_run20_m1):
        policy_run20_m1.load_state_dict(torch.load(ckpt_run20_m1, map_location=device))
        print(f"  [Loaded Exp 2 Mode 1 from {ckpt_run20_m1}]")
    else:
        env_r20_1 = VectorStableEnv(clip_run20, num_envs=64, mode="mode1")
        policy_run20_m1 = train_ppo_loop(env_r20_1, policy_run20_m1, num_iters=50, lr=2.5e-4, label="Exp2-Run20-M1")
        torch.save(policy_run20_m1.state_dict(), ckpt_run20_m1)

    policy_run20_m2 = StableTrackingPolicy(obs_dim=64, action_dim=7, device=device)
    if os.path.exists(ckpt_run20_m2):
        policy_run20_m2.load_state_dict(torch.load(ckpt_run20_m2, map_location=device))
        print(f"  [Loaded Exp 2 Mode 2 from {ckpt_run20_m2}]")
    else:
        policy_run20_m2.load_state_dict(policy_run20_m1.state_dict())
        print("  --- Exp 2 Mode 2 Phase 1: Soft MTG Penalty (15 iters) ---")
        env_r20_soft = VectorStableEnv(clip_run20, num_envs=64, mode="mode2", mtg_phase="soft")
        policy_run20_m2 = train_ppo_loop(env_r20_soft, policy_run20_m2, num_iters=15, lr=1.5e-4, label="Exp2-Run20-M2-Soft")
        print("  --- Exp 2 Mode 2 Phase 2: Hard 500 Hz MTG Clamping (25 iters) ---")
        env_r20_hard = VectorStableEnv(clip_run20, num_envs=64, mode="mode2", mtg_phase="hard")
        policy_run20_m2 = train_ppo_loop(env_r20_hard, policy_run20_m2, num_iters=25, lr=1.0e-4, label="Exp2-Run20-M2-Hard")
        torch.save(policy_run20_m2.state_dict(), ckpt_run20_m2)

    evaluate_500hz_experiment(
        exp_title="EXP 2: SLOW RUNNING (2.0 m/s)",
        ref_speed=run20_ref_vx,
        clip=clip_run20,
        policy1=policy_run20_m1,
        policy2=policy_run20_m2,
        gif_path="renders/exp2_slow_running_2ms_overlay.gif",
        plot_path="renders/exp2_slow_running_2ms_torque_limits.png",
        max_steps=180,
        env_cls=StableTrackingEnv
    )

    # =========================================================================
    # EXPERIMENT 3: FAST RUNNING IMITATION (5.0 m/s)
    # =========================================================================
    print("\n" + "#" * 78)
    print("   EXPERIMENT 3: FAST RUNNING (5.0 m/s, Hamner2013_Run50_cycle.npz)")
    print("#" * 78)
    clip_run50 = load_calibrated_run50_clip("motions/Hamner2013_Run50_cycle.npz")
    run50_ref_vx = float(np.mean(clip_run50.u[:, 0]))

    ckpt_run50_m1 = "checkpoints/exp3_run50_mode1.pt"
    ckpt_run50_m2 = "checkpoints/exp3_run50_mode2.pt"

    policy_run50_m1 = make_sprint_policy(obs_dim=64, device=device, warm_start_path=ckpt_run50_m1)
    policy_run50_m2 = make_sprint_policy(obs_dim=64, device=device, warm_start_path=ckpt_run50_m2)

    evaluate_500hz_experiment(
        exp_title="EXP 3: FAST RUNNING (5.0 m/s)",
        ref_speed=run50_ref_vx,
        clip=clip_run50,
        policy1=policy_run50_m1,
        policy2=policy_run50_m2,
        gif_path="renders/exp3_fast_running_5ms_overlay.gif",
        plot_path="renders/exp3_fast_running_5ms_torque_limits.png",
        max_steps=180,
        env_cls=SprintTrackingEnv
    )

    # =========================================================================
    # EXPERIMENT 4: MAXIMUM HEIGHT JUMP (PURE MOTION GENERATION, NO REFERENCE)
    # =========================================================================
    print("\n" + "#" * 78)
    print("   EXPERIMENT 4: MAXIMUM HEIGHT JUMP (Pure Motion Generation via RL)")
    print("#" * 78)

    ckpt_jump_m1 = "checkpoints/exp4_jump_mode1.pt"
    ckpt_jump_m2 = "checkpoints/exp4_jump_mode2.pt"

    policy_jump_m1 = make_jump_policy(obs_dim=64, device=device)
    if os.path.exists(ckpt_jump_m1):
        policy_jump_m1.load_state_dict(torch.load(ckpt_jump_m1, map_location=device))
        print(f"  [Loaded Exp 4 Mode 1 from {ckpt_jump_m1}]")
    else:
        env_j1 = VectorJumpEnv(num_envs=64, mode="mode1", max_steps=60)
        policy_jump_m1 = train_ppo_loop(env_j1, policy_jump_m1, num_iters=45, lr=3.0e-4, label="Exp4-Jump-M1")
        torch.save(policy_jump_m1.state_dict(), ckpt_jump_m1)

    policy_jump_m2 = make_jump_policy(obs_dim=64, device=device)
    if os.path.exists(ckpt_jump_m2):
        policy_jump_m2.load_state_dict(torch.load(ckpt_jump_m2, map_location=device))
        print(f"  [Loaded Exp 4 Mode 2 from {ckpt_jump_m2}]")
    else:
        policy_jump_m2.load_state_dict(policy_jump_m1.state_dict())
        with torch.no_grad():
            policy_jump_m2.log_std.clamp_(min=-1.5, max=-1.1)
        print("  --- Exp 4 Mode 2 Phase 1: Soft MTG Penalty (15 iters) ---")
        env_j2_soft = VectorJumpEnv(num_envs=64, mode="mode2", mtg_phase="soft", max_steps=60)
        policy_jump_m2 = train_ppo_loop(env_j2_soft, policy_jump_m2, num_iters=15, lr=1.5e-4, label="Exp4-Jump-M2-Soft")
        print("  --- Exp 4 Mode 2 Phase 2: Hard 500 Hz MTG Clamping (35 iters) ---")
        env_j2_hard = VectorJumpEnv(num_envs=64, mode="mode2", mtg_phase="hard", max_steps=60)
        policy_jump_m2 = train_ppo_loop(env_j2_hard, policy_jump_m2, num_iters=35, lr=1.5e-4, label="Exp4-Jump-M2-Hard")
        torch.save(policy_jump_m2.state_dict(), ckpt_jump_m2)

    evaluate_500hz_jump_experiment(
        exp_title="EXP 4: MAX HEIGHT JUMP",
        policy1=policy_jump_m1,
        policy2=policy_jump_m2,
        gif_path="renders/exp4_max_jump_overlay.gif",
        plot_path="renders/exp4_max_jump_torque_limits.png",
        max_steps=60
    )

    print("\n" + "=" * 78)
    print("   ALL 4 EXPERIMENTS (WALKING, 2 m/s RUN, 5 m/s RUN, MAX JUMP) COMPLETE!")
    print("=" * 78)


if __name__ == "__main__":
    main()

