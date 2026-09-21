import os
import numpy as np
import torch
import mujoco

from Biped_Actuators import create_actuator

class BipedEnv:
    """
    MuJoCo 10-DoF Musculoskeletal Biped Environment supporting 3 Actuator Tiers:
    1. 'ideal_unlimited': Direct joint torque PD control without limits.
    2. 'ideal_limited': Direct joint torque PD control scaled & saturated to peak isometric limits T_max.
    3. 'mtg': Full Hill-type Muscle Torque Generator with 1st-order activation ODEs,
              force-length, force-velocity, and sensory-motor reflex dynamics.
    """
    def __init__(self, actuator_type="ideal_unlimited", device="cpu"):
        self.actuator_type = actuator_type
        self.device = torch.device(device)
        
        # Load MuJoCo XML from workspace root
        xml_path = os.path.join(os.path.dirname(__file__), "biped_10dof.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Instantiate PyTorch Actuator Model
        self.actuator = create_actuator(actuator_type=actuator_type, device=self.device)
        self.actuator_state = None
        
        # Renderer setup
        self.renderer = None
        self.camera = None
        
    def reset(self, qpos_init=None, qvel_init=None):
        """Resets simulation state to nominal initial posture or specified state."""
        mujoco.mj_resetData(self.model, self.data)
        
        if qpos_init is not None:
            self.data.qpos[:] = np.asarray(qpos_init, dtype=np.float64)
        else:
            # Default initial standing posture: z = 0.92m
            self.data.qpos[0] = 0.0    # x
            self.data.qpos[1] = 0.92   # z
            self.data.qpos[2] = 0.0    # pitch
            self.data.qpos[3] = 0.035  # q_lum
            self.data.qpos[4] = -0.05  # q_lh
            self.data.qpos[5] = -0.05  # q_lk
            self.data.qpos[6] = 0.05   # q_la
            self.data.qpos[7] = -0.05  # q_rh
            self.data.qpos[8] = -0.05  # q_rk
            self.data.qpos[9] = 0.05   # q_ra

        if qvel_init is not None:
            self.data.qvel[:] = np.asarray(qvel_init, dtype=np.float64)
        else:
            self.data.qvel[:] = 0.0

        # Reset actuator internal muscle activation state (if MTG, nominal resting tone = 0.15)
        if self.actuator_type in ["mtg", "muscle", "tier3"]:
            self.actuator_state = torch.full((1, 7), 0.15, dtype=torch.float32, device=self.device)
        else:
            self.actuator_state = None
            
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def _get_obs(self):
        """Returns 19-dim observation array: [z, p, 7 joint angles, 10 velocities]."""
        qpos = self.data.qpos.copy() # 10 elements: [x, z, p, q_lum, 6 leg joints]
        qvel = self.data.qvel.copy() # 10 elements: [vx, vz, vp, dq_lum, 6 leg velocities]
        return np.concatenate([qpos[1:], qvel], dtype=np.float32)

    def step_simulation(self, q_des, dt_sim=0.002, u_des=None):
        """
        Executes a single physics timestep (dt_sim = 0.002s) using PyTorch actuator torques.
        
        Args:
            q_des: Target joint angles array or tensor [7] (rad)
            dt_sim: Timestep in seconds (default 0.002s)
            u_des: Target joint velocities array or tensor [7] (rad/s) or None
        """
        if isinstance(q_des, np.ndarray):
            q_des_t = torch.tensor(q_des, dtype=torch.float32, device=self.device).unsqueeze(0)
        elif isinstance(q_des, torch.Tensor):
            q_des_t = q_des.to(self.device)
            if q_des_t.dim() == 1:
                q_des_t = q_des_t.unsqueeze(0)
        else:
            raise TypeError("q_des must be a numpy array or torch Tensor")

        u_des_t = None
        if u_des is not None:
            if isinstance(u_des, np.ndarray):
                u_des_t = torch.tensor(u_des, dtype=torch.float32, device=self.device).unsqueeze(0)
            elif isinstance(u_des, torch.Tensor):
                u_des_t = u_des.to(self.device)
                if u_des_t.dim() == 1:
                    u_des_t = u_des_t.unsqueeze(0)

        # Current 7 joint angles & 7 joint velocities
        q_j = torch.tensor(self.data.qpos[3:10], dtype=torch.float32, device=self.device).unsqueeze(0)
        u_j = torch.tensor(self.data.qvel[3:10], dtype=torch.float32, device=self.device).unsqueeze(0)

        # Compute torques via PyTorch actuator model (Unlimited, Limited, or MTG)
        tau_total, self.actuator_state = self.actuator.compute_torques(
            q_j, u_j, q_des_t, self.actuator_state, dt_sim=dt_sim, u_des=u_des_t
        )

        # Apply torques to MuJoCo direct motor actuators
        tau_np = tau_total.squeeze(0).detach().cpu().numpy()
        self.data.ctrl[:] = tau_np

        # Step MuJoCo LCP physics
        mujoco.mj_step(self.model, self.data)
        
        return self._get_obs(), tau_np

    def render_frame(self, height=720, width=1280, camera_dist=3.2, camera_elev=-12.0, camera_azim=65.0):
        """Renders offscreen RGB frame with camera view."""
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=height, width=width)
            self.camera = mujoco.MjvCamera()
            self.camera.distance = camera_dist
            self.camera.elevation = camera_elev
            self.camera.azimuth = camera_azim

        self.camera.lookat = np.array([self.data.qpos[0], 0.0, 0.6])
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render()
