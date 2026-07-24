"""
Smoke test: verifies the full training loop API works end-to-end
using only 4 envs and 2 update steps (fast, ~30s).
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np

from Biped_Gym_Env import BipedAMPEnv
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.amp import AMP, AMP_CFG
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ---------- tiny env ----------
NUM_ENVS = 4
env_raw = BipedAMPEnv(num_envs=NUM_ENVS, device=device)
env = wrap_env(env_raw)

# ---------- models ----------
class Policy(GaussianMixin, Model):
    def __init__(self, obs_sp, act_sp, dev):
        Model.__init__(self, observation_space=obs_sp, action_space=act_sp, device=dev)
        GaussianMixin.__init__(self, role="policy", clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2)
        self.net = nn.Sequential(nn.Linear(self.num_observations, 64), nn.ELU(), nn.Linear(64, self.num_actions))
        self.log_std = nn.Parameter(torch.zeros(self.num_actions))
    def compute(self, inputs, role):
        s = inputs.get("states") if inputs.get("states") is not None else inputs.get("observations")
        return self.net(s), {"log_std": self.log_std}

class Value(DeterministicMixin, Model):
    def __init__(self, obs_sp, act_sp, dev):
        Model.__init__(self, observation_space=obs_sp, action_space=act_sp, device=dev)
        DeterministicMixin.__init__(self, role="value")
        self.net = nn.Sequential(nn.Linear(self.num_observations, 64), nn.ELU(), nn.Linear(64, 1))
    def compute(self, inputs, role):
        s = inputs.get("states") if inputs.get("states") is not None else inputs.get("observations")
        return self.net(s), {}

class Discriminator(DeterministicMixin, Model):
    def __init__(self, obs_sp, act_sp, dev):
        Model.__init__(self, observation_space=obs_sp, action_space=act_sp, device=dev)
        DeterministicMixin.__init__(self, role="discriminator")
        self.net = nn.Sequential(nn.Linear(self.num_observations, 64), nn.ELU(), nn.Linear(64, 1))
    def compute(self, inputs, role):
        s = inputs.get("states") if inputs.get("states") is not None else inputs.get("observations")
        return self.net(s), {}

models = {
    "policy": Policy(env.observation_space, env.action_space, device),
    "value": Value(env.observation_space, env.action_space, device),
    "discriminator": Discriminator(env.observation_space, env.action_space, device),
}

# ---------- cfg ----------
cfg = AMP_CFG()
cfg.rollouts = 4
cfg.learning_epochs = 2
cfg.mini_batches = 1
cfg.learning_rate = 5e-5
cfg.state_preprocessor = None
cfg.experiment.directory = "runs"
cfg.experiment.experiment_name = "smoke_test"
cfg.experiment.write_interval = 999999
cfg.experiment.checkpoint_interval = 999999

ROLLOUTS = cfg.rollouts
TOTAL    = NUM_ENVS * ROLLOUTS * 2  # 2 updates only

memory       = RandomMemory(memory_size=ROLLOUTS, device=device, num_envs=NUM_ENVS)
reply_buffer = RandomMemory(memory_size=1000, device=device, num_envs=1, replacement=True)
motion_dataset = RandomMemory(memory_size=1000, device=device, replacement=True)

# Simple synthetic ref motion (17D obs)
ref_obs = torch.randn(1000, env_raw.num_observations, device=device)

def collect_reference_motions(n):
    idx = torch.randint(0, len(ref_obs), (n,), device=device)
    return ref_obs[idx]

agent = AMP(
    models=models, memory=memory,
    observation_space=env.observation_space, action_space=env.action_space,
    device=device, cfg=cfg,
    amp_observation_space=env.observation_space,
    motion_dataset=motion_dataset, reply_buffer=reply_buffer,
    collect_reference_motions=collect_reference_motions,
)

# ---------- training loop ----------
print("\nRunning 2-update smoke test...")
timestep = 0
update   = 0

states, infos = env.reset()
agent_states  = states

while timestep < TOTAL:
    agent.pre_interaction(timestep=timestep, timesteps=TOTAL)

    for _ in range(ROLLOUTS):
        with torch.no_grad():
            actions, _ = agent.act(states, agent_states, timestep=timestep, timesteps=TOTAL)

        next_states, rewards, terminated, truncated, infos = env.step(actions)

        agent.record_transition(
            observations=states, states=agent_states,
            actions=actions, rewards=rewards,
            next_observations=next_states, next_states=next_states,
            terminated=terminated, truncated=truncated,
            infos=infos, timestep=timestep, timesteps=TOTAL,
        )

        done = terminated | truncated
        if done.any():
            reset_obs, _ = env.reset()
            next_states = next_states.clone()
            next_states[done] = reset_obs[done]

        states       = next_states
        agent_states = next_states
        timestep    += NUM_ENVS

    agent.post_interaction(timestep=timestep, timesteps=TOTAL)
    update += 1
    print(f"  Update {update} done (timestep={timestep})")

print("\n✓ SMOKE TEST PASSED: all API calls succeeded without error")
