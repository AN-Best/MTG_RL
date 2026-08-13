import sys
import torch
sys.path.insert(0, '.')
from Biped_Gym_Env import BipedAMPEnv
from skrl.envs.wrappers.torch import wrap_env
from Biped_Train_AMP import Policy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env = BipedAMPEnv(num_envs=1, device=device)
env = wrap_env(env)

policy = Policy(env.observation_space, env.action_space, device)
policy.to(device)

try:
    checkpoint = torch.load("amp_biped_policy.pt", map_location=device)
    if isinstance(checkpoint, dict) and "policy" in checkpoint:
        policy.load_state_dict(checkpoint["policy"])
    else:
        policy.load_state_dict(checkpoint)
    print("Successfully loaded policy.")
except Exception as e:
    print(f"Could not load policy: {e}")
    exit(1)

policy.eval()
obs, info = env.reset()

print("\n--- Running Diagnostic Rollout ---")
for step in range(100):
    with torch.no_grad():
        inputs = {"states": obs}
        actions, _ = policy.act(inputs, role="policy")
        
    obs, reward, terminated, truncated, info = env.step(actions)
    
    # Extract raw env states
    q = env.unwrapped.q[0] if env.unwrapped.q.ndim > 1 else env.unwrapped.q
    u = env.unwrapped.u[0] if env.unwrapped.u.ndim > 1 else env.unwrapped.u
    
    torso_angle = q[2].item()
    hip_height = q[1].item()
    vel_x = u[0].item()
    max_joint_vel = torch.max(torch.abs(u)).item()
    
    print(f"Step {step:2d} | Hip Height: {hip_height:.3f} m | Torso Angle: {torso_angle:+.3f} rad | Hip Vel: {vel_x:+.3f} m/s | Max Joint Vel: {max_joint_vel:.3f} rad/s")
    
    if terminated[0] or truncated[0]:
        print("\n--- Rollout Terminated! Check reasons: ---")
        print(f"  - Torso angle limit exceeded (> 0.78 rad): {abs(torso_angle) > 0.78} (value: {torso_angle:.3f})")
        print(f"  - Hip height limit exceeded (< 0.60 m): {hip_height < 0.60} (value: {hip_height:.3f})")
        print(f"  - Horizontal velocity exceeded (> 5.0 m/s): {abs(vel_x) > 5.0} (value: {vel_x:.3f})")
        print(f"  - Joint velocity limit exceeded (> 50.0 rad/s): {max_joint_vel > 50.0} (value: {max_joint_vel:.3f})")
        break
