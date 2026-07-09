import os
import glob
import torch
from Biped_Gym_Env import BipedAMPEnv
from skrl.envs.wrappers.torch import wrap_env
from Biped_Train_AMP import Policy

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize environment
print("Initializing environment...")
env = BipedAMPEnv(num_envs=1, device=device)
env = wrap_env(env)

# Initialize policy
print("Loading trained policy...")
policy = Policy(env.observation_space, env.action_space, device)
policy.to(device)

# Load the saved state dict from agent checkpoint
try:
    checkpoint = torch.load("amp_biped_policy.pt", map_location=device)
    if isinstance(checkpoint, dict) and "policy" in checkpoint:
        policy.load_state_dict(checkpoint["policy"])
    else:
        policy.load_state_dict(checkpoint)
    print("Successfully loaded trained policy from amp_biped_policy.pt!")
except Exception as e:
    print(f"Could not load amp_biped_policy.pt directly: {e}")
    print("Attempting to load from the checkpoints folder...")
    
    checkpoint_path = "runs/biped_amp/checkpoints/best_agent.pt"
    if not os.path.exists(checkpoint_path):
        pts = glob.glob("runs/biped_amp/checkpoints/*.pt")
        if pts:
            checkpoint_path = max(pts, key=os.path.getmtime)
            print(f"best_agent.pt not found. Falling back to the latest checkpoint: {checkpoint_path}")
        else:
            raise FileNotFoundError("No checkpoint files found in runs/biped_amp/checkpoints/")
            
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "policy" in checkpoint:
        policy.load_state_dict(checkpoint["policy"])
    else:
        policy.load_state_dict(checkpoint)
    print(f"Successfully loaded trained policy from {checkpoint_path}!")

policy.eval()

# Reset environment
obs, info = env.reset()

print("\n--- Running Policy Evaluation ---")
print("Press Ctrl+C in your terminal to stop.")
try:
    for step in range(1000):
        # Predict actions
        with torch.no_grad():
            inputs = {"states": obs}
            # gaussian model returns (mean, log_std, extra) in act()
            actions, _ = policy.act(inputs, role="policy")
            
        # Step environment
        obs, reward, terminated, truncated, info = env.step(actions)
        
        # Print states
        torso_angle = obs[0, 1].item()
        print(f"Step {step:03d} | Torso Angle: {torso_angle:6.3f} rad | Reward: {reward[0].item():6.3f}")
        
        if terminated[0] or truncated[0]:
            print("Episode ended (Biped fell or time limit). Resetting...")
            obs, info = env.reset()
except KeyboardInterrupt:
    print("Evaluation stopped by user.")
