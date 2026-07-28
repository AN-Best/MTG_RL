import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# Find the latest event file in runs/biped_amp/
log_dir = "runs/biped_amp"
event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))

if not event_files:
    # Check top-level runs directory
    event_files = glob.glob("runs/events.out.tfevents.*")

if not event_files:
    print("No TensorBoard event files found!")
    exit(1)

# Sort by modification time to get the newest run
latest_event_file = max(event_files, key=os.path.getmtime)
print(f"Parsing latest event file: {latest_event_file}")

# Load event accumulator
ea = event_accumulator.EventAccumulator(
    latest_event_file,
    size_guidance={
        event_accumulator.SCALARS: 0,  # Load all scalars
    }
)
ea.Reload()

# Check available tags
tags = ea.Tags().get('scalars', [])
print("\nAvailable Scalar Tags:")
for tag in tags:
    print(f"  - {tag}")

# Parse rewards and losses
metrics = {
    "Reward": "Reward / Total reward (mean)",
    "P-Loss": "Loss / Policy loss",
    "V-Loss": "Loss / Value loss",
    "D-Loss": "Loss / Discriminator loss"
}

data = {}
for name, tag in metrics.items():
    if tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[name] = (steps, values)
        print(f"Loaded {len(values)} points for {name}. Final value: {values[-1]:.4f}")
    else:
        # Fallback to look for substrings
        matching_tags = [t for t in tags if name.lower() in t.lower() or (name == "Reward" and "reward" in t.lower())]
        if matching_tags:
            tag = matching_tags[0]
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            data[name] = (steps, values)
            print(f"Loaded {len(values)} points for {name} (matched tag: {tag}). Final value: {values[-1]:.4f}")
        else:
            print(f"Metric {name} not found in event file.")

# Plot the training curves
if data:
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.ravel()
    
    for i, (name, (steps, values)) in enumerate(data.items()):
        axs[i].plot(steps, values, label=name)
        axs[i].set_title(name)
        axs[i].set_xlabel("Steps")
        axs[i].grid(True)
        
    plt.tight_layout()
    plt.savefig("runs/biped_amp/training_curves.png")
    print("\nSaved training curves plot to runs/biped_amp/training_curves.png")
else:
    print("No metric data could be extracted.")
