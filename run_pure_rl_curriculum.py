import os
import sys
import subprocess
import time
import argparse

parser = argparse.ArgumentParser(description="Run 3-Stage Pure RL Walking Curriculum across Actuator Tiers")
parser.add_argument("--motion", type=str, default="motions/Arnold2013Walking_subject01_walk2.npz", help="Reference motion file (overrides --dataset if provided)")
parser.add_argument("--dataset", type=str, default="motions", help="Dataset directory containing pure walking trials (default: motions)")
parser.add_argument("--eval-motion", type=str, default="Arnold2013Walking_subject01_walk2", help="Evaluation clip name (default: Arnold2013Walking_subject01_walk2)")
parser.add_argument("--s1-iters", type=int, default=120, help="Stage 1 iterations (ideal_unlimited, default: 120)")
parser.add_argument("--s2-iters", type=int, default=60, help="Stage 2 iterations (ideal_limited, default: 60)")
parser.add_argument("--s3-iters", type=int, default=70, help="Stage 3 iterations (mtg, default: 70)")
parser.add_argument("--steps", type=int, default=4096, help="Rollout steps per iteration (default: 4096)")
parser.add_argument("--num-envs", type=int, default=64, help="Parallel environments on GPU (default: 64)")
parser.add_argument("--batch-size", type=int, default=512, help="PPO mini-batch size on GPU (default: 512)")
parser.add_argument("--eval-steps", type=int, default=150, help="Evaluation rollout steps at 50 Hz (default: 150 = 3.0s)")
args = parser.parse_args()

python_exe = sys.executable
script_path = os.path.join(os.path.dirname(__file__), "Universal_Motion_Tracker.py")

if args.motion is not None:
    motion_args = ["--motion", args.motion]
    motion_tag = os.path.splitext(os.path.basename(args.motion))[0]
    motion_desc = f"Single Motion: {args.motion}"
else:
    motion_args = ["--dataset", args.dataset]
    motion_tag = "dataset"
    motion_desc = f"Full Dataset: {args.dataset} ({len(os.listdir(args.dataset))} pure walking trials)"

print("==========================================================================")
print("   LAUNCHING PURE END-TO-END WALKING RL CURRICULUM ON RTX 5060 GPU")
print("==========================================================================")
print(f"  - Library: {motion_desc}")
print(f"  - Parallel Vector Envs: {args.num_envs} on GPU")
print(f"  - Transitions Per Iter: {args.steps} (Mini-Batch: {args.batch_size})")
print("  - Observation Space: 62-dim (Includes cyclic gait phase [sin 2pi*phi, cos 2pi*phi])")
print("  - Action Filter: Low-latency responsive (0.40 persist / 0.60 innovation)")
print("  - Cartesian Tracking: Pelvis-relative foot sites (translation-invariant)")
print("  - Zero Heuristics: 100% direct neural RL authority")
print("  - Push Perturbations: ACTIVE (+/- 30 N horizontal pelvis pushes)")
print(f"  - Curriculum: S1 ({args.s1_iters} iters) -> S2 ({args.s2_iters} iters) -> S3 ({args.s3_iters} iters)")
print("==========================================================================\n")

s1_ckpt = os.path.join("checkpoints", f"policy_universal_tracker_{motion_tag}_ideal_unlimited_latest.pt")
s2_ckpt = os.path.join("checkpoints", f"policy_universal_tracker_{motion_tag}_ideal_limited_latest.pt")
s3_ckpt = os.path.join("checkpoints", f"policy_universal_tracker_{motion_tag}_mtg_latest.pt")

# STAGE 1: Ideal Unlimited (from scratch)
print(f"\n>>> STARTING STAGE 1: IDEAL_UNLIMITED ({args.s1_iters} iters from scratch) <<<")
cmd_s1 = [
    python_exe, script_path,
    *motion_args,
    "--actuator", "ideal_unlimited",
    "--iters", str(args.s1_iters),
    "--steps", str(args.steps),
    "--batch-size", str(args.batch_size),
    "--num-envs", str(args.num_envs),
    "--eval-motion", args.eval_motion,
    "--eval-steps", str(args.eval_steps),
    "--eval-phase", "0"
]
t0 = time.time()
res1 = subprocess.run(cmd_s1)
print(f"Stage 1 finished in {time.time() - t0:.1f}s with exit code {res1.returncode}")
if res1.returncode != 0:
    sys.exit(1)

# STAGE 2: Ideal Limited (warmstarted from Stage 1)
print(f"\n>>> STARTING STAGE 2: IDEAL_LIMITED ({args.s2_iters} iters warmstarted from Stage 1) <<<")
cmd_s2 = [
    python_exe, script_path,
    *motion_args,
    "--actuator", "ideal_limited",
    "--warmstart-from", s1_ckpt,
    "--iters", str(args.s2_iters),
    "--steps", str(args.steps),
    "--batch-size", str(args.batch_size),
    "--num-envs", str(args.num_envs),
    "--eval-motion", args.eval_motion,
    "--eval-steps", str(args.eval_steps),
    "--eval-phase", "0"
]
t0 = time.time()
res2 = subprocess.run(cmd_s2)
print(f"Stage 2 finished in {time.time() - t0:.1f}s with exit code {res2.returncode}")
if res2.returncode != 0:
    sys.exit(1)

# STAGE 3: Full MTG (warmstarted from Stage 2)
print(f"\n>>> STARTING STAGE 3: FULL MTG ({args.s3_iters} iters warmstarted from Stage 2) <<<")
cmd_s3 = [
    python_exe, script_path,
    *motion_args,
    "--actuator", "mtg",
    "--warmstart-from", s2_ckpt,
    "--iters", str(args.s3_iters),
    "--steps", str(args.steps),
    "--batch-size", str(args.batch_size),
    "--num-envs", str(args.num_envs),
    "--eval-motion", args.eval_motion,
    "--eval-steps", str(args.eval_steps),
    "--eval-phase", "0"
]
t0 = time.time()
res3 = subprocess.run(cmd_s3)
print(f"Stage 3 finished in {time.time() - t0:.1f}s with exit code {res3.returncode}")
if res3.returncode != 0:
    sys.exit(1)

print("\n==========================================================================")
print("   PURE WALKING RL CURRICULUM COMPLETE ACROSS ALL 3 ACTUATOR TIERS!")
print(f"   Final MTG Checkpoint: {s3_ckpt}")
print("==========================================================================")
