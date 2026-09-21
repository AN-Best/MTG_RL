# 10-DoF Musculoskeletal Biped — MuJoCo Physics & MTG Actuators

A high-performance **MuJoCo** framework for a 10-Degree-of-Freedom (10-DoF) Musculoskeletal Biped with custom PyTorch Muscle Torque Generators (MTG) and Imitation Learning.

---

## 📁 Clean Repository Structure

```
MTG_SymPy/
├── biped_10dof.xml              # 10-DoF MuJoCo kinematic, mass & contact model (Z-up, human knee axes)
├── Biped_Env.py                 # Core MuJoCo environment class wrapping biped_10dof.xml
├── Biped_Actuators.py           # PyTorch Actuator models (Ideal Unlimited, Ideal Limited, Hill MTG)
├── Run_Drop_Test.py             # Passive freefall drop test verification & HD rendering
├── Run_Actuator_Benchmark.py    # Reference motion tracking benchmark across 3 actuator levels
├── Run_Train_Imitation.py       # PPO Imitation Learning trainer & policy evaluator
├── subject01_walk1_ik.mot       # OpenSim kinematic gait dataset
└── requirements.txt             # Python dependencies
```

---

## 🚀 Quickstart Usage

### 1. Run Passive Drop Test
```bash
python Run_Drop_Test.py
```
Outputs `drop_test.gif`, `drop_test.mp4`, and `drop_test.png`.

### 2. Benchmark Actuator Levels (Unlimited, Limited, MTG)
```bash
python Run_Actuator_Benchmark.py
```
Outputs comparative animations (`biped_mujoco_ideal_unlimited.gif`, `biped_mujoco_ideal_limited.gif`, `biped_mujoco_mtg.gif`) and diagnostic plot `actuators_comparison.png`.

### 3. Train Imitation Learning Policy (PPO)
```bash
python Run_Train_Imitation.py
```
Trains PPO imitation learning policy and outputs `imitation_ideal_unlimited.gif` and policy checkpoint `checkpoints/policy_ideal_unlimited.pt`.
