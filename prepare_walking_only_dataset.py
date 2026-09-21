import os
import urllib.request
import json
import numpy as np
import mujoco

def prepare_pure_walking_dataset(output_dir="motions_walking"):
    os.makedirs(output_dir, exist_ok=True)

    print("==========================================================================")
    print("   CURATING PURE HUMAN WALKING DATASET (ADDBIOMECHANICS)")
    print("==========================================================================")

    # 1. Query GitHub API for all walking IK files
    api_url = "https://api.github.com/repos/stanfordnmbl/addbiomechanics-paper/git/trees/main?recursive=1"
    req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"Error fetching repo tree: {e}")
        return []

    tree = data.get('tree', [])
    walk_paths = [
        f['path'] for f in tree 
        if f['path'].endswith('_ik.mot') and ('walk' in f['path'].lower()) and not ('run' in f['path'].lower() or 'squat' in f['path'].lower())
    ]
    print(f"Found {len(walk_paths)} dedicated walking IK trials in repository.")

    # 2. Setup MuJoCo forward kinematics model for contact calibration
    xml_path = os.path.join(os.path.dirname(__file__), "biped_10dof.xml")
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    sole_l_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "contact_p1_l")
    sole_r_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "contact_p1_r")

    base_raw_url = "https://raw.githubusercontent.com/stanfordnmbl/addbiomechanics-paper/main"
    saved_npzs = []
    dt = 0.002

    for path in walk_paths:
        parts = path.split('/')
        dataset_name = parts[0]
        subject = parts[2]
        trial = parts[-1].replace('_ik.mot', '')
        name = f"{dataset_name}_{subject}_{trial}"
        
        npz_path = os.path.join(output_dir, f"{name}.npz")
        raw_mot = os.path.join(output_dir, f"{name}.mot")

        # Skip if already downloaded and valid
        if os.path.exists(npz_path):
            try:
                d = np.load(npz_path)
                if len(d["t"]) > 50:
                    saved_npzs.append(npz_path)
                    print(f"  [Cached] {name} ({len(d['t'])} frames)")
                    continue
            except Exception:
                pass

        url = f"{base_raw_url}/{path}"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=12) as resp, open(raw_mot, 'wb') as f:
                f.write(resp.read())
        except Exception as e:
            print(f"  [!] Failed download for {name}: {e}")
            continue

        # Parse .mot file
        try:
            with open(raw_mot, 'r') as f:
                lines = f.readlines()
        except Exception:
            if os.path.exists(raw_mot): os.remove(raw_mot)
            continue

        in_degrees = True
        header_idx = 0
        for i, line in enumerate(lines):
            if 'inDegrees=no' in line:
                in_degrees = False
            if 'endheader' in line.lower():
                header_idx = i + 1
                break

        if header_idx >= len(lines):
            if os.path.exists(raw_mot): os.remove(raw_mot)
            continue

        cols = lines[header_idx].strip().split()
        data = []
        for line in lines[header_idx+1:]:
            if line.strip():
                try:
                    data.append([float(x) for x in line.strip().split()])
                except ValueError:
                    pass

        arr = np.array(data)
        if len(arr) < 10:
            if os.path.exists(raw_mot): os.remove(raw_mot)
            continue

        t_raw = arr[:, cols.index('time')]
        t_start, t_end = t_raw[0], t_raw[-1]
        t_uniform = np.arange(t_start, t_end, dt)
        N = len(t_uniform)
        if N < 50:
            if os.path.exists(raw_mot): os.remove(raw_mot)
            continue

        def get_col(col_name, default=0.0):
            if col_name in cols:
                raw_vals = arr[:, cols.index(col_name)]
                val = np.interp(t_uniform, t_raw, raw_vals)
                if in_degrees and ('rotation' in col_name.lower() or 'angle' in col_name.lower() or 'flex' in col_name.lower()):
                    val = np.radians(val)
                return val
            return np.full(N, default)

        # 10 DoFs: [x, z, pitch, lumbar, hip_l, knee_l, ankle_l, hip_r, knee_r, ankle_r]
        # OpenSim sign convention corrections:
        #   pelvis_tilt: positive = posterior tilt (backward lean) -> negate for MuJoCo (+pitch = forward)
        #   lumbar_extension: positive = extension (backward bend) -> negate for MuJoCo (+lumbar = forward)
        lumbar = -get_col('lumbar_extension', 0.0)
        pelvis_pitch = -get_col('pelvis_tilt', 0.0)
        hip_l = get_col('hip_flexion_l', 0.0)
        hip_r = get_col('hip_flexion_r', 0.0)
        knee_l = -abs(get_col('knee_angle_l', 0.0))  # negative is knee flexion backward
        knee_r = -abs(get_col('knee_angle_r', 0.0))
        ankle_l = get_col('ankle_angle_l', 0.0)
        ankle_r = get_col('ankle_angle_r', 0.0)

        # Compute overground forward velocity from pelvis_tx or time
        pelvis_tx = get_col('pelvis_tx', 0.0)
        if np.ptp(pelvis_tx) > 0.3:
            x_traj = pelvis_tx - pelvis_tx[0]
        else:
            # Treadmill walk: assign standard natural walking speed (1.20 m/s for normal, 0.9 for TS1, 1.4 for TS3)
            walk_speed = 1.20
            if "TS1" in trial: walk_speed = 0.90
            elif "TS3" in trial: walk_speed = 1.45
            x_traj = (t_uniform - t_uniform[0]) * walk_speed

        # Forward kinematics to ground-calibrate pelvis height z
        q_arr = np.zeros((N, 10), dtype=np.float32)
        q_arr[:, 0] = x_traj
        q_arr[:, 2] = pelvis_pitch
        q_arr[:, 3] = lumbar
        q_arr[:, 4] = hip_l
        q_arr[:, 5] = knee_l
        q_arr[:, 6] = ankle_l
        q_arr[:, 7] = hip_r
        q_arr[:, 8] = knee_r
        q_arr[:, 9] = ankle_r

        # Calibrate height so lowest foot sole capsule surface touches z = 0.002
        z_calib = np.zeros(N, dtype=np.float32)
        foot_l_body = mj_model.body("foot_l").id
        foot_r_body = mj_model.body("foot_r").id

        for i in range(N):
            mj_data.qpos[:] = 0.0
            mj_data.qpos[1] = 0.90 # nominal
            mj_data.qpos[2:10] = q_arr[i, 2:10]
            mujoco.mj_forward(mj_model, mj_data)

            fl_pos = mj_data.xpos[foot_l_body]
            fl_mat = mj_data.xmat[foot_l_body].reshape(3, 3)
            heel_l = (fl_pos + fl_mat @ np.array([-0.07, 0, 0]))[2] - 0.02
            toe_l = (fl_pos + fl_mat @ np.array([0.17, 0, 0]))[2] - 0.02

            fr_pos = mj_data.xpos[foot_r_body]
            fr_mat = mj_data.xmat[foot_r_body].reshape(3, 3)
            heel_r = (fr_pos + fr_mat @ np.array([-0.07, 0, 0]))[2] - 0.02
            toe_r = (fr_pos + fr_mat @ np.array([0.17, 0, 0]))[2] - 0.02

            lowest_surface = min(heel_l, toe_l, heel_r, toe_r)
            z_calib[i] = 0.90 - lowest_surface + 0.002

        # Smooth with lower bound protection against ground collision
        kernel_size = 15
        pad_len = kernel_size // 2
        z_padded = np.pad(z_calib, pad_len, mode='edge')
        z_conv = np.convolve(z_padded, np.ones(kernel_size)/kernel_size, mode='valid')
        z_smooth = np.maximum(z_conv, z_calib - 0.001)
        q_arr[:, 1] = z_smooth

        # Compute smooth velocities
        u_arr = np.gradient(q_arr, dt, axis=0)

        # Save .npz
        np.savez_compressed(npz_path, t=t_uniform - t_uniform[0], q=q_arr, u=u_arr)
        if os.path.exists(raw_mot): os.remove(raw_mot)
        saved_npzs.append(npz_path)
        print(f"  [Saved] {name:<35} | {N*dt:.2f}s ({N} frames)")

    print(f"\n==========================================================================")
    print(f"   PURE WALKING DATASET READY: {len(saved_npzs)} motions in '{output_dir}'")
    print(f"==========================================================================")
    return saved_npzs

if __name__ == "__main__":
    prepare_pure_walking_dataset("motions_walking")
