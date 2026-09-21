import os
import numpy as np
import mujoco

def recalibrate_all_motions(motion_dir="motions"):
    xml_path = "biped_10dof.xml"
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)

    files = sorted([f for f in os.listdir(motion_dir) if f.endswith(".npz")])
    print(f"Recalibrating {len(files)} motions in '{motion_dir}' with exact foot capsule geometry...")

    dt = 0.002
    fixed_count = 0

    foot_l_body = mj_model.body("foot_l").id
    foot_r_body = mj_model.body("foot_r").id

    for f in files:
        path = os.path.join(motion_dir, f)
        data = np.load(path)
        q = data["q"].copy()
        t = data["t"].copy()
        N = len(t)

        z_calib = np.zeros(N, dtype=np.float32)
        for i in range(N):
            mj_data.qpos[:] = 0.0
            mj_data.qpos[1] = 0.90
            mj_data.qpos[2:10] = q[i, 2:10]
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
            # Exactly 2mm clearance so foot never collides or penetrates prematurely
            z_calib[i] = 0.90 - lowest_surface + 0.002

        # Gentle Savitzky-Golay / moving-average smoothing with lower-bound protection
        kernel_size = 15 # 30ms window
        pad_len = kernel_size // 2
        z_padded = np.pad(z_calib, pad_len, mode='edge')
        z_conv = np.convolve(z_padded, np.ones(kernel_size)/kernel_size, mode='valid')
        # Ensure smooth trajectory never dips below physical contact floor
        z_smooth = np.maximum(z_conv, z_calib - 0.001)

        q[:, 1] = z_smooth
        u = np.gradient(q, dt, axis=0)

        np.savez_compressed(path, t=t, q=q, u=u)
        fixed_count += 1
        if fixed_count % 10 == 0 or fixed_count == len(files):
            print(f"  [{fixed_count:02d}/{len(files)}] {f:<38} | z: [{q[:, 1].min():.3f}, {q[:, 1].max():.3f}]")

    print("\nAll motion files successfully recalibrated with exact capsule clearance!")

if __name__ == "__main__":
    recalibrate_all_motions()
