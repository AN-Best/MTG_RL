import sympy as sm
import sympy.physics.mechanics as me
from sympy import S, Abs, exp, symbols
import torch
import numpy as np

# Dict for symbol creation options
sym_kwargs = {'real': True}

# Define contact parameters globally so they are shared
kc, cc = symbols('kc, cc', **sym_kwargs)
mu, vs = symbols('mu, vs', **sym_kwargs)

def contact_force(point, ground, origin, belt_speed=S(0), stiffness_exp=2):
    """Returns a contact force vector acting on the given point made of
    friction along the contact surface and elastic force in the vertical
    direction.

    Parameters
    ==========
    point : sympy.physics.mechanics.Point
        The point which the contact force should be computed for.
    ground : sympy.physics.mechanics.ReferenceFrame
        A reference frame which represents the inertial ground in 2D space.
        The x axis defines the ground line and positive y is up.
    origin : sympy.physics.mechanics.Point
        An origin point located on the ground line.
    belt_speed : sympifiable, i.e. Symbol, Function(), number, etc.
        A variable or value that represents the possibly time varying belt
        speed for treadmill walking.
    stiffness_exp : float, optional
        Exponent of the stiffness force.

    Returns
    =======
    force : sympy.physics.mechanics.Vector
        The contact force between the point and the ground.

    """
    # This is the "height" of the point above the ground, where a negative
    # value means that the point is below the ground.
    y_location = point.pos_from(origin).dot(ground.y)

    # The penetration into the ground is mathematically defined as:
    #
    #               { 0 if y_location > 0
    # deformation = {
    #               { abs(y_location) if y_location < 0
    #

    penetration = (Abs(y_location) - y_location) / 2

    velocity = point.vel(ground)

    # The addition of "- y_location" here adds a small linear term to the
    # cubic stiffness and creates a light attractive force towards the
    # ground. This is in place to ensure that gradients can be computed for
    # the optimization used in Ackermann and van den Bogert 2010.
    vertical_force = (kc*penetration**stiffness_exp -
                      y_location)*(1 - cc*velocity.dot(ground.y))

    # Friction force depends on velocity of the contact point relative to the
    # treadmill belt (which is moving backwards).
    friction = (-mu*vertical_force*((2/(1 + exp(
        (-belt_speed - velocity.dot(ground.x))/vs))) - 1))

    return friction * ground.x + vertical_force * ground.y

# -----------------------------------------------------------------------------
# 1. Setup & State Variables
# -----------------------------------------------------------------------------
# Generalized coordinates (Translation: X, Y; Angles: Torso, Left Hip/Knee/Ankle, Right Hip/Knee/Ankle)
q_x, q_y = me.dynamicsymbols('q_x q_y')
q_tr = me.dynamicsymbols('q_tr')
q_lh, q_lk, q_la = me.dynamicsymbols('q_lh q_lk q_la')
q_rh, q_rk, q_ra = me.dynamicsymbols('q_rh q_rk q_ra')

# Generalized speeds (Translational velocities; Angular velocities)
u_x, u_y = me.dynamicsymbols('u_x u_y')
u_tr = me.dynamicsymbols('u_tr')
u_lh, u_lk, u_la = me.dynamicsymbols('u_lh u_lk u_la')
u_rh, u_rk, u_ra = me.dynamicsymbols('u_rh u_rk u_ra')

# Kinematic differential equations: dq/dt = u
kdes = [
    q_x.diff() - u_x,
    q_y.diff() - u_y,
    q_tr.diff() - u_tr,
    q_lh.diff() - u_lh,
    q_lk.diff() - u_lk,
    q_la.diff() - u_la,
    q_rh.diff() - u_rh,
    q_rk.diff() - u_rk,
    q_ra.diff() - u_ra
]

# -----------------------------------------------------------------------------
# 2. Parameters
# -----------------------------------------------------------------------------
# Segment masses
m_tr = sm.symbols('m_torso')
m_t, m_l, m_f = sm.symbols('m_thigh m_leg m_foot')

# Segment lengths
l_tr = sm.symbols('l_torso')
l_t, l_l, l_f = sm.symbols('l_thigh l_leg l_foot')
l_heel, l_toe = sm.symbols('l_heel l_toe')

# COM distances (from proximal joint to COM)
c_tr = sm.symbols('c_torso')
c_t, c_l, c_f = sm.symbols('c_thigh c_leg c_foot')

# Moments of inertia (Sagittal plane -> rotation about Z-axis)
I_trz = sm.symbols('I_torso_z')
I_tz, I_lz, I_fz = sm.symbols('I_thigh_z I_leg_z I_foot_z')

# Gravity and Active Torques
g, h_g = sm.symbols('g h_g')
tau_lh, tau_lk, tau_la = me.dynamicsymbols('tau_lh tau_lk tau_la')
tau_rh, tau_rk, tau_ra = me.dynamicsymbols('tau_rh tau_rk tau_ra')

# -----------------------------------------------------------------------------
# 3. Reference Frames
# -----------------------------------------------------------------------------
# N = World Frame
N = me.ReferenceFrame('N')

# Tr = Torso Frame (Rotates about N.z by q_tr)
Tr = N.orientnew('Tr', 'Axis', [q_tr, N.z])
Tr.set_ang_vel(N, u_tr * N.z)

# Left Leg Frames
T_l = Tr.orientnew('T_l', 'Axis', [q_lh, Tr.z])
T_l.set_ang_vel(Tr, u_lh * Tr.z)

L_l = T_l.orientnew('L_l', 'Axis', [q_lk, T_l.z])
L_l.set_ang_vel(T_l, u_lk * T_l.z)

F_l = L_l.orientnew('F_l', 'Axis', [q_la, L_l.z])
F_l.set_ang_vel(L_l, u_la * L_l.z)

# Right Leg Frames
T_r = Tr.orientnew('T_r', 'Axis', [q_rh, Tr.z])
T_r.set_ang_vel(Tr, u_rh * Tr.z)

L_r = T_r.orientnew('L_r', 'Axis', [q_rk, T_r.z])
L_r.set_ang_vel(T_r, u_rk * T_r.z)

F_r = L_r.orientnew('F_r', 'Axis', [q_ra, L_r.z])
F_r.set_ang_vel(L_r, u_ra * L_r.z)

# -----------------------------------------------------------------------------
# 4. Points & Kinematics
# -----------------------------------------------------------------------------
# Ground Origin (fixed in world N)
O = me.Point('O')
O.set_vel(N, 0)

# Hip Joint Center (translates relative to O)
HJC = O.locatenew('HJC', q_x * N.x + q_y * N.y)
HJC.set_vel(N, u_x * N.x + u_y * N.y)

# Torso COM (extends superiorly along +Tr.y)
Tr_com = HJC.locatenew('Tr_com', c_tr * Tr.y)
Tr_com.v2pt_theory(HJC, N, Tr)

# Left Leg Kinematics
T_l_com = HJC.locatenew('T_l_com', -c_t * T_l.y)
T_l_com.v2pt_theory(HJC, N, T_l)

KJC_l = HJC.locatenew('KJC_l', -l_t * T_l.y)
KJC_l.v2pt_theory(HJC, N, T_l)

L_l_com = KJC_l.locatenew('L_l_com', -c_l * L_l.y)
L_l_com.v2pt_theory(KJC_l, N, L_l)

AJC_l = KJC_l.locatenew('AJC_l', -l_l * L_l.y)
AJC_l.v2pt_theory(KJC_l, N, L_l)

F_l_com = AJC_l.locatenew('F_l_com', c_f * F_l.x) 
F_l_com.v2pt_theory(AJC_l, N, F_l)

# Left Foot Contact Points
Heel_l = AJC_l.locatenew('Heel_l', -l_heel * F_l.x)
Heel_l.v2pt_theory(AJC_l, N, F_l)

Toe_l = AJC_l.locatenew('Toe_l', l_toe * F_l.x)
Toe_l.v2pt_theory(AJC_l, N, F_l)

# Right Leg Kinematics
T_r_com = HJC.locatenew('T_r_com', -c_t * T_r.y)
T_r_com.v2pt_theory(HJC, N, T_r)

KJC_r = HJC.locatenew('KJC_r', -l_t * T_r.y)
KJC_r.v2pt_theory(HJC, N, T_r)

L_r_com = KJC_r.locatenew('L_r_com', -c_l * L_r.y)
L_r_com.v2pt_theory(KJC_r, N, L_r)

AJC_r = KJC_r.locatenew('AJC_r', -l_l * L_r.y)
AJC_r.v2pt_theory(KJC_r, N, L_r)

F_r_com = AJC_r.locatenew('F_r_com', c_f * F_r.x) 
F_r_com.v2pt_theory(AJC_r, N, F_r)

# Right Foot Contact Points
Heel_r = AJC_r.locatenew('Heel_r', -l_heel * F_r.x)
Heel_r.v2pt_theory(AJC_r, N, F_r)

Toe_r = AJC_r.locatenew('Toe_r', l_toe * F_r.x)
Toe_r.v2pt_theory(AJC_r, N, F_r)

# -----------------------------------------------------------------------------
# 5. Inertia & Rigid Bodies
# -----------------------------------------------------------------------------
# Central Inertias
I_torso   = me.inertia(Tr, 0, 0, I_trz)
I_thigh_l = me.inertia(T_l, 0, 0, I_tz)
I_leg_l   = me.inertia(L_l, 0, 0, I_lz)
I_foot_l  = me.inertia(F_l, 0, 0, I_fz)

I_thigh_r = me.inertia(T_r, 0, 0, I_tz)
I_leg_r   = me.inertia(L_r, 0, 0, I_lz)
I_foot_r  = me.inertia(F_r, 0, 0, I_fz)

# Create Rigid Bodies
Torso   = me.RigidBody('Torso', Tr_com, Tr, m_tr, (I_torso, Tr_com))
Thigh_l = me.RigidBody('Thigh_l', T_l_com, T_l, m_t, (I_thigh_l, T_l_com))
Leg_l   = me.RigidBody('Leg_l', L_l_com, L_l, m_l, (I_leg_l, L_l_com))
Foot_l  = me.RigidBody('Foot_l', F_l_com, F_l, m_f, (I_foot_l, F_l_com))

Thigh_r = me.RigidBody('Thigh_r', T_r_com, T_r, m_t, (I_thigh_r, T_r_com))
Leg_r   = me.RigidBody('Leg_r', L_r_com, L_r, m_l, (I_leg_r, L_r_com))
Foot_r  = me.RigidBody('Foot_r', F_r_com, F_r, m_f, (I_foot_r, F_r_com))

bodies = [Torso, Thigh_l, Leg_l, Foot_l, Thigh_r, Leg_r, Foot_r]

# -----------------------------------------------------------------------------
# 6. Forces & Torques
# -----------------------------------------------------------------------------
# Gravity applied to COMs
forces = [
    (Tr_com, -m_tr * g * N.y),
    (T_l_com, -m_t * g * N.y),
    (L_l_com, -m_l * g * N.y),
    (F_l_com, -m_f * g * N.y),
    (T_r_com, -m_t * g * N.y),
    (L_r_com, -m_l * g * N.y),
    (F_r_com, -m_f * g * N.y)
]

# Joint torques applied between frames
forces.extend([
    # Left leg torques
    (T_l, tau_lh * Tr.z), (Tr, -tau_lh * Tr.z),  # Hip torque
    (L_l, tau_lk * T_l.z), (T_l, -tau_lk * T_l.z),  # Knee torque
    (F_l, tau_la * L_l.z), (L_l, -tau_la * L_l.z),  # Ankle torque
    # Right leg torques
    (T_r, tau_rh * Tr.z), (Tr, -tau_rh * Tr.z),  # Hip torque
    (L_r, tau_rk * T_r.z), (T_r, -tau_rk * T_r.z),  # Knee torque
    (F_r, tau_ra * L_r.z), (L_r, -tau_ra * L_r.z)   # Ankle torque
])

# Contact forces applied to the heel and toe of both feet
forces.extend([
    (Heel_l, contact_force(Heel_l, N, O)),
    (Toe_l, contact_force(Toe_l, N, O)),
    (Heel_r, contact_force(Heel_r, N, O)),
    (Toe_r, contact_force(Toe_r, N, O))
])

# -----------------------------------------------------------------------------
# 7. Equations of Motion (Kane's Method)
# -----------------------------------------------------------------------------
kane = me.KanesMethod(
    N,
    q_ind=[q_x, q_y, q_tr, q_lh, q_lk, q_la, q_rh, q_rk, q_ra],
    u_ind=[u_x, u_y, u_tr, u_lh, u_lk, u_la, u_rh, u_rk, u_ra],
    kd_eqs=kdes
)
fr, frstar = kane.kanes_equations(bodies, forces)

# The Mass Matrix (M) and Forcing Vector (F)
M = kane.mass_matrix_full
F_vec = kane.forcing_full

# -----------------------------------------------------------------------------
# 8. Compile to a Callable Function (lambdify with PyTorch)
# -----------------------------------------------------------------------------
q_vars = [q_x, q_y, q_tr, q_lh, q_lk, q_la, q_rh, q_rk, q_ra]
u_vars = [u_x, u_y, u_tr, u_lh, u_lk, u_la, u_rh, u_rk, u_ra]
torque_vars = [tau_lh, tau_lk, tau_la, tau_rh, tau_rk, tau_ra]
parameter_vars = [
    m_tr, m_t, m_l, m_f, 
    l_tr, l_t, l_l, l_f, l_heel, l_toe, 
    c_tr, c_t, c_l, c_f, 
    I_trz, I_tz, I_lz, I_fz, 
    g, h_g, 
    kc, cc, mu, vs
]

# Flatten matrices for 130x faster compilation & evaluation
M_flat = [M[i, j] for i in range(18) for j in range(18)]
F_flat = [F_vec[i, 0] for i in range(18)]

print("Compiling equations of motion to PyTorch functions... (This may take a moment)")
M_torch_func = sm.lambdify((q_vars, u_vars, torque_vars, parameter_vars), M_flat, modules='torch')
F_torch_func = sm.lambdify((q_vars, u_vars, torque_vars, parameter_vars), F_flat, modules='torch')

def skeleton_dynamics(q_val, u_val, torques_val, params_val):
    """Computes the state derivative dx/dt = [dq/dt, du/dt] for the biped skeleton.
    Evaluates equations of motion on CPU (to avoid massive CUDA launch latency),
    then transfers stacked matrices to GPU to solve the linear system.
    """
    device = q_val.device
    
    # Move inputs to CPU for fast equation evaluation
    q_cpu = q_val.to(device="cpu", dtype=torch.float32)
    u_cpu = u_val.to(device="cpu", dtype=torch.float32)
    torques_cpu = torques_val.to(device="cpu", dtype=torch.float32)
    params_cpu = params_val.to(device="cpu", dtype=torch.float32)
    
    is_batched = q_cpu.ndim > 1
    
    if is_batched:
        q_seq = [q_cpu[:, i] for i in range(9)]
        u_seq = [u_cpu[:, i] for i in range(9)]
        torques_seq = [torques_cpu[:, i] for i in range(6)]
        params_seq = [params_cpu[:, i] for i in range(24)]
        
        M_num = M_torch_func(q_seq, u_seq, torques_seq, params_seq)
        F_num = F_torch_func(q_seq, u_seq, torques_seq, params_seq)
        
        # Fast CPU broadcast
        M_num_tensors = [val if isinstance(val, torch.Tensor) else torch.tensor(val, device="cpu") for val in M_num]
        F_num_tensors = [val if isinstance(val, torch.Tensor) else torch.tensor(val, device="cpu") for val in F_num]
        
        M_num_expanded = torch.broadcast_tensors(*M_num_tensors)
        F_num_expanded = torch.broadcast_tensors(*F_num_tensors)
        
        M_mat_cpu = torch.stack(M_num_expanded, dim=0).view(18, 18, -1).permute(2, 0, 1)
        F_mat_cpu = torch.stack(F_num_expanded, dim=0).t().unsqueeze(-1)
        
        # Copy to GPU in a single batch transfer
        M_mat = M_mat_cpu.to(device)
        F_mat = F_mat_cpu.to(device)
        
        # Solve in double precision to prevent float32 ill-conditioning
        M_mat_double = M_mat.double()
        F_mat_double = F_mat.double()
        eye = torch.eye(18, device=device, dtype=torch.float64)
        dx_dt = torch.linalg.solve(M_mat_double + 1e-6 * eye, F_mat_double).float().squeeze(-1)
    else:
        q_list = list(q_cpu)
        u_list = list(u_cpu)
        torques_list = list(torques_cpu)
        params_list = list(params_cpu)
        
        M_num = M_torch_func(q_list, u_list, torques_list, params_list)
        F_num = F_torch_func(q_list, u_list, torques_list, params_list)
        
        M_num_tensor = [torch.tensor(val, dtype=torch.float32, device="cpu") if not isinstance(val, torch.Tensor) else val for val in M_num]
        F_num_tensor = [torch.tensor(val, dtype=torch.float32, device="cpu") if not isinstance(val, torch.Tensor) else val for val in F_num]
        
        M_mat_cpu = torch.stack(M_num_tensor, dim=0).view(18, 18)
        F_mat_cpu = torch.stack(F_num_tensor, dim=0).unsqueeze(-1)
        
        # Copy to GPU
        M_mat = M_mat_cpu.to(device)
        F_mat = F_mat_cpu.to(device)
        
        # Solve in double precision to prevent float32 ill-conditioning
        M_mat_double = M_mat.double()
        F_mat_double = F_mat.double()
        eye = torch.eye(18, device=device, dtype=torch.float64)
        try:
            dx_dt = torch.linalg.solve(M_mat_double + 1e-6 * eye, F_mat_double).float().squeeze(-1)
        except Exception as e:
            print("\n=== SOLVER FAILED ===")
            print("M_mat contains NaN:", torch.isnan(M_mat).any().item())
            print("M_mat contains inf:", torch.isinf(M_mat).any().item())
            print("q_list:", [val.item() if isinstance(val, torch.Tensor) else val for val in q_list])
            raise e
        
    return dx_dt[..., :9], dx_dt[..., 9:]

print("Equations of motion successfully compiled!")
print("skeleton_dynamics is ready to use with PyTorch.")