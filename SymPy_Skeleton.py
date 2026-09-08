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

    # Hunt-Crossley contact model:
    # 1. Penetration delta = max(0, -y_location)
    # 2. Damping is proportional to penetration delta^1.5
    # 3. Ground force is non-negative (max(0, ...)) to prevent sticky ground tension on lift-off
    pen_raw = (Abs(y_location) - y_location) / 2
    penetration = (Abs(pen_raw) + pen_raw) / 2
    velocity = point.vel(ground)
    vy = velocity.dot(ground.y)
    damping_term = (1 - cc * vy)
    # Smooth non-negative Hunt-Crossley contact force
    vertical_force = kc * (penetration**1.5) * ((Abs(damping_term) + damping_term) / 2)

    # Friction force depends on velocity of the contact point relative to ground belt
    vx_rel = velocity.dot(ground.x) + belt_speed
    friction = -mu * vertical_force * ((2 / (1 + exp(-vx_rel / vs))) - 1)

    return friction * ground.x + vertical_force * ground.y

# -----------------------------------------------------------------------------
# 1. Setup & State Variables (10-DoF)
# -----------------------------------------------------------------------------
# Generalized coordinates
# Translation: X, Y; Angles: Pelvis Tilt, Lumbar Extension, Left Hip/Knee/Ankle, Right Hip/Knee/Ankle
q_x, q_y, q_p, q_lum = me.dynamicsymbols('q_x q_y q_p q_lum')
q_lh, q_lk, q_la = me.dynamicsymbols('q_lh q_lk q_la')
q_rh, q_rk, q_ra = me.dynamicsymbols('q_rh q_rk q_ra')

# Generalized speeds
u_x, u_y, u_p, u_lum = me.dynamicsymbols('u_x u_y u_p u_lum')
u_lh, u_lk, u_la = me.dynamicsymbols('u_lh u_lk u_la')
u_rh, u_rk, u_ra = me.dynamicsymbols('u_rh u_rk u_ra')

# Kinematic differential equations: dq/dt = u
kdes = [
    q_x.diff() - u_x,
    q_y.diff() - u_y,
    q_p.diff() - u_p,
    q_lum.diff() - u_lum,
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
# Segment masses (Pelvis: 10.65 kg, Torso: 25.0 kg - Nasr et al. 2024)
m_p, m_tr = sm.symbols('m_pelvis m_torso')
m_t, m_l, m_f = sm.symbols('m_thigh m_leg m_foot')

# Segment lengths & offsets
l_p, l_tr = sm.symbols('l_pelvis l_torso')
l_t, l_l, l_f = sm.symbols('l_thigh l_leg l_foot')
l_heel, l_toe = sm.symbols('l_heel l_toe')

c_p, c_tr = sm.symbols('c_pelvis c_torso')
c_t, c_l, c_f = sm.symbols('c_thigh c_leg c_foot')

# Moments of inertia (Sagittal plane -> rotation about Z-axis)
I_pz, I_trz = sm.symbols('I_pelvis_z I_torso_z')
I_tz, I_lz, I_fz = sm.symbols('I_thigh_z I_leg_z I_foot_z')

# Gravity and Active Torques (7 actuators: 1 lumbar + 6 leg joints)
g, h_g = sm.symbols('g h_g')
tau_lum = me.dynamicsymbols('tau_lum')
tau_lh, tau_lk, tau_la = me.dynamicsymbols('tau_lh tau_lk tau_la')
tau_rh, tau_rk, tau_ra = me.dynamicsymbols('tau_rh tau_rk tau_ra')

# -----------------------------------------------------------------------------
# 3. Reference Frames
# -----------------------------------------------------------------------------
# N = World Frame
N = me.ReferenceFrame('N')

# Pelvis Frame (Rotates about N.z by q_p)
Pel = N.orientnew('Pel', 'Axis', [q_p, N.z])
Pel.set_ang_vel(N, u_p * N.z)

# Torso Frame (Rotates about Pel.z by q_lum)
Tr = Pel.orientnew('Tr', 'Axis', [q_lum, Pel.z])
Tr.set_ang_vel(Pel, u_lum * Pel.z)

# Left Leg Frames (relative to Pelvis)
T_l = Pel.orientnew('T_l', 'Axis', [q_lh, Pel.z])
T_l.set_ang_vel(Pel, u_lh * Pel.z)

L_l = T_l.orientnew('L_l', 'Axis', [q_lk, T_l.z])
L_l.set_ang_vel(T_l, u_lk * T_l.z)

F_l = L_l.orientnew('F_l', 'Axis', [q_la, L_l.z])
F_l.set_ang_vel(L_l, u_la * L_l.z)

# Right Leg Frames (relative to Pelvis)
T_r = Pel.orientnew('T_r', 'Axis', [q_rh, Pel.z])
T_r.set_ang_vel(Pel, u_rh * Pel.z)

L_r = T_r.orientnew('L_r', 'Axis', [q_rk, T_r.z])
L_r.set_ang_vel(T_r, u_rk * T_r.z)

F_r = L_r.orientnew('F_r', 'Axis', [q_ra, L_r.z])
F_r.set_ang_vel(L_r, u_ra * L_r.z)

# -----------------------------------------------------------------------------
# 4. Points & Kinematics
# -----------------------------------------------------------------------------
# Ground Origin
O = me.Point('O')
O.set_vel(N, 0)

# Hip Joint Center
HJC = O.locatenew('HJC', q_x * N.x + q_y * N.y)
HJC.set_vel(N, u_x * N.x + u_y * N.y)

# Pelvis COM
Pel_com = HJC.locatenew('Pel_com', c_p * Pel.y)
Pel_com.v2pt_theory(HJC, N, Pel)

# Lumbar Joint Center
LJC = HJC.locatenew('LJC', l_p * Pel.y)
LJC.v2pt_theory(HJC, N, Pel)

# Torso COM (extends superiorly from LJC along +Tr.y)
Tr_com = LJC.locatenew('Tr_com', c_tr * Tr.y)
Tr_com.v2pt_theory(LJC, N, Tr)

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

Heel_r = AJC_r.locatenew('Heel_r', -l_heel * F_r.x)
Heel_r.v2pt_theory(AJC_r, N, F_r)

Toe_r = AJC_r.locatenew('Toe_r', l_toe * F_r.x)
Toe_r.v2pt_theory(AJC_r, N, F_r)

# -----------------------------------------------------------------------------
# 5. Inertia & Rigid Bodies
# -----------------------------------------------------------------------------
I_pelvis  = me.inertia(Pel, 0, 0, I_pz)
I_torso   = me.inertia(Tr, 0, 0, I_trz)
I_thigh_l = me.inertia(T_l, 0, 0, I_tz)
I_leg_l   = me.inertia(L_l, 0, 0, I_lz)
I_foot_l  = me.inertia(F_l, 0, 0, I_fz)

I_thigh_r = me.inertia(T_r, 0, 0, I_tz)
I_leg_r   = me.inertia(L_r, 0, 0, I_lz)
I_foot_r  = me.inertia(F_r, 0, 0, I_fz)

Pelvis  = me.RigidBody('Pelvis', Pel_com, Pel, m_p, (I_pelvis, Pel_com))
Torso   = me.RigidBody('Torso', Tr_com, Tr, m_tr, (I_torso, Tr_com))
Thigh_l = me.RigidBody('Thigh_l', T_l_com, T_l, m_t, (I_thigh_l, T_l_com))
Leg_l   = me.RigidBody('Leg_l', L_l_com, L_l, m_l, (I_leg_l, L_l_com))
Foot_l  = me.RigidBody('Foot_l', F_l_com, F_l, m_f, (I_foot_l, F_l_com))

Thigh_r = me.RigidBody('Thigh_r', T_r_com, T_r, m_t, (I_thigh_r, T_r_com))
Leg_r   = me.RigidBody('Leg_r', L_r_com, L_r, m_l, (I_leg_r, L_r_com))
Foot_r  = me.RigidBody('Foot_r', F_r_com, F_r, m_f, (I_foot_r, F_r_com))

bodies = [Pelvis, Torso, Thigh_l, Leg_l, Foot_l, Thigh_r, Leg_r, Foot_r]

# -----------------------------------------------------------------------------
# 6. Forces & Torques
# -----------------------------------------------------------------------------
forces = [
    (Pel_com, -m_p * g * N.y),
    (Tr_com, -m_tr * g * N.y),
    (T_l_com, -m_t * g * N.y),
    (L_l_com, -m_l * g * N.y),
    (F_l_com, -m_f * g * N.y),
    (T_r_com, -m_t * g * N.y),
    (L_r_com, -m_l * g * N.y),
    (F_r_com, -m_f * g * N.y),
    # Lumbar torque (Torso relative to Pelvis)
    (Tr, tau_lum * Pel.z), (Pel, -tau_lum * Pel.z),
    # Left leg torques (relative to Pelvis)
    (T_l, tau_lh * Pel.z), (Pel, -tau_lh * Pel.z),
    (L_l, tau_lk * T_l.z), (T_l, -tau_lk * T_l.z),
    (F_l, tau_la * L_l.z), (L_l, -tau_la * L_l.z),
    # Right leg torques (relative to Pelvis)
    (T_r, tau_rh * Pel.z), (Pel, -tau_rh * Pel.z),
    (L_r, tau_rk * T_r.z), (T_r, -tau_rk * T_r.z),
    (F_r, tau_ra * L_r.z), (L_r, -tau_ra * L_r.z),
    # Ground contact forces
    (Heel_l, contact_force(Heel_l, N, O)),
    (Toe_l, contact_force(Toe_l, N, O)),
    (Heel_r, contact_force(Heel_r, N, O)),
    (Toe_r, contact_force(Toe_r, N, O))
]

# -----------------------------------------------------------------------------
# 7. Equations of Motion (Kane's Method)
# -----------------------------------------------------------------------------
kane = me.KanesMethod(
    N,
    q_ind=[q_x, q_y, q_p, q_lum, q_lh, q_lk, q_la, q_rh, q_rk, q_ra],
    u_ind=[u_x, u_y, u_p, u_lum, u_lh, u_lk, u_la, u_rh, u_rk, u_ra],
    kd_eqs=kdes
)
fr, frstar = kane.kanes_equations(bodies, forces)

# The Mass Matrix (M) and Forcing Vector (F)
M = kane.mass_matrix_full
F_vec = kane.forcing_full

# -----------------------------------------------------------------------------
# 8. Compile to a Callable Function (lambdify with PyTorch)
# -----------------------------------------------------------------------------
q_vars = [q_x, q_y, q_p, q_lum, q_lh, q_lk, q_la, q_rh, q_rk, q_ra]
u_vars = [u_x, u_y, u_p, u_lum, u_lh, u_lk, u_la, u_rh, u_rk, u_ra]
torque_vars = [tau_lum, tau_lh, tau_lk, tau_la, tau_rh, tau_rk, tau_ra]
parameter_vars = [
    m_p, m_tr, m_t, m_l, m_f, 
    l_p, l_tr, l_t, l_l, l_f, l_heel, l_toe, 
    c_p, c_tr, c_t, c_l, c_f, 
    I_pz, I_trz, I_tz, I_lz, I_fz, 
    g, h_g, 
    kc, cc, mu, vs
]

# Flatten matrices for 130x faster compilation & evaluation (20x20)
M_flat = [M[i, j] for i in range(20) for j in range(20)]
F_flat = [F_vec[i, 0] for i in range(20)]

print("Compiling equations of motion to PyTorch functions... (This may take a moment)")
M_torch_func = sm.lambdify((q_vars, u_vars, torque_vars, parameter_vars), M_flat, modules='torch')
F_torch_func = sm.lambdify((q_vars, u_vars, torque_vars, parameter_vars), F_flat, modules='torch')

# -----------------------------------------------------------------------------
# JIT Compilation via Tracing
# -----------------------------------------------------------------------------
comp_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)

def _raw_eval_M(q_val, u_val, torques_val, params_val):
    batch_size = q_val.shape[0] if q_val.ndim > 1 else 1
    device = q_val.device
    q_seq = torch.unbind(q_val, dim=-1) if q_val.ndim > 1 else list(q_val)
    u_seq = torch.unbind(u_val, dim=-1) if u_val.ndim > 1 else list(u_val)
    torques_seq = torch.unbind(torques_val, dim=-1) if torques_val.ndim > 1 else list(torques_val)
    params_seq = torch.unbind(params_val, dim=-1) if params_val.ndim > 1 else list(params_val)
    
    M_num = M_torch_func(q_seq, u_seq, torques_seq, params_seq)
    zeros = torch.zeros(batch_size, dtype=torch.float32, device=device)
    M_tensors = [val if isinstance(val, torch.Tensor) else zeros + float(val) for val in M_num]
    return torch.stack(M_tensors, dim=-1).reshape(batch_size, 20, 20)

def _raw_eval_F(q_val, u_val, torques_val, params_val):
    batch_size = q_val.shape[0] if q_val.ndim > 1 else 1
    device = q_val.device
    q_seq = torch.unbind(q_val, dim=-1) if q_val.ndim > 1 else list(q_val)
    u_seq = torch.unbind(u_val, dim=-1) if u_val.ndim > 1 else list(u_val)
    torques_seq = torch.unbind(torques_val, dim=-1) if torques_val.ndim > 1 else list(torques_val)
    params_seq = torch.unbind(params_val, dim=-1) if params_val.ndim > 1 else list(params_val)
    
    F_num = F_torch_func(q_seq, u_seq, torques_seq, params_seq)
    zeros = torch.zeros(batch_size, dtype=torch.float32, device=device)
    F_tensors = [val if isinstance(val, torch.Tensor) else zeros + float(val) for val in F_num]
    return torch.stack(F_tensors, dim=-1).reshape(batch_size, 20, 1)

_jit_cache = {}

def get_jit_evaluators(device):
    dev_str = str(device)
    if dev_str not in _jit_cache:
        dummy_batch = 128
        dummy_q = torch.randn(dummy_batch, 10, device=device)
        dummy_u = torch.randn(dummy_batch, 10, device=device)
        dummy_t = torch.randn(dummy_batch, 7, device=device)
        dummy_p = torch.ones(dummy_batch, 28, device=device)
        try:
            m_jit = torch.jit.trace(_raw_eval_M, (dummy_q, dummy_u, dummy_t, dummy_p))
            f_jit = torch.jit.trace(_raw_eval_F, (dummy_q, dummy_u, dummy_t, dummy_p))
            _jit_cache[dev_str] = (m_jit, f_jit)
        except Exception:
            _jit_cache[dev_str] = (_raw_eval_M, _raw_eval_F)
    return _jit_cache[dev_str]

def skeleton_dynamics(q_val, u_val, torques_val, params_val):
    """Computes the state derivative dx/dt = [dq/dt, du/dt] for the 10-DoF biped skeleton.
    Evaluates JIT compiled equations and solves the linear system entirely on the active device (GPU/CPU).
    """
    device = q_val.device
    
    is_single = q_val.ndim == 1
    if is_single:
        q_val = q_val.unsqueeze(0)
        u_val = u_val.unsqueeze(0)
        torques_val = torques_val.unsqueeze(0)
        params_val = params_val.unsqueeze(0)
        
    q_clean = torch.nan_to_num(q_val, nan=0.0)
    u_clean = torch.nan_to_num(u_val, nan=0.0)
    torques_clean = torch.nan_to_num(torques_val, nan=0.0)
    params_clean = torch.nan_to_num(params_val, nan=0.0)
    
    m_eval, f_eval = get_jit_evaluators(device)
    M_mat = m_eval(q_clean, u_clean, torques_clean, params_clean)
    F_mat = f_eval(q_clean, u_clean, torques_clean, params_clean)
    
    M_mat = torch.nan_to_num(M_mat, nan=0.0)
    F_mat = torch.nan_to_num(F_mat, nan=0.0)
    
    # Solve in double precision to prevent float32 ill-conditioning
    M_mat_double = M_mat.double()
    F_mat_double = F_mat.double()
    eye = torch.eye(20, device=device, dtype=torch.float64)
    dx_dt = torch.linalg.solve(M_mat_double + 1e-6 * eye, F_mat_double).float().squeeze(-1)
    
    dq_dt = torch.nan_to_num(dx_dt[..., :10], nan=0.0)
    du_dt = torch.nan_to_num(dx_dt[..., 10:], nan=0.0)
    
    if is_single:
        dq_dt = dq_dt.squeeze(0)
        du_dt = du_dt.squeeze(0)
        
    return dq_dt, du_dt

print("Equations of motion successfully compiled!")
print("skeleton_dynamics is ready to use with PyTorch JIT.")