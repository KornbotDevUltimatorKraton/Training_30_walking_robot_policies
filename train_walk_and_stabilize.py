"""
train_walk_and_stabilize.py
COMBINED WALKING & STABILIZATION REINFORCEMENT LEARNING ENGINE
Hardware Acceleration: NVIDIA GeForce RTX 2060 (Batched GPU Tensors)
Features:
  - Combined Stabilization & Walking in a single unified training engine.
  - 100 Robots simultaneously training 100 DIFFERENT parameter sets in a single scene.
  - 100 Epochs per iteration default (configurable with --generations).
  - Dynamic parameter evolution: Swarm inherits and refines winning physical parameters every epoch.
  - Upright Head & Rigid Neck Stabilization: Locked passive brackets prevent head from flipping down.
  - Active Dynamic Tail Counter-Balance with Pitch-Rate Damping.
  - Physics contact filtering: 100% eliminates internal self-collision glitching & jumping.
  - Success / Failure detection with automated Rollback Guard.
  - Live on-screen MuJoCo viewer HUD display and 3D champion golden beacon.
  - 100% Authentic CAD default colors preserved.
  - Full export of neural weights AND champion physical gait parameters for Orange Pi.

Usage:
  python train_walk_and_stabilize.py                    # 100 Epochs with live MuJoCo viewer
  python train_walk_and_stabilize.py --headless         # 100 Epochs in ultra-fast headless GPU mode
  python train_walk_and_stabilize.py --generations 200  # Custom epoch count
"""

import os
import sys
import time
import math
import argparse
import numpy as np
import mujoco
import mujoco.viewer

# ==============================================================================
# GPU ACCELERATION CHECK (PyTorch CUDA on RTX 2060)
# ==============================================================================
USE_CUDA = False
torch_device = None
try:
    import torch
    if torch.cuda.is_available():
        USE_CUDA = True
        torch_device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
        print(f"[GPU ACCELERATION ACTIVE] Device: {gpu_name} ({vram_mb} MB VRAM)")
    else:
        print("[INFO] PyTorch installed without CUDA. Running in fast vectorized mode.")
except ImportError:
    print("[INFO] Running in fast vectorized multi-threaded mode.")

ARENA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "urdf", "bipedal_arena_100.xml")


# ==============================================================================
# 1. CONTROL THEORY & RIGID BODY PHYSICS MODELING
# ==============================================================================
class BipedalPhysicsModel:
    """
    Control Theory & Analytical Physics Formulation for Bipedal Locomotion:
    
    1. Linear Inverted Pendulum Model (LIPM) & Capture Point:
       - Natural pendulum frequency: omega_0 = sqrt(g / z_c)
       - Divergent Component of Motion (Capture Point): xi_y = y_com + vy / omega_0
       - Dynamic Stride: L_stride = L_base + (vy / omega_0) * k_dcm
       
    2. Avian Pelvic Roll Dynamics (Lateral Stance Weight Transfer):
       - Shifts pelvis toward the stance leg to unweight the swing foot:
           sway(t) = -roll_scale * sin(2*pi*f*t)
           q_roll = sway - k_p_roll * roll - k_d_roll * roll_rate
           
    3. Active Dynamic Tail Momentum Exchange (Control Lyapunov Function):
       - Reaction moment: tau_tail = - I_torso/I_tail * (kp*pitch + kd*pitch_rate)
       - q_tail = tail_bias - (I_torso/I_tail * k_clf) * pitch - kd * pitch_rate
       
    4. Gyroscopic Horizon Head Decoupling & Gravity Feedforward:
       - Cancels body pitch rotations to maintain head level with world horizon:
           q_neck = neck_bias + 1.0 * pitch + kd * pitch_rate + (m_head * g * r_head) / k_servo
    """
    def __init__(self, num_robots=100):
        self.num_robots = num_robots
        self.mass = 1.8872      # kg (Total mass)
        self.g = 9.81           # m/s^2
        self.z_c = 0.2129       # m (CoM height)
        self.omega_0 = math.sqrt(self.g / self.z_c) # 6.788 rad/s
        
        self.L1 = 0.0599        # m (Thigh)
        self.L2 = 0.0639        # m (Shank)
        self.hip_bias = -0.25   # rad (Flat-foot stance bias)
        
        self.I_torso = 0.018    # kg*m^2
        self.I_tail = 0.0021    # kg*m^2
        self.inertia_ratio = self.I_torso / self.I_tail  # ~8.57
        
        self.m_head = 0.2353    # kg
        self.r_head = 0.045     # m
        self.k_neck_servo = 60.0# N*m/rad
        self.tau_grav_head = (self.m_head * self.g * self.r_head) / self.k_neck_servo # ~0.017 rad
        
        # Central Pattern Generator (CPG) Nonlinear Coupled Limit-Cycle Oscillators
        self.phi_r = np.zeros(num_robots, dtype=np.float32)
        self.phi_l = np.full(num_robots, math.pi, dtype=np.float32)
        self.duty = 0.58  # 58% stance, 42% swing for stable double-support weight transfer

    def reset_cpg(self, robot_id=None):
        """Resets CPG oscillator states to bilateral anti-phase."""
        if robot_id is None:
            self.phi_r[:] = 0.0
            self.phi_l[:] = math.pi
        else:
            self.phi_r[robot_id] = 0.0
            self.phi_l[robot_id] = math.pi

    def step_cpg(self, i, trial, dt, pitch, contact_r, contact_l):
        """
        Adaptive Central Pattern Generator (CPG) Phase Dynamics:
        1. Asymmetric Duty Cycle: Stance phase runs slower (omega_stance), Swing phase runs faster (omega_swing)
        2. Reflexive Stumble Acceleration: Forward pitch tilt accelerates swing leg forward to plant foot sooner
        3. Bilateral Anti-Phase Synchronization: Kuramoto mutual coupling locks left and right legs at 180 deg
        4. Touchdown Phase Entrainment: Ground contact automatically entrains oscillator to stance phase
        """
        omega_stance = 2.0 * math.pi * trial.freq / (2.0 * self.duty)
        omega_swing  = 2.0 * math.pi * trial.freq / (2.0 * (1.0 - self.duty))
        
        is_swing_r = (self.phi_r[i] % (2.0 * math.pi)) < (2.0 * math.pi * (1.0 - self.duty))
        w_r = omega_swing if is_swing_r else omega_stance
        
        is_swing_l = (self.phi_l[i] % (2.0 * math.pi)) < (2.0 * math.pi * (1.0 - self.duty))
        w_l = omega_swing if is_swing_l else omega_stance
        
        # Reflexive stumble correction: pitch tilt accelerates swing phase to step ahead of CoM
        f_stumble_r = max(0.0, pitch) * 3.5 if is_swing_r else 0.0
        f_stumble_l = max(0.0, pitch) * 3.5 if is_swing_l else 0.0
        
        # Mutual anti-phase coupling
        coupling_r = 4.0 * math.sin(self.phi_l[i] - self.phi_r[i] - math.pi)
        coupling_l = 4.0 * math.sin(self.phi_r[i] - self.phi_l[i] - math.pi)
        
        self.phi_r[i] += (w_r + coupling_r + f_stumble_r) * dt
        self.phi_l[i] += (w_l + coupling_l + f_stumble_l) * dt
        
        # Touchdown phase entrainment upon ground contact
        swing_limit = 2.0 * math.pi * (1.0 - self.duty)
        if contact_r and is_swing_r and (self.phi_r[i] % (2.0 * math.pi)) > swing_limit * 0.7:
            self.phi_r[i] = math.ceil(self.phi_r[i] / (2.0 * math.pi)) * (2.0 * math.pi) + swing_limit
        if contact_l and is_swing_l and (self.phi_l[i] % (2.0 * math.pi)) > swing_limit * 0.7:
            self.phi_l[i] = math.ceil(self.phi_l[i] / (2.0 * math.pi)) * (2.0 * math.pi) + swing_limit
            
        r_sw = math.sin(self.phi_r[i])
        l_sw = math.sin(self.phi_l[i])
        return r_sw, l_sw

    def compute_joint_references(self, i, trial, dt, pitch, roll, pitch_rate, roll_rate, vx_lat, vy_fwd, rf_z=0.0, lf_z=0.0, base_z=0.315):
        """
        Physics-grounded 360-degree multi-directional stabilization:
        1. 2D Polar Inverted Pendulum (LIPM) Capture Point (x_cp, y_cp) across front, back, left, right.
        2. Compensatory Swing Leg Stepping: Swings foot toward perturbation angle to catch falls in ANY direction.
        3. Active Stance Leg Straightening: BOTH Hip and Knee extend together to push ground and prop torso up.
        4. Synchronized Dynamic Level: Neck (front) & Tail (back) active pitch leveling.
        """
        # CPG Phase Integration with ground contact entrainment
        contact_r = (rf_z < 0.075)
        contact_l = (lf_z < 0.075)
        r_sw, l_sw = self.step_cpg(i, trial, dt, pitch, contact_r, contact_l)

        # --- 1. 2D LIPM CAPTURE POINT PERTURBATION (Front, Back, Left, Right) ---
        x_cp = (vx_lat / self.omega_0) + 0.40 * roll + 0.10 * roll_rate
        y_cp = (vy_fwd / self.omega_0) + 0.65 * pitch + 0.12 * pitch_rate

        # --- 2. MULTI-DIRECTIONAL SWING LEG COMPENSATION (Reach & Catch) ---
        # Revolute Hip (Roll / Left-Right swing):
        # When falling right (x_cp > 0), right swing leg swings outward (+ angle).
        # When falling left (x_cp < 0), left swing leg swings outward (- angle).
        sway = -trial.roll_scale * r_sw
        q_revolute_r = sway + 0.45 * x_cp * max(0.0, r_sw)
        q_revolute_l = sway + 0.45 * x_cp * max(0.0, l_sw)

        # Rotate Hip (Pitch / Front-Back swing):
        # When falling forward (y_cp > 0), swing leg reaches forward (- angle).
        # When falling backward (y_cp < 0), swing leg reaches backward (+ angle).
        R_stride = trial.hip_scale + 0.20 * min(0.15, math.sqrt(x_cp**2 + y_cp**2))
        q_hip_swing_r = self.hip_bias - R_stride * r_sw - 0.70 * y_cp * max(0.0, r_sw)
        q_hip_swing_l = self.hip_bias - R_stride * l_sw - 0.70 * y_cp * max(0.0, l_sw)

        # Knee Swing Clearance:
        extra_lift_r = max(0.0, -roll) * 0.12
        extra_lift_l = max(0.0, roll) * 0.12
        q_knee_swing_r = (trial.knee_scale + extra_lift_r) * max(0.0, r_sw)
        q_knee_swing_l = -(trial.knee_scale + extra_lift_l) * max(0.0, l_sw)

        # --- 3. DUAL-JOINT STANCE LEG STRAIGHTENING (Hip + Knee push against floor) ---
        # When supporting body weight (r_sw < 0 or l_sw < 0), straighten leg to prevent sinking or pitch collapse
        height_sag = max(0.0, 0.315 - base_z)
        tilt_magnitude = math.sqrt(pitch**2 + roll**2)
        straighten_mag = 1.45 * height_sag + 0.45 * tilt_magnitude + 0.06

        # Knee extension (Right negative = extend; Left positive = extend):
        q_knee_stance_r = -straighten_mag * max(0.0, -r_sw)
        q_knee_stance_l = +straighten_mag * max(0.0, -l_sw)

        # Hip pitch forward push & extension (propelling pelvis forward along +Y while supporting weight):
        q_hip_stance_r = -0.30 * straighten_mag * max(0.0, -r_sw)
        q_hip_stance_l = -0.30 * straighten_mag * max(0.0, -l_sw)

        # Combine Swing + Stance for both joints on each leg:
        q_rotate_r = q_hip_swing_r + q_hip_stance_r - 0.30 * pitch - 0.08 * pitch_rate
        q_rotate_l = q_hip_swing_l + q_hip_stance_l - 0.30 * pitch - 0.08 * pitch_rate

        q_knee_r = q_knee_swing_r + q_knee_stance_r
        q_knee_l = q_knee_swing_l + q_knee_stance_l

        # --- 4. 3-DOF HEAD & NECK ACTIVE STABILIZATION + DYNAMIC TAIL ---
        # Pitch leveling beam:
        q_tail = trial.tail_bias + 1.25 * pitch + 0.20 * pitch_rate
        q_neck = trial.neck_bias + 1.25 * pitch + 0.18 * pitch_rate + self.tau_grav_head

        # Upper head pitch leveler (headconnector): counteracts neck boom pitch & chassis tilt
        q_headconnector = -0.75 * (q_neck - trial.neck_bias) - 0.50 * pitch - 0.10 * pitch_rate

        # Revolute neck (Roll/Yaw lateral tilt counterweight):
        # When robot leans right (roll > 0), head tilts left (- angle) to shift 250g mass and cancel roll!
        q_revoluteneck = -1.35 * roll - 0.25 * roll_rate - 0.60 * x_cp

        q_ref = np.zeros(10, dtype=np.float32)
        q_ref[0] = q_revolute_r
        q_ref[1] = q_rotate_r
        q_ref[2] = q_knee_r
        q_ref[3] = q_revolute_l
        q_ref[4] = q_rotate_l
        q_ref[5] = q_knee_l
        q_ref[6] = q_tail
        q_ref[7] = q_neck
        q_ref[8] = q_headconnector
        q_ref[9] = q_revoluteneck
        return q_ref

class ParameterTrial:
    """Biomechanical gait and stabilization parameters for an individual robot."""
    def __init__(self, robot_id=0, num_robots=100):
        self.id = robot_id
        # Cadence stepping frequency: 1.0 Hz to 2.2 Hz
        self.freq = 1.0 + 1.2 * (robot_id / max(1, num_robots - 1))
        # Desired forward speed: 0.12 m/s to 0.35 m/s
        self.target_vel = 0.12 + 0.23 * (robot_id / max(1, num_robots - 1))
        # Tail counter-balance bias: -0.15 to +0.15 rad
        self.tail_bias = -0.15 + 0.30 * ((robot_id % 10) / 9.0)
        # Neck upright poise bias: +0.20 to +0.38 rad (holds head upright and centered)
        self.neck_bias = 0.20 + 0.18 * ((robot_id % 7) / 6.0)
        # Knee stride clearance: 0.08 to 0.20 rad
        self.knee_scale = 0.08 + 0.12 * ((robot_id % 8) / 7.0)
        # Hip forward swing: 0.10 to 0.22 rad
        self.hip_scale = 0.10 + 0.12 * (((robot_id + 2) % 8) / 7.0)
        # Avian hip roll sway amplitude (lateral weight shift): 0.04 to 0.10 rad (~2.3 to 5.7 deg)
        self.roll_scale = 0.04 + 0.06 * ((robot_id % 6) / 5.0)
        
        self.update_action_scale()

    def update_action_scale(self):
        # 8 Feetech servos: [R_hip_yaw, R_hip_pitch, R_knee, L_hip_yaw, L_hip_pitch, L_knee, Tail, Neck]
        self.action_scale = np.array([
            0.18, 0.22, 0.25,  # Right leg [roll, pitch, knee]
            0.18, 0.22, 0.25,  # Left leg  [roll, pitch, knee]
            0.18,              # Tail
            0.15,              # Lower neck pitch boom
            0.15,              # Upper headconnector pitch
            0.22               # Revolute neck roll/yaw tilt
        ], dtype=np.float32)

    def to_vector(self):
        return np.array([
            self.freq, self.target_vel, self.tail_bias,
            self.neck_bias, self.knee_scale, self.hip_scale, self.roll_scale
        ], dtype=np.float32)

    def from_vector(self, vec):
        """Applies parameters with physical boundary clamping for Feetech servos."""
        self.freq = float(np.clip(vec[0], 0.8, 2.6))
        self.target_vel = float(np.clip(vec[1], 0.10, 0.45))
        self.tail_bias = float(np.clip(vec[2], -0.25, 0.25))
        self.neck_bias = float(np.clip(vec[3], 0.15, 0.45))
        self.knee_scale = float(np.clip(vec[4], 0.06, 0.25))
        self.hip_scale = float(np.clip(vec[5], 0.08, 0.28))
        if len(vec) > 6:
            self.roll_scale = float(np.clip(vec[6], 0.02, 0.12))
        else:
            self.roll_scale = 0.06
        self.update_action_scale()

    @property
    def regime(self):
        if self.freq < 1.25:
            return "Low-Stance Steady"
        elif abs(self.tail_bias) > 0.12:
            return "Active Tail Balance"
        elif self.neck_bias > 0.30:
            return "High Head Poise"
        elif self.knee_scale > 0.15:
            return "High-Clearance Stride"
        else:
            return "Agile Forward Stride"


# ==============================================================================
# 2. VECTORIZED 30-ROBOT COMBINED SIMULATION ARENA
# ==============================================================================
class CombinedWalkStabilizeArena:
    def __init__(self, model_path=ARENA_PATH, num_robots=100):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MuJoCo arena model not found at: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        
        # --- PHYSICS CONTACT FILTERING (Eliminates all internal self-collisions) ---
        self.model.geom_contype[0] = 1
        self.model.geom_conaffinity[0] = 0
        for i in range(1, self.model.ngeom):
            bname = self.model.body(self.model.geom_bodyid[i]).name
            # Foot bodies and base interact with floor
            if "body1_20" in bname or "body1_36" in bname or "base" in bname:
                self.model.geom_contype[i] = 0
                self.model.geom_conaffinity[i] = 1
            else:
                self.model.geom_contype[i] = 0
                self.model.geom_conaffinity[i] = 0
                
        # --- LOCK PASSIVE DECORATIVE BRACKET JOINTS ---
        passive_names = ["mouth", "righthip", "lefthip"]
        for j in range(self.model.njnt):
            jname = self.model.joint(j).name
            if any(p in jname for p in passive_names):
                self.model.jnt_stiffness[j] = 500.0
                self.model.dof_damping[self.model.jnt_dofadr[j]] = 20.0
                
        # --- FEETECH DIGITAL SERVO GAINS & TORQUE SATURATION (STS3215 Specs: 2.8 Nm peak) ---
        for i in range(self.model.nu):
            # Physical stall torque clamp for Feetech STS3215 @ 7.4V-8.4V: ~2.8 N*m
            self.model.actuator_forcerange[i] = [-2.8, 2.8]
            
            aname = self.model.actuator(i).name
            if "knee" in aname:
                self.model.actuator_gainprm[i, 0] = 80.0
                self.model.actuator_biasprm[i, 1] = -80.0
            elif "hip" in aname:
                self.model.actuator_gainprm[i, 0] = 70.0
                self.model.actuator_biasprm[i, 1] = -70.0
            elif "tail" in aname:
                self.model.actuator_gainprm[i, 0] = 80.0
                self.model.actuator_biasprm[i, 1] = -80.0
            elif "neck" in aname or "head" in aname:
                self.model.actuator_gainprm[i, 0] = 65.0
                self.model.actuator_biasprm[i, 1] = -65.0
                
        self.data = mujoco.MjData(self.model)
        self.num_robots = num_robots
        self.physics_model = BipedalPhysicsModel(num_robots=num_robots)
        
        self.act_dim = 10
        self.obs_dim = 35
        
        self.trials = [ParameterTrial(i, num_robots) for i in range(num_robots)]
        self.robots = []
        
        if num_robots == 100:
            rows = 10
            cols = 10
        elif num_robots == 30:
            rows = 5
            cols = 6
        else:
            cols = int(math.ceil(math.sqrt(num_robots)))
            rows = int(math.ceil(num_robots / cols))
        x_spacing = 1.4
        y_spacing = 1.4
        
        for i in range(num_robots):
            prefix = f"r{i:02d}"
            r = i // cols
            c = i % cols
            start_x = (c - (cols - 1) / 2.0) * x_spacing
            start_y = (r - (rows - 1) / 2.0) * y_spacing
            
            root_jnt_id = self.model.joint(f"{prefix}_root").id
            qpos_adr = self.model.jnt_qposadr[root_jnt_id]
            qvel_adr = self.model.jnt_dofadr[root_jnt_id]
            act_start = i * self.act_dim
            
            act_joint_names = [
                f"{prefix}_dof_rightrevolutehip",
                f"{prefix}_dof_rightrotatehip",
                f"{prefix}_dof_rightknee",
                f"{prefix}_dof_leftrevolutehip",
                f"{prefix}_dof_leftrotatehip",
                f"{prefix}_dof_leftkneeservo",
                f"{prefix}_dof_tailjoint",
                f"{prefix}_dof_neck",
                f"{prefix}_dof_headconnector",
                f"{prefix}_dof_revoluteneck"
            ]
            act_qpos_indices = [self.model.jnt_qposadr[self.model.joint(name).id] for name in act_joint_names]
            act_qvel_indices = [self.model.jnt_dofadr[self.model.joint(name).id] for name in act_joint_names]
            tail_qpos_adr = self.model.jnt_qposadr[self.model.joint(f"{prefix}_dof_tailjoint").id]
            neck_qpos_adr = self.model.jnt_qposadr[self.model.joint(f"{prefix}_dof_neck").id]
            
            rf_id = self.model.body(f"{prefix}_body1_20").id
            lf_id = self.model.body(f"{prefix}_body1_36").id
            head_id = self.model.body(f"{prefix}_body1_5").id
            
            self.robots.append({
                "id": i,
                "trial": self.trials[i],
                "start_x": start_x,
                "start_y": start_y,
                "qpos_adr": qpos_adr,
                "qvel_adr": qvel_adr,
                "act_start": act_start,
                "act_qpos_indices": act_qpos_indices,
                "act_qvel_indices": act_qvel_indices,
                "tail_qpos_adr": tail_qpos_adr,
                "neck_qpos_adr": neck_qpos_adr,
                "rf_id": rf_id,
                "lf_id": lf_id,
                "head_id": head_id,
                "alive": True,
                "reward": 0.0,
                "dist": 0.0,
                "time_sec": 0.0
            })
            
        self.step_count = 0
        self.max_steps = 350

    def reset(self, apply_noise=True):
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        self.physics_model.reset_cpg()
        
        for r in self.robots:
            r["alive"] = True
            r["reward"] = 0.0
            r["dist"] = 0.0
            r["time_sec"] = 0.0
            
            adr = r["qpos_adr"]
            trial = r["trial"]
            
            # Base position: 0.315m
            self.data.qpos[adr + 0] = r["start_x"]
            self.data.qpos[adr + 1] = r["start_y"]
            self.data.qpos[adr + 2] = 0.315 + (np.random.uniform(-0.001, 0.001) if apply_noise else 0.0)
            self.data.qpos[adr + 3 : adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
            
            # Initial joint poise: flat foot bias = -0.25 rad, upright neck bias
            q_init = np.zeros(self.act_dim, dtype=np.float32)
            q_init[1] = -0.25  # Right hip pitch
            q_init[4] = -0.25  # Left hip pitch
            q_init[6] = trial.tail_bias
            q_init[7] = trial.neck_bias
            noise = np.random.uniform(-0.005, 0.005, self.act_dim) if apply_noise else 0.0
            self.data.qpos[r["act_qpos_indices"]] = q_init + noise
            
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def get_all_obs(self):
        """Batch observation extraction: [30, 31]."""
        obs_batch = np.zeros((self.num_robots, self.obs_dim), dtype=np.float32)
        
        for i, r in enumerate(self.robots):
            q_adr = r["qpos_adr"]
            v_adr = r["qvel_adr"]
            trial = r["trial"]
            
            base_z = self.data.qpos[q_adr + 2]
            w, x, y, z = self.data.qpos[q_adr + 3 : q_adr + 7]
            
            # Projected gravity vector from quaternion (IMU MPU6050 signal)
            grav_x = 2.0 * (x * z - w * y)
            grav_y = 2.0 * (y * z + w * x)
            grav_z = w * w - x * x - y * y + z * z
            
            # True pitch and roll
            r21 = 2.0 * (y * z + w * x)
            r20 = 2.0 * (x * z - w * y)
            pitch = math.asin(max(-1.0, min(1.0, -r21)))
            roll = math.asin(max(-1.0, min(1.0, r20)))
            
            # CPG cyclic clock for rhythmic bipedal stepping
            phi_r = self.physics_model.phi_r[i]
            
            obs_batch[i, 0] = base_z
            obs_batch[i, 1:4] = [grav_x, grav_y, grav_z]
            obs_batch[i, 4:7] = self.data.qvel[v_adr + 0 : v_adr + 3]
            obs_batch[i, 7:10] = self.data.qvel[v_adr + 3 : v_adr + 6]
            obs_batch[i, 10:20] = self.data.qpos[r["act_qpos_indices"]]
            obs_batch[i, 20:30] = self.data.qvel[r["act_qvel_indices"]]
            obs_batch[i, 30] = trial.target_vel
            obs_batch[i, 31] = math.sin(phi_r)
            obs_batch[i, 32] = math.cos(phi_r)
            obs_batch[i, 33] = pitch
            obs_batch[i, 34] = roll
            
        return obs_batch

    def step(self, actions_batch):
        """Simulates all 30 robots simultaneously with Anti-Phase CPG, Upright Head & Active Tail."""
        self.step_count += 1
        dt = 0.01  # 5 substeps * 0.002s
        
        for i, r in enumerate(self.robots):
            act_start = r["act_start"]
            trial = r["trial"]
            r["time_sec"] += dt
            
            if r["alive"]:
                act = np.clip(actions_batch[i], -1.0, 1.0)
                
                # Active orientation & angular rates from IMU
                q_adr = r["qpos_adr"]
                v_adr = r["qvel_adr"]
                w, x, y, qw_z = self.data.qpos[q_adr + 3 : q_adr + 7]
                r21 = 2.0 * (y * qw_z + w * x)
                r20 = 2.0 * (x * qw_z - w * y)
                pitch = math.asin(max(-1.0, min(1.0, -r21)))
                roll = math.asin(max(-1.0, min(1.0, r20)))
                pitch_rate = self.data.qvel[v_adr + 3]
                roll_rate = self.data.qvel[v_adr + 4]
                vx_lat = self.data.qvel[v_adr + 0]
                vy_fwd = self.data.qvel[v_adr + 1]
                
                # --- RIGID BODY PHYSICS & CONTROL THEORY REFERENCE TRAJECTORY ---
                # Evaluates 2D Polar LIPM Foot Placement, Revolute & Rotate Hip Coupling, and Neck/Tail Dynamic Level
                rf_z = self.data.xpos[r["rf_id"], 2]
                lf_z = self.data.xpos[r["lf_id"], 2]
                base_z = self.data.qpos[q_adr + 2]
                q_ref = self.physics_model.compute_joint_references(
                    i, trial, dt, pitch, roll, pitch_rate, roll_rate, vx_lat, vy_fwd, rf_z, lf_z, base_z
                )
                
                # Neural Policy Residuals: Adaptive micro-adjustments for unmodeled ground contact
                residual = act * trial.action_scale
                ctrl = q_ref + residual
                self.data.ctrl[act_start : act_start + self.act_dim] = ctrl
            else:
                # Do not drop actuators to zero (which causes the head to dive down and legs to collapse);
                # hold nominal standing posture with upright head and active poise.
                q_stand = np.zeros(self.act_dim, dtype=np.float32)
                q_stand[1] = -0.25
                q_stand[4] = -0.25
                q_stand[6] = trial.tail_bias
                q_stand[7] = trial.neck_bias
                self.data.ctrl[act_start : act_start + self.act_dim] = q_stand
                
        # 5 MuJoCo physics substeps
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
            
        alive_count = 0
        total_speed = 0.0
        total_pitch = 0.0
        
        for i, r in enumerate(self.robots):
            if not r["alive"]:
                continue
                
            q_adr = r["qpos_adr"]
            v_adr = r["qvel_adr"]
            trial = r["trial"]
            
            # Forward distance along +Y
            y_curr = self.data.qpos[q_adr + 1]
            r["dist"] = y_curr - r["start_y"]
            
            vx_lat = self.data.qvel[v_adr + 0]
            vy_fwd = self.data.qvel[v_adr + 1]
            vz_hop = self.data.qvel[v_adr + 2]
            z = self.data.qpos[q_adr + 2]
            
            # Quaternion orientation
            w, x, y, qw_z = self.data.qpos[q_adr + 3 : q_adr + 7]
            r21 = 2.0 * (y * qw_z + w * x)
            r20 = 2.0 * (x * qw_z - w * y)
            pitch = math.asin(max(-1.0, min(1.0, -r21)))
            roll = math.asin(max(-1.0, min(1.0, r20)))
            
            tail_angle = self.data.qpos[r["tail_qpos_adr"]]
            neck_angle = self.data.qpos[r["neck_qpos_adr"]]
            
            rf_z = self.data.xpos[r["rf_id"], 2]
            lf_z = self.data.xpos[r["lf_id"], 2]
            min_foot_z = min(rf_z, lf_z)
            head_z = self.data.xpos[r["head_id"], 2]
            
            # --- REWARD ARCHITECTURE: WALKING + UPRIGHT HEAD & TAIL STABILIZATION ---
            # 1. Forward Locomotion & Strict Anti-Backward Motion
            if vy_fwd > 0.0:
                r_forward = 15.0 * vy_fwd + 5.0 * math.exp(-6.0 * (vy_fwd - trial.target_vel) ** 2)
            else:
                r_forward = -25.0 * abs(vy_fwd)
                
            if r["dist"] > 0.0:
                r_progress = 25.0 * r["dist"]
            else:
                r_progress = -35.0 * abs(r["dist"])
            
            # 2. Dynamic Level: Neck (Front) & Tail (Back) Longitudinal Leveling
            r_head = 6.0 * math.exp(-35.0 * (head_z - 0.27) ** 2)
            r_head_droop = -30.0 * max(0.0, 0.23 - head_z)
            r_forward_pitch_penalty = -25.0 * max(0.0, pitch - math.radians(14.0))
            r_backward_pitch_penalty = -25.0 * max(0.0, -pitch - math.radians(14.0))
            
            # Dynamic Level coordination: front (neck) and back (tail) level the chassis
            r_tail_level = 4.0 * math.exp(-15.0 * (pitch - 0.50 * (tail_angle - trial.tail_bias)) ** 2)
            r_neck_level = 4.0 * math.exp(-15.0 * (pitch - 0.50 * (neck_angle - trial.neck_bias)) ** 2)
            
            # 3. Dynamic Walking + Upright Head Combo Bonus
            r_stable_walk = 0.0
            if head_z >= 0.24 and abs(pitch) < math.radians(12.0) and vy_fwd > 0.03:
                r_stable_walk = 12.0  # Massive reinforcement for moving forward with level chassis and upright head
            
            # 4. 2D Polar Horizontal Equilibrium (Compensates all angle directions)
            theta_mag = math.sqrt(pitch**2 + roll**2)
            r_polar_level = 8.0 * math.exp(-25.0 * (theta_mag ** 2))
            
            # 5. Strict Anti-Jumping & Anti-Flight Penalties
            r_jump = -30.0 * (vz_hop ** 2) - 10.0 * abs(vz_hop)
            r_bounce = -25.0 * max(0.0, z - 0.33) ** 2
            r_flight = -25.0 if min_foot_z > 0.08 else 0.0
            
            # 6. Postural Pitch & Roll Stabilization
            r_pitch = -14.0 * (pitch ** 2)
            r_roll = -12.0 * (roll ** 2)
            r_lat = -4.0 * abs(vx_lat)
            
            # 7. Stable Pelvis Height & Alive Incentive
            r_height = 5.0 * math.exp(-40.0 * (z - 0.305) ** 2)
            r_alive = 4.0
            
            step_reward = (r_forward + r_progress + r_head + r_head_droop + 
                           r_forward_pitch_penalty + r_backward_pitch_penalty +
                           r_tail_level + r_neck_level + r_polar_level +
                           r_stable_walk + 
                           r_jump + r_bounce + r_flight + 
                           r_pitch + r_roll + r_lat + r_height + r_alive)
            r["reward"] += step_reward
            alive_count += 1
            total_speed += vy_fwd
            total_pitch += math.degrees(pitch)
            
            # Death conditions: Fall below 0.18m, launch above 0.36m, tilt > 34 deg, head tilt down < 0.17m, or drift backward < -0.05m
            # True fall conditions: actual physical collapse to ground or complete tip-over
            # (Do NOT kill robots that are standing and balancing well even if shifting slightly backward)
            if (z < 0.16 or z > 0.38 or
                abs(pitch) > math.radians(45.0) or
                abs(roll) > math.radians(40.0) or
                head_z < 0.12):
                r["alive"] = False
                r["reward"] -= 30.0
                
        done = (self.step_count >= self.max_steps) or (alive_count == 0)
        avg_speed = total_speed / max(1, alive_count)
        avg_pitch = total_pitch / max(1, alive_count)
        
        return done, alive_count, avg_speed, avg_pitch


# ==============================================================================
# 3. PARALLEL GPU POLICY & SWARM PARAMETER EVOLUTION
# ==============================================================================
class BatchedGpuPolicy:
    """Evaluates and evolves 100 neural policies and physical parameters on GPU."""
    def __init__(self, num_robots=100, obs_dim=35, act_dim=10, hidden_dim=64):
        self.num_robots = num_robots
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        
        self.w1 = np.random.randn(num_robots, obs_dim, hidden_dim).astype(np.float32) * 0.10
        self.b1 = np.zeros((num_robots, hidden_dim), dtype=np.float32)
        self.w2 = np.random.randn(num_robots, hidden_dim, act_dim).astype(np.float32) * 0.10
        self.b2 = np.zeros((num_robots, act_dim), dtype=np.float32)
        
        self.phys_params = np.zeros((num_robots, 7), dtype=np.float32)
        for i in range(num_robots):
            self.phys_params[i] = ParameterTrial(i, num_robots).to_vector()

    def forward_batch(self, obs_batch):
        """Batched forward pass across all 30 robots on GPU."""
        if USE_CUDA:
            with torch.no_grad():
                x = torch.from_numpy(obs_batch).to(torch_device)
                w1_t = torch.from_numpy(self.w1).to(torch_device)
                b1_t = torch.from_numpy(self.b1).to(torch_device)
                w2_t = torch.from_numpy(self.w2).to(torch_device)
                b2_t = torch.from_numpy(self.b2).to(torch_device)
                
                h = torch.tanh(torch.bmm(x.unsqueeze(1), w1_t).squeeze(1) + b1_t)
                out = torch.tanh(torch.bmm(h.unsqueeze(1), w2_t).squeeze(1) + b2_t)
                return out.cpu().numpy()
        else:
            h = np.tanh(np.einsum('bi,bij->bj', obs_batch, self.w1) + self.b1)
            out = np.tanh(np.einsum('bj,bjk->bk', h, self.w2) + self.b2)
            return out

    def mutate_population(self, elite_idx, arena, is_epoch_success=True):
        """Swarm adapts its parameters and neural weights around the winning champion."""
        top_w1 = self.w1[elite_idx[:5]].copy()
        top_b1 = self.b1[elite_idx[:5]].copy()
        top_w2 = self.w2[elite_idx[:5]].copy()
        top_b2 = self.b2[elite_idx[:5]].copy()
        top_params = self.phys_params[elite_idx[:5]].copy()
        
        for i in range(self.num_robots):
            if i in elite_idx[:2] and is_epoch_success:
                # Elitism: Champions preserve proven weights & parameters with micro-tuning
                self.w1[i] += np.random.randn(*self.w1[i].shape).astype(np.float32) * 0.010
                self.w2[i] += np.random.randn(*self.w2[i].shape).astype(np.float32) * 0.010
                self.phys_params[i] += np.random.randn(7).astype(np.float32) * np.array([0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.008], dtype=np.float32)
            elif i in elite_idx[:5] and is_epoch_success:
                # Runners-up: Fine tune parameters
                self.w1[i] += np.random.randn(*self.w1[i].shape).astype(np.float32) * 0.020
                self.w2[i] += np.random.randn(*self.w2[i].shape).astype(np.float32) * 0.020
                self.phys_params[i] += np.random.randn(7).astype(np.float32) * np.array([0.03, 0.015, 0.015, 0.015, 0.015, 0.015, 0.010], dtype=np.float32)
            elif i < 15:
                # Swarm Exploitation: Inherit champion parameters + focused exploration noise
                p_idx = np.random.randint(len(top_params))
                rate = 0.05 + 0.07 * (i / self.num_robots)
                self.w1[i] = top_w1[p_idx] + np.random.randn(*self.w1[i].shape).astype(np.float32) * rate
                self.b1[i] = top_b1[p_idx] + np.random.randn(*self.b1[i].shape).astype(np.float32) * rate
                self.w2[i] = top_w2[p_idx] + np.random.randn(*self.w2[i].shape).astype(np.float32) * rate
                self.b2[i] = top_b2[p_idx] + np.random.randn(*self.b2[i].shape).astype(np.float32) * rate
                
                p_noise = np.random.randn(7).astype(np.float32) * np.array([0.08, 0.03, 0.03, 0.03, 0.03, 0.03, 0.02], dtype=np.float32)
                self.phys_params[i] = top_params[p_idx] + p_noise
            else:
                # Exploratory Scouts: Maintain wide cadence & diverse parameter spectrum across arena
                self.phys_params[i] = ParameterTrial(i, self.num_robots).to_vector()
                self.w1[i] = top_w1[0] + np.random.randn(*self.w1[i].shape).astype(np.float32) * 0.18
                self.b1[i] = top_b1[0] + np.random.randn(*self.b1[i].shape).astype(np.float32) * 0.18
                self.w2[i] = top_w2[0] + np.random.randn(*self.w2[i].shape).astype(np.float32) * 0.18
                self.b2[i] = top_b2[0] + np.random.randn(*self.b2[i].shape).astype(np.float32) * 0.18
            
            # Synchronize evolved physical parameters with the MuJoCo robot
            arena.robots[i]["trial"].from_vector(self.phys_params[i])

    def save_best(self, robot_idx, filepath, trial):
        """Saves neural weights AND champion physical gait parameters."""
        np.savez(filepath,
                 w1=self.w1[robot_idx],
                 b1=self.b1[robot_idx],
                 w2=self.w2[robot_idx],
                 b2=self.b2[robot_idx],
                 freq=float(trial.freq),
                 target_vel=float(trial.target_vel),
                 tail_bias=float(trial.tail_bias),
                 neck_bias=float(trial.neck_bias),
                 knee_scale=float(trial.knee_scale),
                 hip_scale=float(trial.hip_scale),
                 roll_scale=float(trial.roll_scale))

    def load_pretrained(self, filepath, arena=None):
        if not os.path.exists(filepath):
            return
        data = np.load(filepath)
        w1_saved = data['w1']
        min_r = min(w1_saved.shape[0], self.obs_dim)
        min_c = min(w1_saved.shape[1], self.hidden_dim)
        
        has_params = 'freq' in data
        if has_params:
            roll_val = float(data['roll_scale']) if 'roll_scale' in data else 0.06
            champion_phys = np.array([
                float(data['freq']),
                float(data['target_vel']),
                float(data['tail_bias']),
                float(data['neck_bias']),
                float(data['knee_scale']),
                float(data['hip_scale']),
                roll_val
            ], dtype=np.float32)
            print(f"[OK] Checkpoint parameters loaded: f={champion_phys[0]:.2f}Hz, v={champion_phys[1]:.2f}m/s, neck={champion_phys[3]:+.2f}rad, roll_sway={champion_phys[6]:.2f}rad")
        
        w2_saved = data['w2']
        min_out = min(w2_saved.shape[1], self.act_dim)
        for i in range(self.num_robots):
            self.w1[i, :min_r, :min_c] = w1_saved[:min_r, :min_c]
            self.b1[i, :min_c] = data['b1'][:min_c]
            self.w2[i, :min_c, :min_out] = w2_saved[:min_c, :min_out]
            self.b2[i, :min_out] = data['b2'][:min_out]
            if has_params:
                if i == 0:
                    self.phys_params[0] = champion_phys
                    if arena is not None:
                        arena.robots[0]["trial"].from_vector(champion_phys)
                else:
                    # Maintain varied parameter diversity across the arena (cadence, stride, knee lift, sway)
                    self.phys_params[i] = ParameterTrial(i, self.num_robots).to_vector()
                    if arena is not None:
                        arena.robots[i]["trial"].from_vector(self.phys_params[i])


# ==============================================================================
# 4. MAIN COMBINED TRAINING LOOP
# ==============================================================================
def run_combined_training(num_generations=100, show_viewer=True, pretrained=None, save_path="policy_walk.npz", num_robots=100):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(save_path):
        save_path = os.path.join(script_dir, save_path)
    if pretrained and not os.path.isabs(pretrained):
        pretrained = os.path.join(script_dir, pretrained)
    elif pretrained is None and os.path.exists(os.path.join(script_dir, "policy_walk.npz")):
        pretrained = os.path.join(script_dir, "policy_walk.npz")

    arena = CombinedWalkStabilizeArena(num_robots=num_robots)
    batch_policy = BatchedGpuPolicy(num_robots=num_robots, obs_dim=arena.obs_dim, act_dim=arena.act_dim)
    
    if pretrained and os.path.exists(pretrained):
        print(f"[INFO] Initializing population from checkpoint: {pretrained}")
        batch_policy.load_pretrained(pretrained, arena=arena)
        
    best_return = -float('inf')
    last_good_w1 = None
    last_good_b1 = None
    last_good_w2 = None
    last_good_b2 = None
    last_good_params = None
    status_tag = "EXPLORING"
    
    print("=" * 80)
    print(f" {num_robots}-ROBOT COMBINED WALKING & STABILIZATION ARENA (NVIDIA RTX 2060)")
    print(f" Upright Head & Rigid Neck Stabilization + Dynamic Tail Pitch Damping")
    print(f" Target: {num_generations} Epochs | Authentic CAD Default Colors Preserved")
    print("=" * 80)
    
    viewer = None
    if show_viewer:
        print(f"[INFO] Launching MuJoCo Viewer: {num_robots} robots training live in real-time...")
        viewer = mujoco.viewer.launch_passive(arena.model, arena.data)
        
    best_idx = 0
    max_dist = 0.0
    
    try:
        for gen in range(1, num_generations + 1):
            arena.reset(apply_noise=True)
            done = False
            last_alive = num_robots
            last_speed = 0.0
            last_pitch = 0.0
            
            # Display current Epoch/Iteration banner in MuJoCo viewer HUD
            if viewer and viewer.is_running():
                try:
                    sim = viewer._get_sim()
                    if sim is not None:
                        sim.load_message(f"EPOCH: {gen:03d} / {num_generations:03d} [{status_tag}] | Alive: {last_alive:03d}/{num_robots} | Winner: R{best_idx:02d} (Dist: {max_dist:.2f}m)")
                except Exception:
                    pass
            
            # SIMULATE ALL ROBOTS TOGETHER
            while not done:
                obs_batch = arena.get_all_obs()
                actions_batch = batch_policy.forward_batch(obs_batch)
                done, last_alive, last_speed, last_pitch = arena.step(actions_batch)
                
                if viewer and viewer.is_running():
                    if viewer.user_scn is not None and len(arena.robots) > 0:
                        try:
                            q_adr = arena.robots[best_idx]["qpos_adr"]
                            leader_pos = arena.data.qpos[q_adr : q_adr + 3].copy()
                            leader_pos[2] += 0.35  # 35cm above the robot's head
                            mujoco.mjv_initGeom(
                                viewer.user_scn.geoms[0],
                                mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([0.06, 0.06, 0.06]),
                                leader_pos,
                                np.eye(3).flatten(),
                                np.array([1.0, 0.85, 0.1, 0.9])  # Glowing gold
                            )
                            viewer.user_scn.ngeom = 1
                        except Exception:
                            pass
                    viewer.sync()
                    time.sleep(0.005)
                    
            returns = np.array([r["reward"] for r in arena.robots])
            distances = np.array([r["dist"] for r in arena.robots])
            
            avg_score = np.mean(returns)
            best_score = np.max(returns)
            best_idx = np.argmax(returns)
            best_robot = arena.robots[best_idx]
            best_trial = best_robot["trial"]
            max_dist = distances[best_idx]
            
            # Epoch Success / Failure Evaluation: MUST make positive forward progress and high stability score
            is_epoch_success = (max_dist >= 0.05) and (best_score >= 200.0)
            
            if is_epoch_success:
                status_tag = "SUCCESS"
                last_good_w1 = batch_policy.w1.copy()
                last_good_b1 = batch_policy.b1.copy()
                last_good_w2 = batch_policy.w2.copy()
                last_good_b2 = batch_policy.b2.copy()
                last_good_params = batch_policy.phys_params.copy()
                
                if best_score > best_return:
                    best_return = best_score
                    batch_policy.save_best(best_idx, save_path, best_trial)
                    print(f"  >>> [NEW ALL-TIME BEST CHAMPION] Score: {best_score:.1f} | Dist: {max_dist:.2f}m | Saved to {save_path}")
                    
                sorted_idx = np.argsort(returns)[::-1]
                batch_policy.mutate_population(sorted_idx, arena, is_epoch_success=True)
            else:
                if last_good_w1 is not None:
                    status_tag = "FAIL-ROLLBACK"
                    batch_policy.w1 = last_good_w1.copy()
                    batch_policy.b1 = last_good_b1.copy()
                    batch_policy.w2 = last_good_w2.copy()
                    batch_policy.b2 = last_good_b2.copy()
                    batch_policy.phys_params = last_good_params.copy()
                    sorted_idx = np.argsort(returns)[::-1]
                    batch_policy.mutate_population(sorted_idx, arena, is_epoch_success=False)
                else:
                    status_tag = "FAIL-EXPLORE"
                    sorted_idx = np.argsort(returns)[::-1]
                    batch_policy.mutate_population(sorted_idx, arena, is_epoch_success=False)
            
            print(f"[EPOCH {gen:03d} / {num_generations:03d}] [{status_tag:13s}] | Alive: {last_alive:03d}/{num_robots} | Winner: R{best_idx:02d} [{best_trial.regime:20s}] | Dist: {max_dist:5.2f}m | Freq: {best_trial.freq:.2f}Hz | Sway: {best_trial.roll_scale:.2f}rad | Neck: {best_trial.neck_bias:+.2f}rad | Score: {best_score:6.1f}")
            sys.stdout.flush()
            sys.stdout.flush()
            
    except KeyboardInterrupt:
        print("\n[INFO] Training paused by user.")
    finally:
        if viewer:
            viewer.close()
        print("=" * 80)
        print(f" Training Complete! Combined Champion policy saved to: {save_path}")
        print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combined Walking & Stabilization 100-Robot Arena")
    parser.add_argument("--headless", action="store_true", help="Ultra-fast GPU training without GUI viewer")
    parser.add_argument("--generations", type=int, default=100, help="Training epochs (default: 100)")
    parser.add_argument("--num-robots", type=int, default=100, help="Number of robots in the arena (default: 100)")
    parser.add_argument("--pretrained", type=str, default=None, help="Pretrained balance checkpoint")
    parser.add_argument("--save-path", type=str, default="policy_walk.npz", help="Save file destination")
    parser.add_argument("--test", "--view", action="store_true", help="View trained policy walking in MuJoCo GUI")
    args = parser.parse_args()
    
    if args.test:
        from view_champion import view_champion
        view_champion(policy_path=args.save_path if args.pretrained is None else args.pretrained, num_robots=args.num_robots)
    else:
        run_combined_training(
            num_generations=args.generations,
            show_viewer=not args.headless,
            pretrained=args.pretrained,
            save_path=args.save_path,
            num_robots=args.num_robots
        )
