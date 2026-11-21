"""
train_multi_walk_gpu.py
GPU-Accelerated 30-Robot Parallel Parameter Exploration Arena.
Hardware: NVIDIA GeForce RTX 2060 (6 GB VRAM)
Features:
  - 100% Original Robot CAD/URDF default colors preserved!
  - 30 Robots simultaneously testing 30 DIFFERENT parameter sets in a single iteration.
  - Batched GPU tensor policy evaluations for high-throughput training.
  - Live MuJoCo viewer displaying all 30 robots competing in real-time.

Usage:
  python train_multi_walk_gpu.py                         # Live GPU training in viewer
  python train_multi_walk_gpu.py --headless              # Fast GPU training without GUI
  python train_multi_walk_gpu.py --pretrained policy_stabilize.npz
"""

import os
import sys
import time
import math
import argparse
import numpy as np
import mujoco
import mujoco.viewer

# Check GPU availability
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

ARENA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "urdf", "bipedal_arena_30.xml")

class ParameterTrial:
    """Dynamic biomechanical parameter configuration for an individual robot."""
    def __init__(self, robot_id=0, num_robots=30):
        self.id = robot_id
        # Parameter 1: Stepping frequency / cadence (1.1 Hz to 3.0 Hz)
        self.freq = 1.1 + 1.9 * (robot_id / max(1, num_robots - 1))
        # Parameter 2: Target forward speed (0.15 m/s to 0.55 m/s)
        self.target_vel = 0.15 + 0.40 * (robot_id / max(1, num_robots - 1))
        # Parameter 3: Tail counter-balance pitch bias (-0.25 to +0.25 rad)
        self.tail_bias = -0.25 + 0.50 * ((robot_id % 6) / 5.0)
        # Parameter 4: Neck posture tilt bias (-0.15 to +0.20 rad)
        self.neck_bias = -0.15 + 0.35 * ((robot_id % 5) / 4.0)
        # Parameter 5: Knee stride clearance amplitude (shallow vs high step)
        self.knee_scale = 0.25 + 0.20 * ((robot_id % 6) / 5.0)
        # Parameter 6: Hip forward swing amplitude
        self.hip_scale = 0.35 + 0.20 * (((robot_id + 2) % 6) / 5.0)
        
        self.update_action_scale()

    def update_action_scale(self):
        self.action_scale = np.array([
            0.30, self.hip_scale, self.knee_scale,
            0.30, self.hip_scale, self.knee_scale,
            0.70, 0.50
        ], dtype=np.float32)

    def to_vector(self):
        return np.array([
            self.freq, self.target_vel, self.tail_bias,
            self.neck_bias, self.knee_scale, self.hip_scale
        ], dtype=np.float32)

    def from_vector(self, vec):
        """Applies new parameters with biomechanical boundary clamping."""
        self.freq = float(np.clip(vec[0], 1.0, 3.2))
        self.target_vel = float(np.clip(vec[1], 0.12, 0.60))
        self.tail_bias = float(np.clip(vec[2], -0.35, 0.35))
        self.neck_bias = float(np.clip(vec[3], -0.25, 0.25))
        self.knee_scale = float(np.clip(vec[4], 0.18, 0.52))
        self.hip_scale = float(np.clip(vec[5], 0.25, 0.55))
        self.update_action_scale()

    @property
    def regime(self):
        if self.freq < 1.4:
            return "Low-Stance Glide"
        elif abs(self.tail_bias) > 0.15:
            return "Tail Pitch Balance"
        elif self.knee_scale > 0.35:
            return "High-Clearance Step"
        elif self.target_vel > 0.40:
            return "Rapid Cadence Trot"
        else:
            return "Athletic Forward Lean"


class DiverseGpuWalkArena:
    """30-Robot Parallel Arena with Parameter Variation and Default Colors."""
    def __init__(self, model_path=ARENA_PATH, num_robots=30):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.num_robots = num_robots
        
        self.act_dim = 8
        self.obs_dim = 31
        
        self.trials = [ParameterTrial(i, num_robots) for i in range(num_robots)]
        self.robots = []
        
        rows = 5
        cols = 6
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
            
            rf_id = self.model.body(f"{prefix}_body1_20").id
            lf_id = self.model.body(f"{prefix}_body1_36").id
            
            self.robots.append({
                "id": i,
                "trial": self.trials[i],
                "start_x": start_x,
                "start_y": start_y,
                "qpos_adr": qpos_adr,
                "qvel_adr": qvel_adr,
                "act_start": act_start,
                "rf_id": rf_id,
                "lf_id": lf_id,
                "alive": True,
                "reward": 0.0,
                "dist": 0.0,
                "time_sec": 0.0
            })
            
        self.step_count = 0
        self.max_steps = 400

    def reset(self, apply_noise=True):
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        
        for r in self.robots:
            r["alive"] = True
            r["reward"] = 0.0
            r["dist"] = 0.0
            r["time_sec"] = 0.0
            adr = r["qpos_adr"]
            trial = r["trial"]
            
            # Base position (x, y, z)
            self.data.qpos[adr + 0] = r["start_x"]
            self.data.qpos[adr + 1] = r["start_y"]
            self.data.qpos[adr + 2] = 0.215 + (np.random.uniform(-0.005, 0.005) if apply_noise else 0.0)
            self.data.qpos[adr + 3 : adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
            
            # Initial joint angles using each robot's parameter trial poise
            q_init = np.zeros(self.act_dim)
            q_init[6] = trial.tail_bias
            q_init[7] = trial.neck_bias
            noise = np.random.uniform(-0.02, 0.02, self.act_dim) if apply_noise else 0.0
            self.data.qpos[adr + 7 : adr + 7 + self.act_dim] = q_init + noise
            
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def get_all_obs(self):
        """Extracts observations for all 30 robots into a batch matrix [30, 31]."""
        obs_batch = np.zeros((self.num_robots, self.obs_dim), dtype=np.float32)
        
        for i, r in enumerate(self.robots):
            q_adr = r["qpos_adr"]
            v_adr = r["qvel_adr"]
            trial = r["trial"]
            
            base_z = self.data.qpos[q_adr + 2]
            w, x, y, z = self.data.qpos[q_adr + 3 : q_adr + 7]
            
            # Projected gravity
            grav_x = 2 * (x * z - w * y)
            grav_y = 2 * (y * z + w * x)
            grav_z = w * w - x * x - y * y + z * z
            
            # CPG phase clock
            phi = 2.0 * math.pi * trial.freq * r["time_sec"]
            
            obs_batch[i, 0] = base_z
            obs_batch[i, 1:4] = [grav_x, grav_y, grav_z]
            obs_batch[i, 4:7] = self.data.qvel[v_adr + 0 : v_adr + 3]
            obs_batch[i, 7:10] = self.data.qvel[v_adr + 3 : v_adr + 6]
            obs_batch[i, 10:18] = self.data.qpos[q_adr + 7 : q_adr + 7 + self.act_dim]
            obs_batch[i, 18:26] = self.data.qvel[v_adr + 6 : v_adr + 6 + self.act_dim]
            obs_batch[i, 26] = trial.target_vel
            obs_batch[i, 27] = math.sin(phi)
            obs_batch[i, 28] = math.cos(phi)
            
        return obs_batch

    def step(self, actions_batch):
        """Executes simulation step across all 30 robots simultaneously."""
        self.step_count += 1
        dt = 0.01  # 5 substeps * 0.002s
        
        for i, r in enumerate(self.robots):
            act_start = r["act_start"]
            trial = r["trial"]
            r["time_sec"] += dt
            
            if r["alive"]:
                act = np.clip(actions_batch[i], -1.0, 1.0)
                q_nom = np.zeros(self.act_dim)
                q_nom[6] = trial.tail_bias
                q_nom[7] = trial.neck_bias
                ctrl = q_nom + act * trial.action_scale
                self.data.ctrl[act_start : act_start + self.act_dim] = ctrl
            else:
                self.data.ctrl[act_start : act_start + self.act_dim] = np.zeros(self.act_dim)
                
        # 5 substeps
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
            
            x_curr = self.data.qpos[q_adr + 0]
            r["dist"] = x_curr - r["start_x"]
            
            vx = self.data.qvel[v_adr + 0]
            vy = self.data.qvel[v_adr + 1]
            vz = self.data.qvel[v_adr + 2]
            z = self.data.qpos[q_adr + 2]
            
            w, x, y, qw_z = self.data.qpos[q_adr + 3 : q_adr + 7]
            sin_pitch = 2.0 * (w * y - x * qw_z)
            pitch = math.asin(max(-1.0, min(1.0, sin_pitch)))
            roll = math.atan2(2.0 * (w * x + y * qw_z), 1.0 - 2.0 * (x * x + y * y))
            
            tail_angle = self.data.qpos[q_adr + 7 + 6]
            neck_angle = self.data.qpos[q_adr + 7 + 7]
            
            rf_z = self.data.xpos[r["rf_id"], 2]
            lf_z = self.data.xpos[r["lf_id"], 2]
            min_foot_z = min(rf_z, lf_z)
            
            # Forward speed tracking
            r_forward = 5.0 * math.exp(-8.0 * (vx - trial.target_vel) ** 2)
            # Anti-jumping penalties
            r_jump = -4.0 * abs(vz)
            r_bounce = -15.0 * max(0.0, z - 0.245) ** 2
            r_flight = -4.0 if min_foot_z > 0.035 else 0.0
            # Pitch and roll stability
            r_pitch = -7.0 * (pitch ** 2)
            r_roll = -4.0 * (roll ** 2)
            # Tail and head dynamic pitch counter-balance
            r_tail = 2.5 * math.exp(-12.0 * (pitch - 0.25 * (tail_angle - trial.tail_bias)) ** 2)
            r_neck = 2.0 * math.exp(-15.0 * (pitch - 0.30 * (neck_angle - trial.neck_bias)) ** 2)
            # Alternating stride foot clearance
            r_stride = 1.2 * min(0.05, abs(rf_z - lf_z))
            
            r_height = 2.0 * math.exp(-35.0 * (z - 0.215) ** 2)
            r_lat = -2.5 * abs(vy)
            r_alive = 3.0
            
            step_reward = (r_forward + r_jump + r_bounce + r_flight + 
                           r_pitch + r_roll + r_tail + r_neck + 
                           r_stride + r_height + r_lat + r_alive)
            r["reward"] += step_reward
            alive_count += 1
            total_speed += vx
            total_pitch += math.degrees(pitch)
            
            if z < 0.13 or z > 0.295 or abs(pitch) > math.radians(35.0) or abs(roll) > math.radians(35.0):
                r["alive"] = False
                r["reward"] -= 15.0
                
        done = (self.step_count >= self.max_steps) or (alive_count == 0)
        avg_speed = total_speed / max(1, alive_count)
        avg_pitch = total_pitch / max(1, alive_count)
        
        return done, alive_count, avg_speed, avg_pitch


class BatchedGpuPolicy:
    """High-Throughput Batched Policy: Evaluates and evolves all 30 neural policies & parameters in parallel."""
    def __init__(self, num_robots=30, obs_dim=31, act_dim=8, hidden_dim=64):
        self.num_robots = num_robots
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        
        # Neural weights: [30, obs_dim, hidden_dim] and [30, hidden_dim, act_dim]
        self.w1 = np.random.randn(num_robots, obs_dim, hidden_dim).astype(np.float32) * 0.15
        self.b1 = np.zeros((num_robots, hidden_dim), dtype=np.float32)
        self.w2 = np.random.randn(num_robots, hidden_dim, act_dim).astype(np.float32) * 0.15
        self.b2 = np.zeros((num_robots, act_dim), dtype=np.float32)
        
        # Biomechanical Gait Parameters [30, 6]:
        # [freq, target_vel, tail_bias, neck_bias, knee_scale, hip_scale]
        self.phys_params = np.zeros((num_robots, 6), dtype=np.float32)
        for i in range(num_robots):
            self.phys_params[i] = ParameterTrial(i, num_robots).to_vector()

    def forward_batch(self, obs_batch):
        """Processes [30, obs_dim] into [30, act_dim] in parallel on GPU."""
        if USE_CUDA:
            with torch.no_grad():
                x = torch.from_numpy(obs_batch).to(torch_device)
                w1_t = torch.from_numpy(self.w1).to(torch_device)
                b1_t = torch.from_numpy(self.b1).to(torch_device)
                w2_t = torch.from_numpy(self.w2).to(torch_device)
                b2_t = torch.from_numpy(self.b2).to(torch_device)
                
                # Batched matrix multiplication: x is [30, 1, obs], w1 is [30, obs, hidden]
                h = torch.tanh(torch.bmm(x.unsqueeze(1), w1_t).squeeze(1) + b1_t)
                out = torch.tanh(torch.bmm(h.unsqueeze(1), w2_t).squeeze(1) + b2_t)
                return out.cpu().numpy()
        else:
            # Multi-threaded BLAS vectorized batch forward pass
            h = np.tanh(np.einsum('bi,bij->bj', obs_batch, self.w1) + self.b1)
            out = np.tanh(np.einsum('bj,bjk->bk', h, self.w2) + self.b2)
            return out

    def mutate_population(self, elite_idx, arena, is_epoch_success=True):
        """Evolves BOTH neural network weights AND physical biomechanical parameters."""
        top_w1 = self.w1[elite_idx[:5]].copy()
        top_b1 = self.b1[elite_idx[:5]].copy()
        top_w2 = self.w2[elite_idx[:5]].copy()
        top_b2 = self.b2[elite_idx[:5]].copy()
        top_params = self.phys_params[elite_idx[:5]].copy()
        
        for i in range(self.num_robots):
            if i in elite_idx[:2] and is_epoch_success:
                # Top 2 champions: Keep proven winning parameters and fine-tune slightly
                self.w1[i] += np.random.randn(*self.w1[i].shape).astype(np.float32) * 0.015
                self.w2[i] += np.random.randn(*self.w2[i].shape).astype(np.float32) * 0.015
                self.phys_params[i] += np.random.randn(6).astype(np.float32) * np.array([0.02, 0.01, 0.01, 0.01, 0.01, 0.01], dtype=np.float32)
            elif i in elite_idx[:5] and is_epoch_success:
                # Elites: Keep winning configuration with slight mutation
                self.w1[i] += np.random.randn(*self.w1[i].shape).astype(np.float32) * 0.03
                self.w2[i] += np.random.randn(*self.w2[i].shape).astype(np.float32) * 0.03
                self.phys_params[i] += np.random.randn(6).astype(np.float32) * np.array([0.04, 0.02, 0.02, 0.02, 0.02, 0.02], dtype=np.float32)
            elif i < 25:
                # 20 Robots: INHERIT WINNING PHYSICAL PARAMETERS + EXPLORATION NOISE!
                p_idx = np.random.randint(len(top_params))
                rate = 0.08 + 0.10 * (i / self.num_robots)
                self.w1[i] = top_w1[p_idx] + np.random.randn(*self.w1[i].shape).astype(np.float32) * rate
                self.b1[i] = top_b1[p_idx] + np.random.randn(*self.b1[i].shape).astype(np.float32) * rate
                self.w2[i] = top_w2[p_idx] + np.random.randn(*self.w2[i].shape).astype(np.float32) * rate
                self.b2[i] = top_b2[p_idx] + np.random.randn(*self.b2[i].shape).astype(np.float32) * rate
                
                # Cross-breed the champion's stepping frequency, speed, and tail pitch!
                p_noise = np.random.randn(6).astype(np.float32) * np.array([0.08, 0.03, 0.03, 0.02, 0.03, 0.03], dtype=np.float32)
                self.phys_params[i] = top_params[p_idx] + p_noise
            else:
                # Exploratory Scouts (Robots 25-29): Test wider varied cadences to prevent local minima
                self.phys_params[i] = ParameterTrial(i, self.num_robots).to_vector()
                self.w1[i] = top_w1[0] + np.random.randn(*self.w1[i].shape).astype(np.float32) * 0.25
                self.b1[i] = top_b1[0] + np.random.randn(*self.b1[i].shape).astype(np.float32) * 0.25
                self.w2[i] = top_w2[0] + np.random.randn(*self.w2[i].shape).astype(np.float32) * 0.25
                self.b2[i] = top_b2[0] + np.random.randn(*self.b2[i].shape).astype(np.float32) * 0.25
            
            # Synchronize the newly evolved physical parameters to the robot in MuJoCo
            arena.robots[i]["trial"].from_vector(self.phys_params[i])

    def save_best(self, robot_idx, filepath, trial):
        """Saves neural network weights AND champion physical gait parameters."""
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
                 hip_scale=float(trial.hip_scale))

    def load_pretrained(self, filepath, arena=None):
        data = np.load(filepath)
        w1_saved = data['w1']
        min_r = min(w1_saved.shape[0], self.obs_dim)
        min_c = min(w1_saved.shape[1], self.hidden_dim)
        
        has_params = 'freq' in data
        if has_params:
            champion_phys = np.array([
                float(data['freq']),
                float(data['target_vel']),
                float(data['tail_bias']),
                float(data['neck_bias']),
                float(data['knee_scale']),
                float(data['hip_scale'])
            ], dtype=np.float32)
            print(f"[OK] Pretrained parameters found: f={champion_phys[0]:.2f}Hz, v={champion_phys[1]:.2f}m/s, tail={champion_phys[2]:+.2f}rad")
        
        for i in range(self.num_robots):
            self.w1[i, :min_r, :min_c] = w1_saved[:min_r, :min_c]
            self.b1[i, :min_c] = data['b1'][:min_c]
            self.w2[i, :min_c, :] = data['w2'][:min_c, :]
            self.b2[i] = data['b2']
            if has_params:
                # Seed population centered around pretrained champion parameters
                noise = np.random.randn(6).astype(np.float32) * np.array([0.05, 0.02, 0.02, 0.02, 0.02, 0.02], dtype=np.float32) if i > 0 else 0.0
                self.phys_params[i] = champion_phys + noise
                if arena is not None:
                    arena.robots[i]["trial"].from_vector(self.phys_params[i])


def run_gpu_training(num_generations=1000, show_viewer=True, pretrained=None, save_path="policy_walk.npz"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(save_path):
        save_path = os.path.join(script_dir, save_path)
    if pretrained and not os.path.isabs(pretrained):
        pretrained = os.path.join(script_dir, pretrained)

    arena = DiverseGpuWalkArena(num_robots=30)
    batch_policy = BatchedGpuPolicy(num_robots=30, obs_dim=arena.obs_dim, act_dim=arena.act_dim)
    
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
    
    print("=" * 78)
    print(" 30-ROBOT PARALLEL GPU ARENA: DYNAMIC ADAPTIVE PARAMETER SELECTION")
    print(f" Accelerating 30 robots with evolving cadences, strides, and tail counter-weights")
    print(f" Robot Models: 100% Default Authentic CAD Colors Preserved")
    print("=" * 78)
    
    viewer = None
    if show_viewer:
        print("[INFO] Launching MuJoCo Viewer: 30 robots testing different parameters in real-time...")
        viewer = mujoco.viewer.launch_passive(arena.model, arena.data)
        
    best_idx = 0
    max_dist = 0.0
    
    try:
        for gen in range(1, num_generations + 1):
            arena.reset(apply_noise=True)
            done = False
            last_alive = 30
            last_speed = 0.0
            last_pitch = 0.0
            
            # Display current Epoch/Iteration banner directly in the MuJoCo Viewer
            if viewer and viewer.is_running():
                try:
                    sim = viewer._get_sim()
                    if sim is not None:
                        sim.load_message(f"EPOCH: {gen:04d} / {num_generations:04d} [{status_tag}] | Alive: {last_alive}/30 | Best Dist: {max_dist:.2f}m")
                except Exception:
                    pass
            
            # SIMULATE ALL 30 ROBOTS RUNNING DIFFERENT PARAMETERS SIMULTANEOUSLY
            while not done:
                # 1. Batch observation extraction for all 30 robots
                obs_batch = arena.get_all_obs()
                
                # 2. Parallel batched policy inference on GPU
                actions_batch = batch_policy.forward_batch(obs_batch)
                
                # 3. Simulate all 30 robots together
                done, last_alive, last_speed, last_pitch = arena.step(actions_batch)
                
                if viewer and viewer.is_running():
                    # Place a glowing 3D beacon sphere above the current leader robot in the arena
                    if viewer.user_scn is not None and len(arena.robots) > 0:
                        try:
                            q_adr = arena.robots[best_idx]["qpos_adr"]
                            leader_pos = arena.data.qpos[q_adr : q_adr + 3].copy()
                            leader_pos[2] += 0.35  # Hover 35cm above the robot's head
                            mujoco.mjv_initGeom(
                                viewer.user_scn.geoms[0],
                                mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([0.06, 0.06, 0.06]),
                                leader_pos,
                                np.eye(3).flatten(),
                                np.array([1.0, 0.85, 0.1, 0.9])  # Golden beacon
                            )
                            viewer.user_scn.ngeom = 1
                        except Exception:
                            pass
                    viewer.sync()
                    time.sleep(0.01)
                    
            returns = np.array([r["reward"] for r in arena.robots])
            distances = np.array([r["dist"] for r in arena.robots])
            
            avg_score = np.mean(returns)
            best_score = np.max(returns)
            best_idx = np.argmax(returns)
            best_robot = arena.robots[best_idx]
            best_trial = best_robot["trial"]
            max_dist = distances[best_idx]
            
            # Dynamic Epoch Success / Failure Evaluation
            is_epoch_success = (max_dist >= 0.25) or (last_alive >= 2 and best_score > 40.0)
            
            if is_epoch_success:
                status_tag = "SUCCESS"
                # Snapshot verified good population
                last_good_w1 = batch_policy.w1.copy()
                last_good_b1 = batch_policy.b1.copy()
                last_good_w2 = batch_policy.w2.copy()
                last_good_b2 = batch_policy.b2.copy()
                last_good_params = batch_policy.phys_params.copy()
                
                if best_score > best_return:
                    best_return = best_score
                    batch_policy.save_best(best_idx, save_path, best_trial)
                    
                sorted_idx = np.argsort(returns)[::-1]
                batch_policy.mutate_population(sorted_idx, arena, is_epoch_success=True)
            else:
                # Catastrophic failure (e.g., all 30 fell down immediately)
                if last_good_w1 is not None:
                    status_tag = "FAIL-ROLLBACK"
                    # Restore from last verified good state
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
            
            print(f"[EPOCH {gen:04d} / {num_generations:04d}] [{status_tag:13s}] | Alive: {last_alive:02d}/30 | Winner: R{best_idx:02d} [{best_trial.regime:20s}] | Dist: {max_dist:5.2f}m | Freq: {best_trial.freq:.2f}Hz | Speed: {best_trial.target_vel:.2f}m/s | Tail: {best_trial.tail_bias:+.2f}rad | Score: {best_score:6.1f}")
            
    except KeyboardInterrupt:
        print("\n[INFO] Training paused by user.")
    finally:
        if viewer:
            viewer.close()
        try:
            best_r = arena.robots[best_idx]
            batch_policy.save_best(best_idx, save_path, best_r["trial"])
        except Exception:
            pass
        print("=" * 78)
        print(f" GPU Training Complete! Champion policy saved to: {save_path}")
        print("=" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="30-Robot Parallel GPU Walking Training")
    parser.add_argument("--headless", action="store_true", help="Fast GPU training without GUI viewer")
    parser.add_argument("--generations", type=int, default=1000, help="Training generations (epochs)")
    parser.add_argument("--pretrained", type=str, default="policy_stabilize.npz", help="Pretrained balance policy")
    parser.add_argument("--save-path", type=str, default="policy_walk.npz", help="Save file")
    args = parser.parse_args()
    
    run_gpu_training(
        num_generations=args.generations,
        show_viewer=not args.headless,
        pretrained=args.pretrained,
        save_path=args.save_path
    )
