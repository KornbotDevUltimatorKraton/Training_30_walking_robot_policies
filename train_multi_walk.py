"""
train_multi_walk.py
DIVERSE MULTI-ROBOT ARENA: 30 Bipedal Robots training with 30 DIFFERENT walking strategies!
Each robot has its own stepping cadence, gait frequency, tail posture, stride height, and neural weights.

Usage:
  python train_multi_walk.py                         # Watch 30 diverse robots training live in MuJoCo!
  python train_multi_walk.py --headless              # Fast training without GUI
  python train_multi_walk.py --pretrained policy_stabilize.npz  # Warm-start from balance policy
"""

import os
import sys
import time
import math
import argparse
import numpy as np
import mujoco
import mujoco.viewer

ARENA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "urdf", "bipedal_arena_30.xml")

class DiverseWalkArena:
    """Vectorized Environment with 30 Visibly Diverse Walking Strategies."""
    def __init__(self, model_path=ARENA_PATH, num_robots=30):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.num_robots = num_robots
        
        self.act_dim = 8
        self.obs_dim = 31  # 28 state + 1 cmd_speed + 2 CPG clock signals (sin, cos)
        
        self.robots = []
        rows = 5
        cols = 6
        x_spacing = 1.4
        y_spacing = 1.4
        
        # 5 distinct behavioral niches across the 30 robots
        niche_names = [
            "Cautious Low-Stance Walk",
            "Active Tail Counter-Balance",
            "High-Clearance Stride",
            "Forward-Leaning Athletic",
            "Rapid Cadence Dynamic Walk"
        ]
        
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
            
            # --- UNIQUE PERSONALITY / DIVERSE PARAMETERS FOR THIS ROBOT ---
            niche_idx = i // 6
            niche = niche_names[niche_idx]
            
            # 1. Distinct stepping frequency (1.1 Hz slow to 3.0 Hz rapid)
            freq = 1.1 + 1.9 * (i / (num_robots - 1))
            
            # 2. Distinct target speed (0.15 m/s to 0.55 m/s)
            target_vel = 0.15 + 0.40 * (i / (num_robots - 1))
            
            # 3. Distinct tail posture bias and knee stride scale
            tail_bias = -0.25 + 0.50 * ((i % 6) / 5.0)  # Tail angle bias from -0.25 to +0.25 rad
            neck_bias = -0.15 + 0.30 * ((i % 5) / 4.0)  # Neck posture bias
            knee_scale = 0.25 + 0.20 * ((i % 6) / 5.0)  # Stride height (shallow vs high step)
            hip_scale = 0.35 + 0.20 * (((i + 2) % 6) / 5.0)
            
            # Action scale tailored to this robot's individual style
            # [r_hip_roll, r_hip_pitch, r_knee, l_hip_roll, l_hip_pitch, l_knee, tail, neck]
            action_scale = np.array([0.30, hip_scale, knee_scale, 0.30, hip_scale, knee_scale, 0.70, 0.50])
            
            self.robots.append({
                "id": i,
                "prefix": prefix,
                "niche": niche,
                "start_x": start_x,
                "start_y": start_y,
                "qpos_adr": qpos_adr,
                "qvel_adr": qvel_adr,
                "act_start": act_start,
                "rf_id": rf_id,
                "lf_id": lf_id,
                "freq": freq,
                "target_vel": target_vel,
                "tail_bias": tail_bias,
                "neck_bias": neck_bias,
                "action_scale": action_scale,
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
            
            # Base position (x, y, z)
            self.data.qpos[adr + 0] = r["start_x"]
            self.data.qpos[adr + 1] = r["start_y"]
            self.data.qpos[adr + 2] = 0.28 + (np.random.uniform(-0.01, 0.01) if apply_noise else 0.0)
            self.data.qpos[adr + 3 : adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
            
            # Initial joint posture with each robot's individual tail and neck poise
            q_init = np.zeros(self.act_dim)
            q_init[6] = r["tail_bias"]
            q_init[7] = r["neck_bias"]
            noise = np.random.uniform(-0.02, 0.02, self.act_dim) if apply_noise else 0.0
            self.data.qpos[adr + 7 : adr + 7 + self.act_dim] = q_init + noise
            
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def get_robot_obs(self, robot_idx):
        r = self.robots[robot_idx]
        q_adr = r["qpos_adr"]
        v_adr = r["qvel_adr"]
        
        base_z = self.data.qpos[q_adr + 2 : q_adr + 3]
        w, x, y, z = self.data.qpos[q_adr + 3 : q_adr + 7]
        
        grav_x = 2 * (x * z - w * y)
        grav_y = 2 * (y * z + w * x)
        grav_z = w * w - x * x - y * y + z * z
        proj_gravity = np.array([grav_x, grav_y, grav_z])
        
        linvel = self.data.qvel[v_adr + 0 : v_adr + 3]
        angvel = self.data.qvel[v_adr + 3 : v_adr + 6]
        
        joint_pos = self.data.qpos[q_adr + 7 : q_adr + 7 + self.act_dim]
        joint_vel = self.data.qvel[v_adr + 6 : v_adr + 6 + self.act_dim]
        
        # Central Pattern Generator (CPG) internal clock for distinct stepping cadence
        phi = 2.0 * math.pi * r["freq"] * r["time_sec"]
        cpg_sin = np.array([math.sin(phi)])
        cpg_cos = np.array([math.cos(phi)])
        cmd_vel = np.array([r["target_vel"]])
        
        obs = np.concatenate([
            base_z,
            proj_gravity,
            linvel,
            angvel,
            joint_pos,
            joint_vel,
            cmd_vel,
            cpg_sin,
            cpg_cos
        ]).astype(np.float32)
        if len(obs) < self.obs_dim:
            obs = np.pad(obs, (0, self.obs_dim - len(obs)))
        return obs

    def step(self, actions):
        """Simulate all 30 diverse robots simultaneously."""
        self.step_count += 1
        dt = 0.01  # 5 substeps * 0.002s
        
        for i, r in enumerate(self.robots):
            act_start = r["act_start"]
            r["time_sec"] += dt
            if r["alive"]:
                act = np.clip(actions[i], -1.0, 1.0)
                # Nominal baseline posture with unique tail/neck bias
                q_nom = np.zeros(self.act_dim)
                q_nom[6] = r["tail_bias"]
                q_nom[7] = r["neck_bias"]
                ctrl = q_nom + act * r["action_scale"]
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
            
            # Forward walking speed tracking for this robot's individual target speed
            r_forward = 5.0 * math.exp(-8.0 * (vx - r["target_vel"]) ** 2)
            
            # Anti-Jumping & Anti-Hop constraints
            r_jump = -4.0 * abs(vz)
            r_bounce = -15.0 * max(0.0, z - 0.295) ** 2
            r_flight = -4.0 if min_foot_z > 0.04 else 0.0
            
            # Anti-Faceplant Pitch & Roll Leveling
            r_pitch = -7.0 * (pitch ** 2)
            r_roll = -4.0 * (roll ** 2)
            
            # Active Tail & Head Counter-Balance (cancels pitching momentum)
            r_tail = 2.5 * math.exp(-12.0 * (pitch - 0.25 * (tail_angle - r["tail_bias"])) ** 2)
            r_neck = 2.0 * math.exp(-15.0 * (pitch - 0.30 * (neck_angle - r["neck_bias"])) ** 2)
            
            # Alternating Stride foot clearance
            r_stride = 1.2 * min(0.05, abs(rf_z - lf_z))
            
            r_height = 2.0 * math.exp(-35.0 * (z - 0.28) ** 2)
            r_lat = -2.5 * abs(vy)
            r_alive = 3.0
            
            step_reward = (r_forward + r_jump + r_bounce + r_flight + 
                           r_pitch + r_roll + r_tail + r_neck + 
                           r_stride + r_height + r_lat + r_alive)
            r["reward"] += step_reward
            alive_count += 1
            total_speed += vx
            total_pitch += math.degrees(pitch)
            
            # Fall check
            if z < 0.17 or z > 0.35 or abs(pitch) > math.radians(30.0) or abs(roll) > math.radians(35.0):
                r["alive"] = False
                r["reward"] -= 15.0
                
        done = (self.step_count >= self.max_steps) or (alive_count == 0)
        avg_speed = total_speed / max(1, alive_count)
        avg_pitch = total_pitch / max(1, alive_count)
        
        return done, alive_count, avg_speed, avg_pitch


class SimplePolicy:
    """Neural Network Policy."""
    def __init__(self, obs_dim=31, act_dim=8, hidden_dim=64):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        
        self.w1 = np.random.randn(obs_dim, hidden_dim) * 0.15
        self.b1 = np.zeros(hidden_dim)
        self.w2 = np.random.randn(hidden_dim, act_dim) * 0.15
        self.b2 = np.zeros(act_dim)

    def forward(self, obs):
        h = np.tanh(np.dot(obs, self.w1) + self.b1)
        return np.tanh(np.dot(h, self.w2) + self.b2)

    def get_params(self):
        return np.concatenate([self.w1.flatten(), self.b1, self.w2.flatten(), self.b2])

    def set_params(self, params):
        idx = 0
        w1_size = self.obs_dim * self.hidden_dim
        self.w1 = params[idx:idx + w1_size].reshape(self.obs_dim, self.hidden_dim)
        idx += w1_size
        self.b1 = params[idx:idx + self.hidden_dim]
        idx += self.hidden_dim
        w2_size = self.hidden_dim * self.act_dim
        self.w2 = params[idx:idx + w2_size].reshape(self.hidden_dim, self.act_dim)
        idx += w2_size
        self.b2 = params[idx:idx + self.act_dim]

    def save(self, filepath):
        np.savez(filepath, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)

    def load(self, filepath):
        data = np.load(filepath)
        w1 = data['w1']
        min_r = min(w1.shape[0], self.w1.shape[0])
        min_c = min(w1.shape[1], self.w1.shape[1])
        self.w1[:min_r, :min_c] = w1[:min_r, :min_c]
        self.b1[:min_c] = data['b1'][:min_c]
        self.w2[:min_c, :] = data['w2'][:min_c, :]
        self.b2 = data['b2']


def train_30_diverse_walking(num_generations=150, show_viewer=True, pretrained=None, save_path="policy_walk.npz"):
    arena = DiverseWalkArena(num_robots=30)
    
    # 30 distinct policies - each initialized with distinct personality exploration
    policies = [SimplePolicy(arena.obs_dim, arena.act_dim) for _ in range(30)]
    
    # Initialize from pretrained stabilization if available
    if pretrained and os.path.exists(pretrained):
        print(f"[INFO] Initializing diverse population from checkpoint: {pretrained}")
        for p in policies:
            p.load(pretrained)
            
    # Initial diverse parameter perturbation so they immediately walk differently!
    for i, p in enumerate(policies):
        if i > 0:
            params = p.get_params()
            diversity_noise = np.random.randn(len(params)) * 0.20
            p.set_params(params + diversity_noise)
            
    best_return = -float('inf')
    champion_policy = SimplePolicy(arena.obs_dim, arena.act_dim)
    
    print("=" * 76)
    print(" 30-ROBOT DIVERSE WALKING ARENA TRAINING (HIGH DIVERSITY LOCOMOTION)")
    print(f" Population: 30 Robots with 30 Distinct Walking Personalities | Generations: {num_generations}")
    print("=" * 76)
    
    viewer = None
    if show_viewer:
        print("[INFO] Launching MuJoCo Viewer: 30 robots with different cadences, strides & postures...")
        viewer = mujoco.viewer.launch_passive(arena.model, arena.data)
        
    try:
        for gen in range(1, num_generations + 1):
            arena.reset(apply_noise=True)
            done = False
            last_alive = 30
            last_speed = 0.0
            last_pitch = 0.0
            
            # SIMULATE ALL 30 VISIBLY DIFFERENT ROBOTS TOGETHER
            while not done:
                actions = []
                for i in range(30):
                    obs_i = arena.get_robot_obs(i)
                    act_i = policies[i].forward(obs_i)
                    actions.append(act_i)
                    
                done, last_alive, last_speed, last_pitch = arena.step(actions)
                
                if viewer and viewer.is_running():
                    viewer.sync()
                    time.sleep(0.01)
                    
            returns = np.array([r["reward"] for r in arena.robots])
            distances = np.array([r["dist"] for r in arena.robots])
            
            avg_score = np.mean(returns)
            best_score = np.max(returns)
            best_idx = np.argmax(returns)
            best_robot = arena.robots[best_idx]
            max_dist = distances[best_idx]
            
            # Save the current overall champion
            if best_score > best_return:
                best_return = best_score
                champion_policy.set_params(policies[best_idx].get_params())
                champion_policy.save(save_path)
                
            # Genetic selection & diversity crossover across niches:
            # Rank robots by performance
            sorted_indices = np.argsort(returns)[::-1]
            top_elites = sorted_indices[:5]  # Top 5 champion robots
            
            # Reproduce next generation: Top elites preserve and cross-pollinate, while maintaining individual diversity!
            elite_params = [policies[idx].get_params() for idx in top_elites]
            
            for i in range(30):
                if i in top_elites:
                    # Elites keep their winning weights with small refinement noise
                    p_base = policies[i].get_params()
                    policies[i].set_params(p_base + np.random.randn(len(p_base)) * 0.03)
                else:
                    # Non-elites sample from a random top elite and mutate with exploration noise
                    parent = elite_params[np.random.randint(len(elite_params))]
                    mutation_rate = 0.15 + 0.10 * (i / 30.0)
                    new_params = parent + np.random.randn(len(parent)) * mutation_rate
                    policies[i].set_params(new_params)
                    
            # Display current Epoch/Iteration banner directly in the MuJoCo Viewer
            if viewer and viewer.is_running():
                try:
                    sim = viewer._get_sim()
                    if sim is not None:
                        sim.load_message(f"EPOCH: {gen:04d} / {num_generations:04d}  |  Alive: {last_alive}/30  |  Best Dist: {max_dist:.2f}m")
                except Exception:
                    pass

            print(f"[EPOCH {gen:04d} / {num_generations:04d}] | Alive: {last_alive:02d}/30 | Best: R{best_idx:02d} [{best_robot['niche']}] | Dist: {max_dist:5.2f}m | Speed: {last_speed:4.2f}m/s | Score: {best_score:6.1f}")
            
    except KeyboardInterrupt:
        print("\n[INFO] Training paused by user.")
    finally:
        champion_policy.save(save_path)
        if viewer:
            viewer.close()
        print("=" * 76)
        print(f" Training Complete! Best walking policy saved to: {save_path}")
        print("=" * 76)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="30-Robot Diverse Walking Arena Training")
    parser.add_argument("--headless", action="store_true", help="Fast training without GUI viewer")
    parser.add_argument("--generations", type=int, default=1000, help="Training generations (epochs)")
    parser.add_argument("--pretrained", type=str, default="policy_stabilize.npz", help="Pretrained balance policy")
    parser.add_argument("--save-path", type=str, default="policy_walk.npz", help="Save file")
    args = parser.parse_args()
    
    train_30_diverse_walking(
        num_generations=args.generations,
        show_viewer=not args.headless,
        pretrained=args.pretrained,
        save_path=args.save_path
    )
