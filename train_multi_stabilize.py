"""
train_multi_stabilize.py
DIVERSE MULTI-ROBOT STABILIZATION: 30 Bipedal Robots training with 30 DIFFERENT balance strategies!
Each robot has its own unique posture bias, tail counter-balance angle, and recovery dynamics.

Usage:
  python train_multi_stabilize.py                    # Train 30 diverse robots with live viewer
  python train_multi_stabilize.py --headless         # Fast training without GUI
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

class DiverseStabilizeArena:
    """Vectorized Environment with 30 Diverse Stabilization Strategies."""
    def __init__(self, model_path=ARENA_PATH, num_robots=30):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.num_robots = num_robots
        
        self.act_dim = 8
        self.obs_dim = 28
        
        self.robots = []
        rows = 5
        cols = 6
        x_spacing = 1.4
        y_spacing = 1.4
        
        balance_styles = [
            "Tail-Dominant Counter-Balance",
            "Wide-Stance Hip Roll Damping",
            "Knee Flexion Shock Absorber",
            "Neck Upright Horizon Tracker",
            "Stiff Rapid Equilibrium"
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
            
            style_idx = i // 6
            style = balance_styles[style_idx]
            
            tail_bias = -0.30 + 0.60 * ((i % 6) / 5.0)
            neck_bias = -0.20 + 0.40 * ((i % 5) / 4.0)
            hip_scale = 0.30 + 0.20 * ((i % 5) / 4.0)
            tail_scale = 0.50 + 0.35 * ((i % 6) / 5.0)
            
            action_scale = np.array([hip_scale, 0.45, 0.35, hip_scale, 0.45, 0.35, tail_scale, 0.55])
            
            self.robots.append({
                "id": i,
                "prefix": prefix,
                "style": style,
                "start_x": start_x,
                "start_y": start_y,
                "qpos_adr": qpos_adr,
                "qvel_adr": qvel_adr,
                "act_start": act_start,
                "tail_bias": tail_bias,
                "neck_bias": neck_bias,
                "action_scale": action_scale,
                "alive": True,
                "reward": 0.0
            })
            
        self.step_count = 0
        self.max_steps = 350

    def reset(self, apply_noise=True):
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        
        for r in self.robots:
            r["alive"] = True
            r["reward"] = 0.0
            adr = r["qpos_adr"]
            
            self.data.qpos[adr + 0] = r["start_x"]
            self.data.qpos[adr + 1] = r["start_y"]
            self.data.qpos[adr + 2] = 0.28 + (np.random.uniform(-0.01, 0.01) if apply_noise else 0.0)
            self.data.qpos[adr + 3 : adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
            
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
        
        obs = np.concatenate([
            base_z,
            proj_gravity,
            linvel,
            angvel,
            joint_pos,
            joint_vel
        ]).astype(np.float32)
        if len(obs) < self.obs_dim:
            obs = np.pad(obs, (0, self.obs_dim - len(obs)))
        return obs

    def step(self, actions, apply_perturbations=True):
        self.step_count += 1
        
        # Random horizontal pushes
        if apply_perturbations and self.step_count % 80 == 0:
            for r in self.robots:
                push = np.random.uniform(-5.0, 5.0, 2)
                v_adr = r["qvel_adr"]
                self.data.qfrc_applied[v_adr + 0 : v_adr + 2] = push
        else:
            self.data.qfrc_applied[:] = 0.0
            
        for i, r in enumerate(self.robots):
            act_start = r["act_start"]
            if r["alive"]:
                act = np.clip(actions[i], -1.0, 1.0)
                q_nom = np.zeros(self.act_dim)
                q_nom[6] = r["tail_bias"]
                q_nom[7] = r["neck_bias"]
                ctrl = q_nom + act * r["action_scale"]
                self.data.ctrl[act_start : act_start + self.act_dim] = ctrl
            else:
                self.data.ctrl[act_start : act_start + self.act_dim] = np.zeros(self.act_dim)
                
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
            
        alive_count = 0
        total_pitch = 0.0
        
        for i, r in enumerate(self.robots):
            if not r["alive"]:
                continue
                
            q_adr = r["qpos_adr"]
            v_adr = r["qvel_adr"]
            
            z = self.data.qpos[q_adr + 2]
            vz = self.data.qvel[v_adr + 2]
            w, x, y, qw_z = self.data.qpos[q_adr + 3 : q_adr + 7]
            
            sin_pitch = 2.0 * (w * y - x * qw_z)
            pitch = math.asin(max(-1.0, min(1.0, sin_pitch)))
            roll = math.atan2(2.0 * (w * x + y * qw_z), 1.0 - 2.0 * (x * x + y * y))
            total_pitch += math.degrees(pitch)
            
            tail_angle = self.data.qpos[q_adr + 7 + 6]
            neck_angle = self.data.qpos[q_adr + 7 + 7]
            
            r_pitch = -7.0 * (pitch ** 2)
            r_roll = -4.0 * (roll ** 2)
            r_tail = 2.5 * math.exp(-12.0 * (pitch - 0.25 * (tail_angle - r["tail_bias"])) ** 2)
            r_neck = 2.0 * math.exp(-15.0 * (pitch - 0.30 * (neck_angle - r["neck_bias"])) ** 2)
            r_jump = -4.0 * abs(vz)
            r_height = 2.5 * math.exp(-35.0 * (z - 0.28) ** 2)
            r_drift = -2.0 * (self.data.qvel[v_adr] ** 2 + self.data.qvel[v_adr + 1] ** 2)
            r_alive = 3.0
            
            step_reward = r_pitch + r_roll + r_tail + r_neck + r_jump + r_height + r_drift + r_alive
            r["reward"] += step_reward
            alive_count += 1
            
            if z < 0.17 or z > 0.35 or abs(pitch) > math.radians(30.0) or abs(roll) > math.radians(35.0):
                r["alive"] = False
                r["reward"] -= 10.0
                
        done = (self.step_count >= self.max_steps) or (alive_count == 0)
        avg_pitch = total_pitch / max(1, alive_count)
        
        return done, alive_count, avg_pitch


class SimplePolicy:
    """Neural Network Policy."""
    def __init__(self, obs_dim=28, act_dim=8, hidden_dim=64):
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
        self.w1 = data['w1']
        self.b1 = data['b1']
        self.w2 = data['w2']
        self.b2 = data['b2']


def train_30_diverse_stabilization(num_generations=120, show_viewer=True, save_path="policy_stabilize.npz"):
    arena = DiverseStabilizeArena(num_robots=30)
    policies = [SimplePolicy(arena.obs_dim, arena.act_dim) for _ in range(30)]
    champion_policy = SimplePolicy(arena.obs_dim, arena.act_dim)
    
    # Diverse initial exploration weights
    for i, p in enumerate(policies):
        if i > 0:
            params = p.get_params()
            p.set_params(params + np.random.randn(len(params)) * 0.18)
            
    best_return = -float('inf')
    
    print("=" * 76)
    print(" 30-ROBOT DIVERSE STABILIZATION ARENA (MULTIPLE BALANCE STRATEGIES)")
    print(f" Population: 30 Robots with Distinct Posture & Recovery Styles | Generations: {num_generations}")
    print("=" * 76)
    
    viewer = None
    if show_viewer:
        print("[INFO] Opening MuJoCo Arena Viewer with 30 diverse stabilizing robots...")
        viewer = mujoco.viewer.launch_passive(arena.model, arena.data)
        
    try:
        for gen in range(1, num_generations + 1):
            arena.reset(apply_noise=True)
            done = False
            last_alive = 30
            last_pitch = 0.0
            
            while not done:
                actions = []
                for i in range(30):
                    obs_i = arena.get_robot_obs(i)
                    act_i = policies[i].forward(obs_i)
                    actions.append(act_i)
                    
                done, last_alive, last_pitch = arena.step(actions, apply_perturbations=True)
                
                if viewer and viewer.is_running():
                    viewer.sync()
                    time.sleep(0.01)
                    
            returns = np.array([r["reward"] for r in arena.robots])
            avg_score = np.mean(returns)
            best_score = np.max(returns)
            best_idx = np.argmax(returns)
            best_robot = arena.robots[best_idx]
            
            if best_score > best_return:
                best_return = best_score
                champion_policy.set_params(policies[best_idx].get_params())
                champion_policy.save(save_path)
                
            sorted_indices = np.argsort(returns)[::-1]
            top_elites = sorted_indices[:5]
            elite_params = [policies[idx].get_params() for idx in top_elites]
            
            for i in range(30):
                if i in top_elites:
                    p_base = policies[i].get_params()
                    policies[i].set_params(p_base + np.random.randn(len(p_base)) * 0.03)
                else:
                    parent = elite_params[np.random.randint(len(elite_params))]
                    mutation_rate = 0.12 + 0.10 * (i / 30.0)
                    policies[i].set_params(parent + np.random.randn(len(parent)) * mutation_rate)
                    
            print(f"Gen {gen:03d} | Alive: {last_alive:02d}/30 | Best: R{best_idx:02d} [{best_robot['style']}] | Score: {best_score:6.1f} | Pitch: {last_pitch:+5.1f}°")
            
    except KeyboardInterrupt:
        print("\n[INFO] Training paused by user.")
    finally:
        champion_policy.save(save_path)
        if viewer:
            viewer.close()
        print("=" * 76)
        print(f" Diverse Stabilization Complete! Best Policy: {save_path}")
        print("=" * 76)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="30-Robot Diverse Stabilization Training")
    parser.add_argument("--headless", action="store_true", help="Fast training without GUI")
    parser.add_argument("--generations", type=int, default=120, help="Training generations")
    parser.add_argument("--save-path", type=str, default="policy_stabilize.npz", help="Save file")
    args = parser.parse_args()
    
    train_30_diverse_stabilization(
        num_generations=args.generations,
        show_viewer=not args.headless,
        save_path=args.save_path
    )
