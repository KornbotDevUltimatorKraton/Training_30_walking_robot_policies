"""
train_stabilize.py - Training bipedal robot to balance and stabilize upright in MuJoCo.
UPGRADED: Active Tail & Head Pitch-Balancing + Anti-Jumping Dynamics!

Usage:
  python train_stabilize.py                    # Train with live MuJoCo viewer showing each iteration
  python train_stabilize.py --headless         # Train fast without GUI window
  python train_stabilize.py --eval             # Test trained policy in viewer with mouse/push perturbations
"""

import os
import sys
import time
import math
import argparse
import numpy as np
import mujoco
import mujoco.viewer

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "urdf", "bipedal_training.xml")

class BipedalStabilizeEnv:
    """Gym-compatible environment for Bipedal Robot Stabilization & Balancing."""
    
    def __init__(self, model_path=MODEL_PATH):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        self.target_height = 0.28
        self.action_dim = self.model.nu  # 8 actuators
        self.obs_dim = 28               # State representation
        
        self.q_nominal = np.zeros(self.action_dim)
        # Strong Tail (0.70) & Neck (0.55), controlled Knees (0.35)
        self.action_scale = np.array([0.30, 0.45, 0.35, 0.30, 0.45, 0.35, 0.70, 0.55])
        
        self.step_count = 0
        self.max_steps = 600

    def get_obs(self):
        base_pos_z = self.data.qpos[2:3]
        base_quat = self.data.qpos[3:7]
        
        w, x, y, z = base_quat
        grav_x = 2 * (x * z - w * y)
        grav_y = 2 * (y * z + w * x)
        grav_z = w * w - x * x - y * y + z * z
        proj_gravity = np.array([grav_x, grav_y, grav_z])
        
        base_linvel = self.data.qvel[0:3]
        base_angvel = self.data.qvel[3:6]
        
        joint_pos = self.data.qpos[7:7 + self.action_dim] - self.q_nominal
        joint_vel = self.data.qvel[6:6 + self.action_dim]
        
        obs = np.concatenate([
            base_pos_z,
            proj_gravity,
            base_linvel,
            base_angvel,
            joint_pos,
            joint_vel
        ]).astype(np.float32)
        if len(obs) < self.obs_dim:
            obs = np.pad(obs, (0, self.obs_dim - len(obs)))
        return obs

    def reset(self, apply_noise=True):
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = self.target_height + (np.random.uniform(-0.01, 0.01) if apply_noise else 0.0)
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        
        noise = np.random.uniform(-0.02, 0.02, self.action_dim) if apply_noise else 0.0
        self.data.qpos[7:7 + self.action_dim] = self.q_nominal + noise
        self.data.qvel[:] = 0.0
        
        mujoco.mj_forward(self.model, self.data)
        return self.get_obs()

    def step(self, action, apply_perturbations=True):
        self.step_count += 1
        action = np.clip(action, -1.0, 1.0)
        
        # Occasional horizontal push to train balance recovery
        if apply_perturbations and self.step_count % 120 == 0:
            push = np.random.uniform(-6.0, 6.0, 2)
            self.data.qfrc_applied[0:2] = push
        else:
            self.data.qfrc_applied[:] = 0.0
        
        target_ctrl = self.q_nominal + action * self.action_scale
        self.data.ctrl[:] = target_ctrl
        
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
            
        obs = self.get_obs()
        
        height = self.data.qpos[2]
        vz = self.data.qvel[2]
        w, x, y, z = self.data.qpos[3:7]
        
        # Pitch angle (nose-down is positive)
        sin_pitch = 2.0 * (w * y - x * z)
        pitch = math.asin(max(-1.0, min(1.0, sin_pitch)))
        roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        
        tail_angle = self.data.qpos[7 + 6]
        neck_angle = self.data.qpos[7 + 7]
        
        # 1. Torso Pitch Leveling & Anti-Faceplant
        r_pitch = -7.0 * (pitch ** 2)
        r_roll = -4.0 * (roll ** 2)
        
        # 2. Active Tail & Neck Balancing
        r_tail_balance = 2.5 * math.exp(-12.0 * (pitch - 0.25 * tail_angle) ** 2)
        r_head_balance = 2.0 * math.exp(-15.0 * (pitch - 0.30 * neck_angle) ** 2)
        
        # 3. Anti-Jumping / Anti-Hop Penalty
        r_jump = -4.0 * abs(vz)
        r_bounce = -15.0 * max(0.0, height - 0.295) ** 2
        
        # 4. Height & Drift suppression
        r_height = 2.5 * math.exp(-35.0 * (height - self.target_height) ** 2)
        r_drift = -2.0 * (self.data.qvel[0] ** 2 + self.data.qvel[1] ** 2)
        r_angvel = -0.1 * np.sum(self.data.qvel[3:6] ** 2)
        r_energy = -0.005 * np.sum(action ** 2)
        r_alive = 3.0
        
        reward = (r_pitch + r_roll + r_tail_balance + r_head_balance + 
                  r_jump + r_bounce + r_height + r_drift + r_angvel + r_energy + r_alive)
        
        done = False
        if height < 0.17 or height > 0.35:
            done = True
            reward -= 15.0
        if abs(pitch) > math.radians(30.0):
            done = True
            reward -= 15.0
        if abs(roll) > math.radians(35.0):
            done = True
            reward -= 15.0
        if self.step_count >= self.max_steps:
            done = True
            
        return obs, reward, done, {
            "pitch_deg": math.degrees(pitch),
            "height": height
        }


class SimplePolicy:
    """Lightweight 2-layer Neural Network Policy with Fast Evolutionary Optimization."""
    def __init__(self, obs_dim=28, act_dim=8, hidden_dim=64):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        
        self.w1 = np.random.randn(obs_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.w2 = np.random.randn(hidden_dim, act_dim) * 0.1
        self.b2 = np.zeros(act_dim)

    def forward(self, obs):
        h = np.tanh(np.dot(obs, self.w1) + self.b1)
        out = np.tanh(np.dot(h, self.w2) + self.b2)
        return out

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


def train_stabilization(num_generations=150, pop_size=24, show_viewer=True, save_path="policy_stabilize.npz"):
    env = BipedalStabilizeEnv()
    policy = SimplePolicy(env.obs_dim, env.action_dim)
    params = policy.get_params()
    
    sigma = 0.05
    learning_rate = 0.03
    best_return = -float('inf')
    
    print("=" * 68)
    print(" BIPEDAL STABILIZATION TRAINING (ACTIVE TAIL & HEAD BALANCE)")
    print(f" Generations: {num_generations} | Population: {pop_size} | Live Viewer: {show_viewer}")
    print("=" * 68)
    
    viewer = None
    if show_viewer:
        print("[INFO] Launching interactive MuJoCo viewer window...")
        viewer = mujoco.viewer.launch_passive(env.model, env.data)
        
    try:
        for gen in range(1, num_generations + 1):
            noise = np.random.randn(pop_size, len(params))
            returns = np.zeros(pop_size)
            pitch_angles = np.zeros(pop_size)
            
            # 1. Candidate evaluation
            for i in range(pop_size):
                candidate_params = params + sigma * noise[i]
                policy.set_params(candidate_params)
                
                obs = env.reset(apply_noise=True)
                total_reward = 0.0
                done = False
                last_info = {}
                
                while not done:
                    action = policy.forward(obs)
                    obs, reward, done, last_info = env.step(action, apply_perturbations=True)
                    total_reward += reward
                    
                returns[i] = total_reward
                pitch_angles[i] = last_info.get("pitch_deg", 0.0)
                
            norm_returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
            grad = np.dot(noise.T, norm_returns) / (pop_size * sigma)
            params += learning_rate * grad
            policy.set_params(params)
            
            avg_ret = np.mean(returns)
            max_ret = np.max(returns)
            avg_pitch = np.mean(pitch_angles)
            
            if max_ret > best_return:
                best_return = max_ret
                policy.save(save_path)
                
            print(f"Iteration {gen:03d} | Avg Score: {avg_ret:7.1f} | Best: {max_ret:7.1f} | Pitch: {avg_pitch:+5.1f}° | Saved: {save_path}")
            
            # 2. LIVE VISUAL DEMO IN VIEWER: Watch the updated champion robot stabilize with active tail/head
            if viewer and viewer.is_running():
                obs = env.reset(apply_noise=False)
                done = False
                steps = 0
                while not done and steps < 250:
                    action = policy.forward(obs)
                    obs, _, done, _ = env.step(action, apply_perturbations=True)
                    viewer.sync()
                    time.sleep(0.01)
                    steps += 1
                    
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user. Latest policy saved.")
    finally:
        policy.save(save_path)
        if viewer:
            viewer.close()
        print("=" * 68)
        print(f" Training Complete! Best stabilization policy saved to: {save_path}")
        print("=" * 68)


def evaluate_policy(save_path="policy_stabilize.npz"):
    env = BipedalStabilizeEnv()
    policy = SimplePolicy(env.obs_dim, env.action_dim)
    
    if os.path.exists(save_path):
        policy.load(save_path)
        print(f"Loaded weights from {save_path}")
    else:
        print(f"Warning: {save_path} not found. Running with default initial weights.")
        
    print("\nRunning interactive stabilization test in MuJoCo viewer!")
    print("Press Ctrl+C or close viewer to exit.\n")
    
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            obs = env.reset(apply_noise=False)
            done = False
            total_reward = 0
            info = {}
            while not done and viewer.is_running():
                action = policy.forward(obs)
                obs, reward, done, info = env.step(action, apply_perturbations=True)
                total_reward += reward
                viewer.sync()
                time.sleep(0.01)
            print(f"Episode completed | Balance score: {total_reward:.1f} | Pitch: {info.get('pitch_deg', 0):+.1f}°")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bipedal Robot Stabilization & Balancing Training")
    parser.add_argument("--headless", action="store_true", help="Run fast training without opening the GUI viewer")
    parser.add_argument("--eval", action="store_true", help="Evaluate the trained policy in the interactive viewer")
    parser.add_argument("--generations", type=int, default=120, help="Number of training generations")
    parser.add_argument("--save-path", type=str, default="policy_stabilize.npz", help="Path to save weights")
    args = parser.parse_args()
    
    if args.eval:
        evaluate_policy(args.save_path)
    else:
        train_stabilization(
            num_generations=args.generations,
            show_viewer=not args.headless,
            save_path=args.save_path
        )
