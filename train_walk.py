"""
train_walk.py - Training bipedal robot to walk forward in MuJoCo.
UPGRADED: Active Tail & Head Pitch-Balancing + Anti-Jumping & Smooth Stride Dynamics!

Usage:
  python train_walk.py                         # Train with live MuJoCo viewer showing each iteration
  python train_walk.py --headless              # Train fast without GUI window
  python train_walk.py --eval                  # Test trained walking policy in viewer
  python train_walk.py --pretrained policy_stabilize.npz # Warm-start from balance policy!
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

class BipedalWalkEnv:
    """Gym-compatible environment for Bipedal Robot Forward Walking Locomotion."""
    
    def __init__(self, model_path=MODEL_PATH, target_vel=0.35):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        self.target_vel = target_vel     # Desired forward speed (m/s)
        self.target_height = 0.28
        self.action_dim = self.model.nu  # 8 actuators
        self.obs_dim = 29                # State vector + command velocity
        
        # Nominal default standing posture
        self.q_nominal = np.zeros(self.action_dim)
        # Action scale: Strong Tail (0.70) & Neck (0.55), Controlled Knees (0.35 to prevent jumping!)
        self.action_scale = np.array([0.30, 0.45, 0.35, 0.30, 0.45, 0.35, 0.70, 0.55])
        
        # Foot body IDs for tracking contact and stride
        self.foot_right_id = self.model.body("body1_20").id
        self.foot_left_id = self.model.body("body1_36").id
        
        self.step_count = 0
        self.max_steps = 800

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
        cmd_vel = np.array([self.target_vel])
        
        obs = np.concatenate([
            base_pos_z,
            proj_gravity,
            base_linvel,
            base_angvel,
            joint_pos,
            joint_vel,
            cmd_vel
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

    def step(self, action):
        self.step_count += 1
        action = np.clip(action, -1.0, 1.0)
        
        target_ctrl = self.q_nominal + action * self.action_scale
        self.data.ctrl[:] = target_ctrl
        
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
            
        obs = self.get_obs()
        
        # State readings
        vx = self.data.qvel[0]
        vy = self.data.qvel[1]
        vz = self.data.qvel[2]
        height = self.data.qpos[2]
        
        # Orientation & Pitch computation
        w, x, y, z = self.data.qpos[3:7]
        # Pitch angle (nose-down is positive, nose-up is negative)
        sin_pitch = 2.0 * (w * y - x * z)
        pitch = math.asin(max(-1.0, min(1.0, sin_pitch)))
        roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        
        # Joint readings for tail and neck
        tail_angle = self.data.qpos[7 + 6]
        neck_angle = self.data.qpos[7 + 7]
        
        # Feet altitudes
        rf_z = self.data.xpos[self.foot_right_id, 2]
        lf_z = self.data.xpos[self.foot_left_id, 2]
        min_foot_z = min(rf_z, lf_z)
        
        # ======================================================================
        # REWARD FORMULATION (STRICT BALANCE & ANTI-JUMP)
        # ======================================================================
        # 1. Forward Speed Tracking
        r_forward = 5.0 * math.exp(-8.0 * (vx - self.target_vel) ** 2)
        
        # 2. SEVERE ANTI-JUMPING PENALTY (Bipeds must WALK, not hop vertically!)
        r_jump = -4.0 * abs(vz)
        r_bounce = -15.0 * max(0.0, height - 0.295) ** 2
        # Penalize airborne hopping if both feet leave the ground simultaneously
        r_flight = -4.0 if min_foot_z > 0.04 else 0.0
        
        # 3. PITCH-DOWN / FLIPPING-DOWN PREVENTION (Keep Head & Torso Level)
        r_pitch = -7.0 * (pitch ** 2)
        r_roll = -4.0 * (roll ** 2)
        
        # 4. ACTIVE TAIL & HEAD DYNAMIC COUNTER-BALANCING
        # When torso tries to pitch down, reward raising the tail and lifting the neck to restore level horizon!
        r_tail_balance = 2.5 * math.exp(-12.0 * (pitch - 0.25 * tail_angle) ** 2)
        r_head_balance = 2.0 * math.exp(-15.0 * (pitch - 0.30 * neck_angle) ** 2)
        
        # 5. Continuous Smooth Walking Stride (Alternating Foot Clearance)
        stride_diff = abs(rf_z - lf_z)
        r_stride = 1.2 * min(0.05, stride_diff)
        
        # 6. Stability & Height Maintenance
        r_height = 2.0 * math.exp(-35.0 * (height - self.target_height) ** 2)
        r_lateral = -2.5 * abs(vy)
        r_energy = -0.005 * np.sum(action ** 2)
        r_alive = 3.0
        
        reward = (r_forward + r_jump + r_bounce + r_flight + 
                  r_pitch + r_roll + r_tail_balance + r_head_balance + 
                  r_stride + r_height + r_lateral + r_energy + r_alive)
        
        # ======================================================================
        # EARLY TERMINATION (Strict Fall Detection)
        # ======================================================================
        done = False
        # Fall detection: if height collapses or launches into air
        if height < 0.17 or height > 0.35:
            done = True
            reward -= 15.0
        # Head flipping down detection: stop episode immediately if pitch > 30 degrees!
        if abs(pitch) > math.radians(30.0):
            done = True
            reward -= 15.0
        if abs(roll) > math.radians(35.0):
            done = True
            reward -= 15.0
        if self.step_count >= self.max_steps:
            done = True
            
        return obs, reward, done, {
            "forward_vel": vx,
            "pitch_deg": math.degrees(pitch),
            "pos_x": self.data.qpos[0]
        }


class SimplePolicy:
    """Lightweight 2-layer Neural Network Policy with Fast Evolutionary Optimization."""
    def __init__(self, obs_dim=29, act_dim=8, hidden_dim=64):
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
        w1 = data['w1']
        if w1.shape == self.w1.shape:
            self.w1 = w1
        else:
            min_r = min(w1.shape[0], self.w1.shape[0])
            self.w1[:min_r, :] = w1[:min_r, :]
        self.b1 = data['b1']
        self.w2 = data['w2']
        self.b2 = data['b2']


def train_walking(num_generations=150, pop_size=24, target_vel=0.35, show_viewer=True, pretrained=None, save_path="policy_walk.npz"):
    env = BipedalWalkEnv(target_vel=target_vel)
    policy = SimplePolicy(env.obs_dim, env.action_dim)
    
    if pretrained and os.path.exists(pretrained):
        print(f"[INFO] Warm-starting from pretrained checkpoint: {pretrained}")
        policy.load(pretrained)
        
    params = policy.get_params()
    sigma = 0.05
    learning_rate = 0.03
    best_return = -float('inf')
    
    print("=" * 68)
    print(" BIPEDAL ROBOT WALKING TRAINING (ACTIVE TAIL & HEAD BALANCE)")
    print(f" Target Speed: {target_vel} m/s | Generations: {num_generations} | Live Viewer: {show_viewer}")
    print("=" * 68)
    
    viewer = None
    if show_viewer:
        print("[INFO] Launching interactive MuJoCo viewer window...")
        viewer = mujoco.viewer.launch_passive(env.model, env.data)
        
    try:
        for gen in range(1, num_generations + 1):
            noise = np.random.randn(pop_size, len(params))
            returns = np.zeros(pop_size)
            forward_speeds = np.zeros(pop_size)
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
                    obs, reward, done, last_info = env.step(action)
                    total_reward += reward
                    
                returns[i] = total_reward
                forward_speeds[i] = last_info.get("forward_vel", 0.0)
                pitch_angles[i] = last_info.get("pitch_deg", 0.0)
                
            norm_returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
            grad = np.dot(noise.T, norm_returns) / (pop_size * sigma)
            params += learning_rate * grad
            policy.set_params(params)
            
            avg_ret = np.mean(returns)
            max_ret = np.max(returns)
            avg_vel = np.mean(forward_speeds)
            avg_pitch = np.mean(pitch_angles)
            
            if max_ret > best_return:
                best_return = max_ret
                policy.save(save_path)
                
            print(f"Iteration {gen:03d} | Avg Score: {avg_ret:7.1f} | Best: {max_ret:7.1f} | Speed: {avg_vel:5.2f} m/s | Pitch: {avg_pitch:+5.1f}°")
            
            # 2. LIVE VISUAL DEMO IN VIEWER: Watch the updated champion robot walk with active tail/head balance!
            if viewer and viewer.is_running():
                obs = env.reset(apply_noise=False)
                done = False
                steps = 0
                while not done and steps < 250:
                    action = policy.forward(obs)
                    obs, _, done, _ = env.step(action)
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
        print(f" Walking Training Complete! Saved Policy: {save_path}")
        print("=" * 68)


def evaluate_walking(save_path="policy_walk.npz", target_vel=0.35):
    env = BipedalWalkEnv(target_vel=target_vel)
    policy = SimplePolicy(env.obs_dim, env.action_dim)
    
    if os.path.exists(save_path):
        policy.load(save_path)
        print(f"Loaded weights from {save_path}")
    else:
        print(f"Warning: {save_path} not found. Running with default weights.")
        
    print(f"\nRunning walking simulation in MuJoCo viewer (Target speed: {target_vel} m/s)!")
    print("Press Ctrl+C or close viewer to exit.\n")
    
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            obs = env.reset(apply_noise=False)
            done = False
            total_reward = 0
            info = {}
            while not done and viewer.is_running():
                action = policy.forward(obs)
                obs, reward, done, info = env.step(action)
                total_reward += reward
                viewer.sync()
                time.sleep(0.01)
            print(f"Episode finished | Walk score: {total_reward:.1f} | Forward velocity: {info.get('forward_vel', 0):.2f} m/s | Pitch: {info.get('pitch_deg', 0):+.1f}° | Distance: {info.get('pos_x', 0):.2f} m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bipedal Robot Walking Training with Tail & Head Balance")
    parser.add_argument("--headless", action="store_true", help="Run fast training without opening the GUI viewer")
    parser.add_argument("--eval", action="store_true", help="Evaluate trained walking policy in interactive viewer")
    parser.add_argument("--generations", type=int, default=150, help="Number of training generations")
    parser.add_argument("--target-vel", type=float, default=0.35, help="Target forward walking speed in m/s")
    parser.add_argument("--pretrained", type=str, default="policy_stabilize.npz", help="Path to pretrained stabilization policy")
    parser.add_argument("--save-path", type=str, default="policy_walk.npz", help="Path to save walking weights")
    args = parser.parse_args()
    
    if args.eval:
        evaluate_walking(args.save_path, target_vel=args.target_vel)
    else:
        train_walking(
            num_generations=args.generations,
            target_vel=args.target_vel,
            show_viewer=not args.headless,
            pretrained=args.pretrained,
            save_path=args.save_path
        )
