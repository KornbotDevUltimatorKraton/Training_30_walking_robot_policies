"""
view_champion.py - Standalone MuJoCo 3D Viewer & Telemetry for Trained Walking Policy
Demonstrates the trained champion policy from 'policy_walk.npz' walking and stabilizing.
"""

import os
import sys
import time
import math
import argparse
import numpy as np
import mujoco
import mujoco.viewer

from train_walk_and_stabilize import CombinedWalkStabilizeArena, BatchedGpuPolicy

def view_champion(policy_path="policy_walk.npz", num_cycles=5, max_steps=1000):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(policy_path):
        policy_path = os.path.join(script_dir, policy_path)
        
    if not os.path.exists(policy_path):
        print(f"[ERROR] Checkpoint not found: {policy_path}")
        print("Please run training first: python train_walk_and_stabilize.py")
        sys.exit(1)
        
    print("=" * 80)
    print(" MUJOCO 3D PLAYBACK: TRAINED BIPEDAL WALKING & STABILIZATION CHAMPION")
    print(f" Loading Policy: {policy_path}")
    print("=" * 80)
    
    arena = CombinedWalkStabilizeArena(num_robots=30)
    policy = BatchedGpuPolicy(num_robots=30, obs_dim=arena.obs_dim, act_dim=arena.act_dim)
    policy.load_pretrained(policy_path, arena=arena)
    
    data = np.load(policy_path)
    print("\n[CHAMPION HARDWARE GAIT PARAMETERS]")
    for k in ['freq', 'target_vel', 'tail_bias', 'neck_bias', 'knee_scale', 'hip_scale', 'roll_scale']:
        if k in data:
            print(f"  {k:15s}: {float(data[k]):.4f}")
    print("-" * 80)
    print("Launching MuJoCo 3D Viewer Window...")
    print("Press Ctrl+C or close viewer to exit.\n")
    
    viewer = mujoco.viewer.launch_passive(arena.model, arena.data)
    
    try:
        for cycle in range(1, num_cycles + 1):
            arena.reset(apply_noise=False)
            r0 = arena.robots[0]
            q_adr = r0["qpos_adr"]
            v_adr = r0["qvel_adr"]
            
            print(f"--- [RUN CYCLE {cycle:02d} / {num_cycles:02d}] ---")
            
            for step in range(max_steps):
                if not viewer.is_running():
                    break
                    
                obs_batch = arena.get_all_obs()
                actions_batch = policy.forward_batch(obs_batch)
                done, alive_cnt, avg_spd, avg_pitch = arena.step(actions_batch)
                
                # Telemetry for Champion Robot 0
                dist = r0["dist"]
                vy = arena.data.qvel[v_adr + 1]
                z = arena.data.qpos[q_adr + 2]
                head_z = arena.data.xpos[r0["head_id"], 2]
                
                # Quat orientation
                w, x, y, qw_z = arena.data.qpos[q_adr + 3 : q_adr + 7]
                r21 = 2.0 * (y * qw_z + w * x)
                pitch = math.degrees(math.asin(max(-1.0, min(1.0, -r21))))
                
                # Update MuJoCo HUD message
                try:
                    sim = viewer._get_sim()
                    if sim is not None:
                        sim.load_message(
                            f"CHAMPION R00: Dist {dist:+.3f}m | Swarm Active: {alive_cnt:02d}/30 | "
                            f"Speed: {avg_spd:+.2f}m/s | Head: {head_z*100:.1f}cm"
                        )
                except Exception:
                    pass
                
                # Golden beacon over champion
                if viewer.user_scn is not None:
                    try:
                        leader_pos = arena.data.qpos[q_adr : q_adr + 3].copy()
                        leader_pos[2] += 0.35
                        mujoco.mjv_initGeom(
                            viewer.user_scn.geoms[0],
                            mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([0.06, 0.06, 0.06]),
                            leader_pos,
                            np.eye(3).flatten(),
                            np.array([1.0, 0.85, 0.1, 0.9])
                        )
                        viewer.user_scn.ngeom = 1
                    except Exception:
                        pass
                
                viewer.sync()
                time.sleep(0.008)  # Smooth real-time visualization pace
                
                if step % 50 == 0:
                    status_str = "WALKING & UPRIGHT" if r0["alive"] else "TERMINATED"
                    print(f"  Step {step:03d}: Champion Dist={dist:+.3f}m | Swarm Active={alive_cnt:02d}/30 | Speed={avg_spd:+.2f}m/s | Torso={z:.3f}m | Pitch={pitch:+.1f} deg | Status={status_str}")
                    
                if done:
                    print(f"  [CYCLE {cycle:02d} END] Completed at step {step} with Champion distance {dist:+.3f}m | Swarm Active={alive_cnt}/30\n")
                    time.sleep(1.0)
                    break
                    
    except KeyboardInterrupt:
        print("\n[INFO] Playback exited by user.")
    finally:
        if viewer:
            viewer.close()
        print("\nMuJoCo Playback Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View Trained Walking & Stabilization Champion in MuJoCo GUI")
    parser.add_argument("--policy", type=str, default="policy_walk.npz", help="Path to policy checkpoint")
    parser.add_argument("--cycles", type=int, default=100, help="Number of playback demonstration cycles")
    parser.add_argument("--steps", type=int, default=600, help="Max steps per cycle")
    args = parser.parse_args()
    
    view_champion(policy_path=args.policy, num_cycles=args.cycles, max_steps=args.steps)
