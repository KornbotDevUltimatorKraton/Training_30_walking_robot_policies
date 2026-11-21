"""
deploy_feetech_orangepi.py
Real-world deployment controller for Orange Pi Zero 3W (Ubuntu / Linux)
Hardware:
  - IMU: MPU6050 (I2C interface, address 0x68)
  - Servos: 8x Feetech STS/SCS serial bus servos (UART /dev/ttyS1 or /dev/ttyUSB0 at 1 Mbps)
  - Policy: Loads 'policy_stabilize.npz' or 'policy_walk.npz'

Run on Orange Pi:
  python3 deploy_feetech_orangepi.py --mode walk --port /dev/ttyS1
"""

import os
import sys
import time
import math
import argparse
import struct
import numpy as np

try:
    from train_walk_and_stabilize import BipedalPhysicsModel, ParameterTrial
except ImportError:
    BipedalPhysicsModel = None
    ParameterTrial = None

# ==============================================================================
# 1. PURE PYTHON MPU6050 DRIVER (I2C)
# ==============================================================================
class MPU6050:
    """Lightweight driver for MPU6050 IMU on Orange Pi."""
    def __init__(self, bus_num=0, address=0x68):
        self.address = address
        self.bus_num = bus_num
        self.bus = None
        try:
            import smbus2
            self.bus = smbus2.SMBus(bus_num)
            # Wake up MPU-6050 (default sleep mode on register 0x6B)
            self.bus.write_byte_data(self.address, 0x6B, 0x00)
            print(f"[OK] MPU6050 initialized on /dev/i2c-{bus_num} (addr: 0x{address:02x})")
        except Exception as e:
            print(f"[WARN] Could not connect to MPU6050 on I2C-{bus_num}: {e}")
            print("       Running in mock IMU mode for testing.")

    def read_raw_data(self, addr):
        if self.bus is None:
            return 0.0, 0.0, 0.0
        high = self.bus.read_byte_data(self.address, addr)
        low = self.bus.read_byte_data(self.address, addr + 1)
        val = (high << 8) | low
        if val > 32768:
            val = val - 65536
        return val

    def get_motion(self):
        """Returns: accel (m/s^2), gyro (rad/s), projected gravity [gx, gy, gz]."""
        if self.bus is None:
            # Mock upright gravity when hardware is not plugged in
            return np.array([0.0, 0.0, 9.81]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])

        # Accel sensitivity: 16384 LSB/g
        ax = self.read_raw_data(0x3B) / 16384.0 * 9.81
        ay = self.read_raw_data(0x3D) / 16384.0 * 9.81
        az = self.read_raw_data(0x3F) / 16384.0 * 9.81

        # Gyro sensitivity: 131 LSB/(deg/s) -> rad/s
        gx = math.radians(self.read_raw_data(0x43) / 131.0)
        gy = math.radians(self.read_raw_data(0x45) / 131.0)
        gz = math.radians(self.read_raw_data(0x47) / 131.0)

        # Normalize gravity vector
        norm = math.sqrt(ax * ax + ay * ay + az * az) + 1e-6
        proj_gravity = np.array([ax / norm, ay / norm, az / norm])

        return np.array([ax, ay, az]), np.array([gx, gy, gz]), proj_gravity


# ==============================================================================
# 2. FEETECH SERIAL BUS SERVO CONTROLLER (UART)
# ==============================================================================
class FeetechServoBus:
    """Driver for Feetech STS/SCS Serial Bus Servo Motors."""
    def __init__(self, port="/dev/ttyS1", baudrate=1000000):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.servo_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 10 joints (Legs, Tail, Neck, Head Connector, Revolute Neck)
        self.center_ticks = [2048] * 8
        self.ticks_per_rad = 4096.0 / (2.0 * math.pi)

        try:
            import serial
            self.ser = serial.Serial(port, baudrate, timeout=0.05)
            print(f"[OK] Feetech Servo Bus opened on {port} @ {baudrate} baud")
        except Exception as e:
            print(f"[WARN] Serial port {port} not available: {e}")
            print("       Running in mock servo mode.")

    def write_position(self, servo_id, position_ticks, speed=1500, acc=50):
        """Feetech STS series Write Position packet."""
        if self.ser is None:
            return
        pos = max(0, min(4095, int(position_ticks)))
        # Packet: 0xFF 0xFF ID LENGTH INST REG_ADDR POS_L POS_H TIME_L TIME_H SPEED_L SPEED_H CHECKSUM
        length = 7
        inst = 0x03  # WRITE
        reg = 0x2A   # Target Position address on STS servos
        p1, p2 = pos & 0xFF, (pos >> 8) & 0xFF
        s1, s2 = speed & 0xFF, (speed >> 8) & 0xFF
        checksum = (~(servo_id + length + inst + reg + p1 + p2 + s1 + s2)) & 0xFF
        packet = bytearray([0xFF, 0xFF, servo_id, length, inst, reg, p1, p2, s1, s2, checksum])
        self.ser.write(packet)

    def set_joint_angles(self, angles_rad):
        """Sends target angles (radians) to all 8 Feetech servos."""
        for i, servo_id in enumerate(self.servo_ids):
            ticks = self.center_ticks[i] + int(angles_rad[i] * self.ticks_per_rad)
            self.write_position(servo_id, ticks)

    def disable_torque(self):
        """Emergency torque release to protect motors when fallen."""
        if self.ser is None:
            return
        for servo_id in self.servo_ids:
            # Write 0 to Torque Enable (Reg 0x28)
            packet = bytearray([0xFF, 0xFF, servo_id, 4, 0x03, 0x28, 0x00, (~(servo_id + 4 + 3 + 0x28)) & 0xFF])
            self.ser.write(packet)
        print("[SAFETY] All Feetech servo torques disabled.")


# ==============================================================================
# 3. REAL-WORLD POLICY RUNNER
# ==============================================================================
class BipedRunner:
    def __init__(self, policy_path, port="/dev/ttyS1", i2c_bus=0, target_speed=0.35):
        self.imu = MPU6050(bus_num=i2c_bus)
        self.servos = FeetechServoBus(port=port)
        self.target_speed = target_speed

        # Load weights
        if not os.path.exists(policy_path):
            raise FileNotFoundError(f"Policy file '{policy_path}' not found! Train it first.")
        data = np.load(policy_path)
        self.w1 = data['w1']
        self.b1 = data['b1']
        self.w2 = data['w2']
        self.b2 = data['b2']
        print(f"[OK] Loaded neural network policy from {policy_path}")

        # Check for evolved physical champion gait parameters
        self.freq = float(data['freq']) if 'freq' in data else 2.0
        self.target_speed = float(data['target_vel']) if 'target_vel' in data else target_speed
        self.tail_bias = float(data['tail_bias']) if 'tail_bias' in data else 0.0
        self.neck_bias = float(data['neck_bias']) if 'neck_bias' in data else 0.0
        self.knee_scale = float(data['knee_scale']) if 'knee_scale' in data else 0.35
        self.hip_scale = float(data['hip_scale']) if 'hip_scale' in data else 0.40
        self.roll_scale = float(data['roll_scale']) if 'roll_scale' in data else 0.06

        # Action scale matching MuJoCo training
        self.action_scale = np.array([
            0.18, 0.22, 0.25,  # Right leg [roll, pitch, knee]
            0.18, 0.22, 0.25,  # Left leg  [roll, pitch, knee]
            0.18,              # Tail
            0.15,              # Lower neck pitch boom
            0.15,              # Upper headconnector pitch
            0.22               # Revolute neck roll/yaw tilt
        ], dtype=np.float32)
        self.q_nominal = np.zeros(len(self.action_scale))
        self.q_nominal[6] = self.tail_bias
        self.q_nominal[7] = self.neck_bias
        self.last_action = np.zeros(len(self.action_scale))
        self.time_sec = 0.0
        
        # Physics Control Model
        if BipedalPhysicsModel is not None:
            self.physics_model = BipedalPhysicsModel(num_robots=1)
            self.trial = ParameterTrial(0, 1)
            self.trial.freq = self.freq
            self.trial.target_vel = self.target_speed
            self.trial.tail_bias = self.tail_bias
            self.trial.neck_bias = self.neck_bias
            self.trial.knee_scale = self.knee_scale
            self.trial.hip_scale = self.hip_scale
            self.trial.roll_scale = self.roll_scale
        else:
            self.physics_model = None
            self.trial = None
        print(f"[OK] Champion Gait Parameters: Cadence={self.freq:.2f}Hz, Speed={self.target_speed:.2f}m/s, TailBias={self.tail_bias:+.2f}rad")

    def predict(self, obs):
        h = np.tanh(np.dot(obs, self.w1) + self.b1)
        return np.tanh(np.dot(h, self.w2) + self.b2)

    def run_loop(self, loop_rate_hz=50.0):
        dt = 1.0 / loop_rate_hz
        print(f"\n[START] Starting real-time control loop @ {loop_rate_hz} Hz on Orange Pi...")
        print("        Press Ctrl+C to stop.\n")

        try:
            while True:
                start_time = time.time()
                self.time_sec += dt

                # 1. Read MPU-6050
                accel, gyro, proj_gravity = self.imu.get_motion()

                # Safety check: if robot has fallen over (tilt > 50 degrees), release torque!
                if proj_gravity[2] < 0.65:
                    print(f"[ALERT] Robot fell over (gz={proj_gravity[2]:.2f})! Disabling torque.")
                    self.servos.disable_torque()
                    time.sleep(1.0)
                    continue

                # 2. Assemble observation matching training state
                # [height_est, proj_gravity, linvel, angvel, joint_pos, joint_vel, cmd_vel, sin_phi, cos_phi]
                height_est = np.array([0.28])
                linvel_est = np.array([0.0, 0.0, 0.0])
                angvel = gyro
                joint_pos = self.last_action * self.action_scale
                joint_vel = np.zeros(len(self.action_scale))
                cmd_vel = np.array([self.target_speed])
                
                # Orientation angles from projected gravity
                pitch = math.atan2(-proj_gravity[1], math.sqrt(proj_gravity[0]**2 + proj_gravity[2]**2))
                roll = math.atan2(proj_gravity[0], proj_gravity[2])

                # Adaptive CPG limit-cycle clock
                if self.physics_model is not None:
                    phi_r = self.physics_model.phi_r[0]
                else:
                    phi_r = 2.0 * math.pi * self.freq * self.time_sec
                cpg_clock = np.array([math.sin(phi_r), math.cos(phi_r)])

                obs = np.concatenate([
                    height_est,
                    proj_gravity,
                    linvel_est,
                    angvel,
                    joint_pos,
                    joint_vel,
                    cmd_vel,
                    cpg_clock,
                    np.array([pitch, roll], dtype=np.float32)
                ]).astype(np.float32)

                # Match input dimensions (31-dim)
                if len(obs) < self.w1.shape[0]:
                    obs = np.pad(obs, (0, self.w1.shape[0] - len(obs)))
                else:
                    obs = obs[:self.w1.shape[0]]

                # 3. Policy forward pass (< 0.2 ms on Orange Pi Cortex-A53)
                action = self.predict(obs)

                # Low-pass filter to smooth motor chatter
                action_filtered = 0.7 * self.last_action + 0.3 * action
                self.last_action = action_filtered

                # 4. Physics-Based Control Reference (Adaptive CPG, 2D Polar LIPM, Coupled Hips, Dynamic Level)
                if self.physics_model is not None:
                    q_ref = self.physics_model.compute_joint_references(
                        0, self.trial, dt, pitch, roll, angvel[0], angvel[1], linvel_est[0], linvel_est[1], 0.0, 0.0, height_est[0]
                    )
                else:
                    phi_r = 2.0 * math.pi * self.freq * self.time_sec
                    phi_l = phi_r + math.pi
                    r_sw = math.sin(phi_r)
                    l_sw = math.sin(phi_l)
                    hip_bias = -0.25
                    sway = -self.roll_scale * r_sw
                    q_ref = np.zeros(8, dtype=np.float32)
                    q_ref[0] = sway
                    q_ref[1] = hip_bias - self.hip_scale * r_sw
                    q_ref[2] = self.knee_scale * max(0.0, r_sw)
                    q_ref[3] = sway
                    q_ref[4] = hip_bias - self.hip_scale * l_sw
                    q_ref[5] = -self.knee_scale * max(0.0, l_sw)
                    q_ref[6] = self.tail_bias
                    q_ref[7] = self.neck_bias
                
                residual = action_filtered * self.action_scale
                target_angles = q_ref + residual
                self.servos.set_joint_angles(target_angles)

                # Sleep to maintain stable 50 Hz loop
                elapsed = time.time() - start_time
                if elapsed < dt:
                    time.sleep(dt - elapsed)

        except KeyboardInterrupt:
            print("\n[STOP] Controller stopped by user.")
        finally:
            self.servos.disable_torque()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orange Pi Zero 3W Bipedal Hardware Deployment")
    parser.add_argument("--mode", choices=["stabilize", "walk"], default="walk", help="Select control mode")
    parser.add_argument("--port", type=str, default="/dev/ttyS1", help="Serial port for Feetech servos")
    parser.add_argument("--i2c", type=int, default=0, help="I2C bus number for MPU6050")
    parser.add_argument("--speed", type=float, default=0.35, help="Walking speed in m/s")
    args = parser.parse_args()

    policy_file = "policy_walk.npz" if args.mode == "walk" else "policy_stabilize.npz"
    runner = BipedRunner(policy_file, port=args.port, i2c_bus=args.i2c, target_speed=args.speed)
    runner.run_loop(loop_rate_hz=50.0)
