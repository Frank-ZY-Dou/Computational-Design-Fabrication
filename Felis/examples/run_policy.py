"""The smallest use of a policy: no robot, no simulator, one 50 Hz step of the walking policy on made-up sensor readings.
Also checks that the TorchScript file gives the same action.

Run from the Felis folder:  python examples/run_policy.py
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from felis_policy import FelisPolicy  # noqa: E402

POLICY = os.path.join(ROOT, "policies", "walking")

pol = FelisPolicy(os.path.join(POLICY, "actor.pt"))
pol.reset()                              # zero action history, gait clock back to phase 0
pol.set_command(0.2, 0.0, 0.0)           # vx (m/s), vy (m/s), wz (rad/s)

# sensor readings (conventions in README.md); here the robot stands still
base_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])   # w, x, y, z of the rear body in the world frame
base_ang_vel = torch.zeros(3)                    # gyro, body frame, rad/s
dof_pos = torch.zeros(6)                         # waist_yaw, waist_pitch, hip_FR, hip_FL, hip_RL, hip_RR, rad
dof_vel = torch.zeros(6)                         # same order, rad/s

target, raw = pol.compute_action(base_quat, base_ang_vel, dof_pos, dof_vel)
print("observation:", pol.last_obs.numpy().round(3))
print("joint position targets (rad):", target.numpy().round(3))

jit = torch.jit.load(os.path.join(POLICY, "actor_jit.pt"))
with torch.no_grad():
    raw_jit = jit(pol.last_obs.unsqueeze(0)).squeeze(0).clamp(-pol.clip_actions, pol.clip_actions)
assert (raw_jit - raw).abs().max() < 1e-5
print("TorchScript gives the same action")
