"""Felis policy runtime (50 Hz, CPU, PyTorch only).

Loads an exported actor (policies/<skill>/actor.pt or actor_jit.pt, with the policy_meta.json next to it) and turns sensor
readings into joint position targets, reproducing the observation pipeline the policy was trained with to float32
tolerance (observation < 1e-4, action < 5e-3).

One runtime serves every skill (policy_meta.json["task"]):
  righting  28 inputs: the 27 common ones + (drop_height - 2.5) / 0.5; call set_drop_height() before the release
  loco      29 inputs: the 27 common ones + the gait clock (sin, cos); walking takes the velocity command from
            set_command(vx, vy, wz), the jump takes its jump command the same way (see README.md)

Sensor contract (frames and signs in README.md):
  base_quat     [w, x, y, z]  rear body orientation in the world frame
  base_ang_vel  [x, y, z]     rad/s in the body frame (gyro)
  dof_pos/vel   [6]           output shaft, rad and rad/s, order
                              [waist_yaw, waist_pitch, hip_FR, hip_FL, hip_RL, hip_RR],
                              zero = assembly pose (legs straight down)

Self-test (no robot needed):  python felis_policy.py --selftest
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(HERE, "policies", "righting", "actor.pt")


def build_actor(num_obs, num_actions, hidden_dims):
    layers, last = [], num_obs
    for h in hidden_dims:
        layers += [nn.Linear(last, h), nn.ELU()]
        last = h
    layers.append(nn.Linear(last, num_actions))
    return nn.Sequential(*layers)


def quat_mul_z(q, angle):
    """q * (rotation by angle about z), wxyz."""
    c, s = math.cos(float(angle) / 2.0), math.sin(float(angle) / 2.0)
    w, x, y, z = q[0], q[1], q[2], q[3]
    return torch.stack([w * c - z * s, x * c + y * s, y * c - x * s, z * c + w * s])


def quat_rotate_inverse(q, v):
    """Rotate world vector v into the body frame. q is wxyz."""
    w, xyz = q[0], q[1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v - w * t + torch.cross(xyz, t, dim=-1)


class FelisPolicy:
    def __init__(self, weights_path=DEFAULT_WEIGHTS, device="cpu", clamp_targets=None):
        # batch-1 MLP inference: one thread is faster than a thread pool at this size
        if device == "cpu":
            torch.set_num_threads(1)
        self.device = device
        meta_path = os.path.join(os.path.dirname(weights_path), "policy_meta.json")
        with open(meta_path) as f:
            self.meta = json.load(f)
        net = self.meta["network"]
        self.num_obs, self.num_actions = net["num_obs"], net["num_actions"]

        if weights_path.endswith("_jit.pt"):
            self.actor = torch.jit.load(weights_path, map_location=device)
        else:
            self.actor = build_actor(self.num_obs, self.num_actions, net["actor_hidden_dims"]).to(device)
            sd = torch.load(weights_path, map_location=device)
            self.actor.load_state_dict({k[len("actor."):]: v for k, v in sd.items() if k.startswith("actor.")}, strict=True)
        self.actor.eval()
        print(f"loaded {weights_path} ({self.meta.get('policy', '')})")

        t = lambda x: torch.tensor(x, dtype=torch.float32, device=device)
        scales = self.meta["obs_scales"]
        self.ang_vel_scale = scales["ang_vel"]
        self.dof_pos_scale = scales["dof_pos"]
        self.dof_vel_scale = scales["dof_vel"]
        self.action_scale = self.meta["action_scale"]
        self.clip_actions = self.meta["clip_actions"]
        self.default_dof_pos = t(self.meta["default_dof_pos"])
        self.joint_names = self.meta["joint_names"]
        self.task = self.meta.get("task", "righting")
        self.commands = torch.zeros(3, device=device)  # righting: always zero; loco: set_command()
        self.command_scale = t(self.meta["command_scale"])
        # height-conditioned righting sees the known release height
        self.obs_drop_height = bool(self.meta.get("obs_drop_height", False))
        dh = self.meta.get("drop_height_obs", {"center": 2.5, "scale": 0.5})
        self.drop_h_center, self.drop_h_scale = float(dh["center"]), float(dh["scale"])
        self.drop_height = torch.tensor([self.drop_h_center], dtype=torch.float32, device=device)
        # release clock (policies with obs_time): seconds since reset(), i.e. since the release
        self.obs_time = bool(self.meta.get("obs_time", False))
        tobs = self.meta.get("time_obs", {"center": 0.75, "scale": 0.75, "max_s": 1.5})
        self.t_center, self.t_scale, self.t_max = float(tobs["center"]), float(tobs["scale"]), float(tobs["max_s"])
        self.control_dt = 1.0 / float(self.meta.get("control", {}).get("rate_hz", 50.0))
        self.steps = 0
        # gait clock: phase = frac(freq * t), t = seconds since reset()
        gc = self.meta.get("gait_clock")
        self.gait_freq = float(gc["freq"]) if gc else None
        self.frame_obs = 27 + int(self.obs_drop_height) + int(self.obs_time) + 2 * int(gc is not None)
        self.obs_hist, self.hist = int(self.meta.get("obs_hist", 1)), None
        expect = self.frame_obs * self.obs_hist
        assert self.num_obs == expect, f"policy expects {self.num_obs} obs, layout gives {expect}"
        self.global_gravity = t([0.0, 0.0, -1.0])
        # the IMU sits on the middle body segment; motor velocity feedback and the gyro saturate
        self.imu_cross = self.meta.get("imu_link") == "cross_body"
        self.dof_vel_clip = self.meta.get("dof_vel_obs_clip")
        self.ang_vel_clip = self.meta.get("ang_vel_obs_clip")
        # targets are kept within the joint stops, as during training
        per = self.meta.get("joint_limits_per_joint")
        if per:
            self.target_lo = t([per[n][0] for n in self.joint_names])
            self.target_hi = t([per[n][1] for n in self.joint_names])
        else:
            lim = self.meta["joint_limits_rad"]
            self.target_lo = t([lim["waist_yaw"][0], lim["waist_pitch"][0]] + [lim["hips"][0]] * 4)
            self.target_hi = t([lim["waist_yaw"][1], lim["waist_pitch"][1]] + [lim["hips"][1]] * 4)
        if clamp_targets is None:
            clamp_targets = bool(self.meta.get("control", {}).get("clamp_target_to_limit", False))
        self.clamp_targets = clamp_targets

        self.last_actions = torch.zeros(self.num_actions, device=device)
        self.last_obs = None  # kept for parity tests and logging

    def reset(self):
        self.last_actions.zero_()
        self.steps = 0
        self.hist = None

    def set_command(self, vx, vy=0.0, wz=0.0):
        """Body-frame velocity command (m/s, m/s, rad/s) for walking, or the jump command."""
        self.commands[0], self.commands[1], self.commands[2] = float(vx), float(vy), float(wz)

    def set_drop_height(self, h):
        """Release height (m) for the height-conditioned righting policy."""
        self.drop_height[0] = float(h)

    def compute_observation(self, base_quat, base_ang_vel, dof_pos, dof_vel, waist=None):
        """base_*: the rear body's attitude and rate. With the IMU on the middle segment they are turned into its frame by the
        waist yaw and its rate (waist = (q, qd), default the measured ones); compute_observation_imu takes the IMU's directly."""
        if self.imu_cross:
            q, qd = (dof_pos[0], dof_vel[0]) if waist is None else waist
            base_quat = quat_mul_z(base_quat, q)
            c, s = math.cos(float(q)), math.sin(float(q))
            w = base_ang_vel
            base_ang_vel = torch.stack([c * w[0] + s * w[1], -s * w[0] + c * w[1], w[2] + float(qd)])
        return self.compute_observation_imu(base_quat, base_ang_vel, dof_pos, dof_vel)

    def compute_observation_imu(self, base_quat, base_ang_vel, dof_pos, dof_vel):
        """The observation from what the robot reads: attitude (wxyz) and body rate of the IMU's link, joints in policy order."""
        # a gyro-integrated attitude drifts off unit norm, which would scale the projected gravity
        base_quat = base_quat / base_quat.norm().clamp(min=1e-8)
        projected_gravity = quat_rotate_inverse(base_quat, self.global_gravity)
        if self.dof_vel_clip:
            dof_vel = dof_vel.clamp(-self.dof_vel_clip, self.dof_vel_clip)
        if self.ang_vel_clip:
            base_ang_vel = base_ang_vel.clamp(-self.ang_vel_clip, self.ang_vel_clip)
        parts = [
            base_ang_vel * self.ang_vel_scale,                       # 3
            projected_gravity,                                       # 3
            self.commands * self.command_scale,                      # 3 (zeros for righting)
            (dof_pos - self.default_dof_pos) * self.dof_pos_scale,   # 6
            dof_vel * self.dof_vel_scale,                            # 6
            self.last_actions,                                       # 6
        ]
        if self.obs_drop_height:
            parts.append((self.drop_height - self.drop_h_center) / self.drop_h_scale)  # 1
        if self.obs_time:
            t = min(self.steps * self.control_dt, self.t_max)
            parts.append(torch.tensor([(t - self.t_center) / self.t_scale], device=self.drop_height.device))  # 1
        if self.gait_freq is not None:
            a = 2.0 * math.pi * ((self.steps * self.control_dt * self.gait_freq) % 1.0)
            parts.append(torch.tensor([math.sin(a), math.cos(a)], device=self.drop_height.device))  # 2
        return torch.cat(parts)

    def compute_action(self, base_quat, base_ang_vel, dof_pos, dof_vel):
        """All inputs float32 tensors per the sensor contract above.
        Returns (target_dof_pos[6] for the joint position loop, raw_action[6])."""
        return self.act(self.compute_observation(base_quat, base_ang_vel, dof_pos, dof_vel))

    def act(self, obs):
        """One policy step from a full observation vector."""
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        self.last_obs = obs
        x = obs
        if self.obs_hist > 1:
            self.hist = x = obs.repeat(self.obs_hist) if self.hist is None else torch.cat([self.hist[self.frame_obs:], obs])
        with torch.no_grad():
            raw = self.actor(x.unsqueeze(0)).squeeze(0)
        raw = torch.clip(raw, -self.clip_actions, self.clip_actions)
        self.last_actions = raw  # the next observation carries the clipped action
        self.steps += 1
        target = self.action_scale * raw + self.default_dof_pos
        if self.clamp_targets:
            target = torch.clamp(target, self.target_lo, self.target_hi)
        return target, raw

    # ------------------------------------------------------------- robot loop
    def run_loop(self, read_sensors=None, send_targets=None, release_detected=None, rate_hz=50.0, hold_s=3.0):
        """read_sensors() -> (base_quat, base_ang_vel, dof_pos, dof_vel) tensors;
        send_targets(target_dof_pos) -> None. Both must be wired to the robot.

        release_detected(sensors) -> bool starts the policy at the actual release (e.g. |accelerometer| ~ 0 in free fall).
        A righting episode starts at the release with a zero action history, so the policy must not run while the robot
        is held; without a trigger (None) it starts right after the hold phase (bench tests only)."""
        if read_sensors is None or send_targets is None:
            raise NotImplementedError("wire read_sensors / send_targets to the motor bus and the IMU first")
        dt = 1.0 / rate_hz
        try:
            print(f"holding the default pose for {hold_s:.0f} s ...")
            next_t = time.monotonic()
            for _ in range(int(hold_s * rate_hz)):
                read_sensors()
                send_targets(self.default_dof_pos)
                next_t += dt
                time.sleep(max(0.0, next_t - time.monotonic()))
            if release_detected is not None:
                print("armed, waiting for the release ...")
                while not release_detected(read_sensors()):
                    send_targets(self.default_dof_pos)
                    next_t += dt
                    time.sleep(max(0.0, next_t - time.monotonic()))
            print("policy engaged")
            self.reset()
            next_t = time.monotonic()
            while True:
                target, _ = self.compute_action(*read_sensors())
                send_targets(target)
                next_t += dt
                rem = next_t - time.monotonic()
                if rem < -dt:  # fell behind a full period: log and re-anchor
                    print(f"loop overrun {-rem * 1e3:.1f} ms")
                    next_t = time.monotonic()
                else:
                    time.sleep(max(0.0, rem))
        except KeyboardInterrupt:
            print("\nstopped")
        except Exception as e:  # a sensor or bus failure must not leave the last target applied
            print(f"\nABORT: {type(e).__name__}: {e}")
        finally:
            try:
                send_targets(self.default_dof_pos)
                print("holding the default pose")
            except Exception as e:
                print(f"failed to send the default pose: {e}")


def selftest(weights_path):
    pol = FelisPolicy(weights_path)
    w = torch.zeros(3)
    qd = torch.zeros(6)
    # attitude checks with non-degenerate quaternions (belly-up alone cannot tell wxyz from xyzw)
    s2 = math.sqrt(0.5)
    cases = [  # (name, quat wxyz, expected projected gravity in obs[3:6])
        ("upright", (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
        ("belly-up", (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, +1.0)),
        ("roll +90 deg", (s2, s2, 0.0, 0.0), (0.0, -1.0, 0.0)),
        ("pitch nose-down", (s2, 0.0, s2, 0.0), (+1.0, 0.0, 0.0)),
    ]
    for name, q, expect in cases:
        pol.reset()
        pol.compute_action(torch.tensor(q), w, pol.default_dof_pos.clone(), qd)
        pg = pol.last_obs[3:6]
        err = (pg - torch.tensor(expect)).abs().max().item()
        assert err < 1e-5, f"{name}: projected gravity {pg.numpy()} != {expect}"
        print(f"projected gravity [{name:>15}] = {pg.numpy().round(3)}  OK")

    pol.reset()
    q = torch.tensor([0.0, 1.0, 0.0, 0.0])  # belly-up
    target, raw = pol.compute_action(q, w, pol.default_dof_pos.clone(), qd)
    print(f"belly-up, zero state: raw action {raw.numpy().round(3)}")
    print(f"                      target {target.numpy().round(3)} rad")

    pol.reset()
    n, times = 2000, []
    for _ in range(n):
        t0 = time.perf_counter()
        pol.compute_action(q, w, pol.default_dof_pos.clone(), qd)
        times.append(time.perf_counter() - t0)
    times = torch.tensor(times[100:]) * 1e3
    p99 = times.kthvalue(int(0.99 * len(times))).values.item()
    print(f"inference time over {len(times)} calls: mean {times.mean():.3f} ms, p99 {p99:.3f} ms, max {times.max():.3f} ms "
          f"(budget 20 ms at 50 Hz) -> {'PASS' if times.max() < 20 else 'FAIL'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest(args.weights)
    else:
        FelisPolicy(args.weights).run_loop()  # raises until the sensors are wired


if __name__ == "__main__":
    main()
