"""Replay the reference data in parity/ through felis_policy.py: from each step's raw sensor readings to the observation, the raw
action and the joint target. A port of the policies to another runtime should give the same three to 1e-4.

Run from the Felis folder:  python examples/check_parity.py
"""
import contextlib
import io
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from felis_policy import FelisPolicy  # noqa: E402

f32 = lambda x: torch.from_numpy(np.asarray(x, dtype=np.float32))
worst = 0.0
for skill in ("walking", "righting", "jump"):
    d = np.load(os.path.join(ROOT, "parity", skill + ".npz"))
    data, names = d["data"], [str(x) for x in d["names"]]
    col = lambda name, n=1: data[:, names.index(name):names.index(name) + n]
    with contextlib.redirect_stdout(io.StringIO()):
        pol = FelisPolicy(os.path.join(ROOT, "policies", skill, "actor.pt"))
    n = pol.num_obs
    errors = np.zeros(3)
    for i in range(len(data)):
        if i == 0 or (skill == "righting" and col("step")[i, 0] == 0):
            pol.reset()                                    # zero action history and clock
            if skill == "righting":
                pol.set_drop_height(float(col("drop_h")[i, 0]))
        if skill != "righting":
            pol.set_command(*col("cmd_vx", 3)[i].tolist())
        obs = pol.compute_observation_imu(f32(col("imu_q0", 4)[i]), f32(col("imu_w0", 3)[i]), f32(col("q0", 6)[i]), f32(col("qd0", 6)[i]))
        target, raw = pol.act(obs)
        for k, (got, want) in enumerate(((obs, col("obs0", n)[i]), (raw, col("act0", 6)[i]), (target, col("target0", 6)[i]))):
            errors[k] = max(errors[k], float(np.abs(got.numpy() - want).max()))
    print(f"{skill}: {len(data)} steps, max error obs {errors[0]:.2e}, raw action {errors[1]:.2e}, target {errors[2]:.2e}")
    worst = max(worst, float(errors.max()))
assert worst < 1e-4, worst
print("parity OK")
