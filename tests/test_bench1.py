import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flybrain import bench1, reference as R
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_reference_network import random_conn

def synthetic_brain(n=400):
    conn = random_conn(n=n)
    rng = np.random.default_rng(3)
    for name in ('dFB', 'ER5', 'motor', 'clock', 'PAM', 'PPL1', 'DAN', 'IPC', 'SMP'):
        conn['set_' + name] = np.sort(rng.choice(n, 10, replace=False)).astype(np.int32)
    conn['n_kc'] = 0
    conn['grid_shape'] = np.array([4, 4, 8])
    conn['grid_spacing_um'] = np.array([20.0, 20.0, 20.0])
    conn['nt'] = rng.integers(0, 7, n).astype(np.int8)
    conn['voxel'] = rng.integers(0, 128, n).astype(np.int32)
    return conn

def test_snapshot_restore_replays_identically():
    sim = R.Sim(synthetic_brain(), device='cpu', dtype=torch.float64)
    I = torch.full((sim.net.N,), 20.0, dtype=torch.float64)
    for _ in range(300):
        sim.step(I)
    snap = bench1.snapshot(sim)
    runs = []
    for _ in range(2):
        bench1.restore(sim, snap)
        trace = []
        for _ in range(400):
            sim.step(I)
            trace.append(sim.net.s['V'].clone())
        runs.append(torch.stack(trace))
    assert int(sim.net.s['spikes'].numel()) > 0
    assert torch.equal(runs[0], runs[1]), 'restore did not replay exactly'

def test_mr_estimator_recovers_branching_ratio():
    rng = np.random.default_rng(0)
    m, h, a = (0.9, 20.0, [200.0])
    for _ in range(20000):
        a.append(rng.poisson(m * a[-1] + h))
    est = bench1.mr_branching(a)
    assert abs(est - m) < 0.02, est
if __name__ == '__main__':
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            try:
                fn()
                print('PASS', name)
            except AssertionError as e:
                fails += 1
                print('FAIL', name, '-', e)
    sys.exit(1 if fails else 0)
