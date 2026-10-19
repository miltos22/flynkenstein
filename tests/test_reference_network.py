import math
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flybrain import params as P
from flybrain import reference as R

def toy(rec, W=1.0, delay=3):
    return {'cls': np.array([2, 2], dtype=np.int8), 'row_ptr': np.array([0, 0, 1]), 'col_idx': np.array([0], dtype=np.int32), 'rec': np.array([rec], dtype=np.uint8), 'W_struct': np.array([W], dtype=np.float32), 'delay': np.array([delay], dtype=np.uint8)}

def run(rec, steps=12, W=2.0, delay=3):
    net = R.Network(toy(rec, W, delay), device='cpu', dtype=torch.float64, event=False)
    I = torch.zeros(2, dtype=torch.float64)
    hold = net.s['c']['E_L_base_mV'].clone()
    out = []
    for n in range(steps):
        net.s['V'] = hold.clone()
        net.s['spikes'] = torch.tensor([1.0 if n == 0 else 0.0, 0.0], dtype=torch.float64)
        t = net.step(I)
        out.append(({k: float(net.s[k][1]) for k in ('g_exc_f', 'g_exc_s', 'g_inh_f', 'g_inh_s')}, float(t[0])))
    return out

def test_exc_spike_arrives_after_delay_and_decays():
    out = run(rec=0)
    rel = out[0][1]
    assert abs(rel - P.CLASS['U0'][2]) < 1e-12
    for n in range(3):
        assert out[n][0]['g_exc_f'] == 0.0 and out[n][0]['g_exc_s'] == 0.0
    b = P.beta_rec[0]
    assert abs(out[3][0]['g_exc_f'] - b[0] * 2.0 * rel) < 1e-12
    assert abs(out[3][0]['g_exc_s'] - b[1] * 2.0 * rel) < 1e-12
    assert out[3][0]['g_inh_f'] == 0.0 and out[3][0]['g_inh_s'] == 0.0
    for n in range(4, 12):
        k = n - 3
        assert abs(out[n][0]['g_exc_f'] - b[0] * 2.0 * rel * math.exp(-k * P.dt_ms / P.tau_syn_excf_ms)) < 1e-12
        assert abs(out[n][0]['g_exc_s'] - b[1] * 2.0 * rel * math.exp(-k * P.dt_ms / P.tau_syn_excs_ms)) < 1e-12

def test_inh_codes_route_to_inhibitory_channels():
    for rec in (2, 3):
        out = run(rec=rec)
        b = P.beta_rec[rec]
        rel = out[0][1]
        assert abs(out[3][0]['g_inh_f'] - b[2] * 2.0 * rel) < 1e-12
        assert abs(out[3][0]['g_inh_s'] - b[3] * 2.0 * rel) < 1e-12
        assert out[3][0]['g_exc_f'] == 0.0 and out[3][0]['g_exc_s'] == 0.0

def random_conn(n=400, m=6000, seed=1):
    rng = np.random.default_rng(seed)
    cls = rng.integers(0, 6, n).astype(np.int8)
    post = np.sort(rng.integers(0, n, m))
    pre = rng.integers(0, n, m)
    keep = pre != post
    post, pre = (post[keep], pre[keep])
    row_ptr = np.concatenate([[0], np.cumsum(np.bincount(post, minlength=n))])
    return {'cls': cls, 'row_ptr': row_ptr, 'col_idx': pre.astype(np.int32), 'rec': rng.integers(0, 4, pre.size).astype(np.uint8), 'W_struct': rng.uniform(0.0, 3.0, pre.size).astype(np.float32), 'delay': rng.integers(3, 64, pre.size).astype(np.uint8)}

def test_event_path_matches_dense():
    conn = random_conn()
    nets = [R.Network(conn, device='cpu', dtype=torch.float64, event=e) for e in (False, True)]
    g = torch.Generator().manual_seed(2)
    V0 = nets[0].s['c']['E_L_base_mV'] + 2.5 * torch.randn(nets[0].N, generator=g, dtype=torch.float64)
    for net in nets:
        net.s['V'] = V0.clone()
    spikes = 0
    for n in range(3000):
        I = 30.0 * torch.rand(nets[0].N, generator=g, dtype=torch.float64)
        ge, gi = (0.5, 0.8) if n < 500 else (1.0, 1.0)
        for net in nets:
            net.step(I, ge, gi)
        assert torch.equal(nets[0].s['spikes'], nets[1].s['spikes']), f'spike trains differ at step {n}'
        spikes += int(nets[0].s['spikes'].sum())
        if n == 3:
            assert float((nets[0].s['V'] - nets[1].s['V']).abs().max()) < 1e-10
    assert spikes > 1000, f'too few spikes to test transmission ({spikes})'
    d = float((nets[0].s['V'] - nets[1].s['V']).abs().max())
    assert d < 0.01, f'V differs by {d} mV'

def test_gap_junction_currents_match_spec_formula():
    conn = toy(0)
    conn.update({'gap_pre': np.array([0, 1]), 'gap_post': np.array([1, 0]), 'gap_g_nS': np.array([0.2, 0.2]), 'gap_alpha': np.array([1.0, 0.3])})
    net = R.Network(conn, device='cpu', dtype=torch.float64, event=False, compile=False)
    net.s['V'] = torch.tensor([-50.0, -60.0], dtype=torch.float64)
    net._gap_update()
    g01 = 0.2
    g10 = 0.2 * (0.3 + 0.7 / (1 + np.exp(-P.beta_rect_per_mV * (-50.0 - -60.0))))
    assert abs(float(net.s['G_gap_sum'][1]) - g01) < 1e-12 and abs(float(net.s['I_gap_num'][1]) - g01 * -50.0) < 1e-12
    assert abs(float(net.s['G_gap_sum'][0]) - g10) < 1e-12 and abs(float(net.s['I_gap_num'][0]) - g10 * -60.0) < 1e-12
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
