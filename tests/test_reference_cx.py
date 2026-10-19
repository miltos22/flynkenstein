import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flybrain import params as P
from flybrain import reference as R

def toy():
    cls = np.array([2, 1, 2, 2, 2, 4, 5], dtype=np.int8)
    posts = [1, 3, 5]
    pre = {1: 0, 3: 2, 5: 4}
    rec = {1: 2, 3: 0, 5: 0}
    row_ptr, col, rc = ([0], [], [])
    for i in range(7):
        if i in pre:
            col.append(pre[i])
            rc.append(rec[i])
        row_ptr.append(len(col))
    conn = {'cls': cls, 'row_ptr': np.array(row_ptr), 'col_idx': np.array(col, dtype=np.int32), 'rec': np.array(rc, dtype=np.uint8), 'W_struct': np.ones(3, dtype=np.float32), 'delay': np.full(3, 3, dtype=np.uint8)}
    net = R.Network(conn, device='cpu', dtype=torch.float64, event=True, compile=False)
    sp_pre = net.sp['pre'].tolist()
    e = {p: sp_pre.index(p) for p in (0, 2, 4)}
    cx = {'er_edge': [e[0]], 'er_pre': [0], 'er_post': [1], 'fb_edge': [e[2]], 'fb_pre': [2], 'fb_post': [3], 'eff_edge': [e[4]], 'eff_pre': [4], 'eff_post': [5], 'eff_E_rev': [P.E_exc_mV], 'ppm3_idx': [6]}
    return (net, R.CXPlasticity(net, cx), e)

def run(cx, seconds, hold):
    for _ in range(round(seconds * 1000 / P.dt_plast_ms)):
        hold()
        cx.clock2(0.0)

def test_er_to_epg_depresses_when_epg_calcium_high_and_potentiates_when_low():
    for c_epg, sign in ((1.5, -1), (0.0, +1)):
        net, cx, e = toy()

        def hold():
            cx.e_trace[0] = 1.0
            net.s['c_ca'][1] = c_epg
        run(cx, 1.0, hold)
        om = float(torch.clamp(1 + cx.w['er']['fast'] + cx.w['er']['slow'], P.omega_min, P.omega_max)[0])
        assert (om - 1.0) * sign > 0.01, (c_epg, om)
        assert abs(float(net.sp['w_f'][e[0]]) - float(cx.base['er'][0][0]) * om) < 1e-12

def test_fb_to_goal_needs_fb_dopamine():
    net, cx, _ = toy()

    def hold():
        cx.e_trace[2] = 1.0
        net.s['c_ca'][3] = 1.0
    run(cx, 1.0, hold)
    assert float(cx.w['fb']['fast'].abs().max()) == 0.0
    cx.D_fb = torch.tensor(1.0, dtype=torch.float64)
    run(cx, 1.0, hold)
    assert float(cx.w['fb']['fast'][0]) > 0.01

def test_efference_weight_falls_when_lptc_depolarised_by_an_excitatory_copy():
    net, cx, _ = toy()

    def hold():
        net.s['V'][5] = -45.0
        net.s['E_L_mod'][5] = -52.0
        net.s['s_bar'][4] = 0.05
    run(cx, 1.0, hold)
    w = float(cx.w['eff']['fast'][0])
    assert w < 0 and w >= P.w_eff_min
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
