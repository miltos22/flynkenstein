import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flybrain import params as P
from flybrain import reference as R

def toy():
    conn = {'cls': np.array([3, 2, 5, 4], dtype=np.int8), 'row_ptr': np.array([0, 0, 1, 1, 1]), 'col_idx': np.array([0], dtype=np.int32), 'rec': np.array([1], dtype=np.uint8), 'W_struct': np.array([2.0], dtype=np.float32), 'delay': np.array([3], dtype=np.uint8)}
    net = R.Network(conn, device='cpu', dtype=torch.float64, event=True, compile=False)
    mb = {'dan_idx': [2], 'dan_comp': [0], 'syn_kc': [0], 'syn_comp': [0], 'syn_edge': [0], 'comp_aversive': [True], 'dpm_idx': [3], 'n_kc': 1}
    return (net, R.MBPlasticity(net, mb))

def run_clock2(mb, seconds, S=0.0, A=0.0, hold=None):
    for _ in range(round(seconds * 1000 / P.dt_plast_ms)):
        if hold:
            hold()
        mb.clock2(S, A)

def test_no_dopamine_no_change():
    net, mb = toy()
    net.s['c_ca'][0] = 1.0
    mb.e_fwd[0] = 0.5
    run_clock2(mb, 2.0)
    assert float(mb.w_fast.abs().max()) == 0.0 and float(mb.omega()[0]) == 1.0

def test_kc_activity_paired_with_dopamine_depresses():
    net, mb = toy()
    base = float(net.sp['w_f'][0])

    def hold():
        net.s['c_ca'][0] = 1.0
        mb.e_fwd[0] = 0.5
        mb.D[0] = 1.0
    run_clock2(mb, 2.0, hold=hold)
    w = float(mb.w_fast[0])
    assert w < -0.01, f'w_fast {w} should be clearly negative (LTD)'
    om = float(mb.omega()[0])
    assert om < 1.0
    assert abs(float(net.sp['w_f'][0]) - base * om) < 1e-12

def test_dopamine_alone_recovers_depressed_synapse():
    net, mb = toy()
    mb.w_fast[0] = -0.3

    def hold():
        net.s['c_ca'][0] = 0.0
        mb.e_fwd[0] = 0.0
        mb.D[0] = 1.0
    w0 = float(mb.w_fast[0])
    run_clock2(mb, 1.0, hold=hold)
    assert float(mb.w_fast[0]) > w0, 'LTP_solo should push a depressed synapse back up'

def test_cascade_stays_in_box():
    net, mb = toy()
    mb.w_fast[0], mb.w_interm[0], mb.w_slow[0] = (-0.9, -0.95, -0.5)
    net.s['r_bar'][3] = 50.0
    run_clock2(mb, 30.0, S=1.0, A=0.25)
    assert -1.0 <= float(mb.w_interm[0]) <= 1.0 and -1.0 <= float(mb.w_slow[0]) <= 0.0
    assert P.omega_min <= float(mb.omega()[0]) <= P.omega_max

def test_replay_volley_and_veto_match_spec():
    net, mb = toy()
    mb.ltm_av = torch.tensor([2])
    mb.veto_av = torch.tensor([1])
    net.s['s_bar'][0] = P.SPEC_CHECKS['up_state_KC_volley'][0]
    net.s['s_bar'][1] = 0.0
    I = float(mb.replay(S_sleep=1.0, p_up=0)[2])
    assert abs(I - P.SPEC_CHECKS['replay_LTM_current'][0]) < 0.05, I
    net.s['s_bar'][1] = P.SPEC_CHECKS['MBON_veto_threshold'][0]
    assert float(mb.replay(S_sleep=1.0, p_up=0)[2]) == 0.0
    net.s['s_bar'][1] = 0.0
    assert float(mb.replay(S_sleep=0.0, p_up=0)[2]) == 0.0
    assert float(mb.replay(S_sleep=1.0, p_up=1)[2]) == 0.0
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
