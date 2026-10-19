import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flybrain import params as P
from flybrain import reference as R

def isolated_rate(cls_idx, E_L, seconds=4.0, skip=1.0):
    s = R.init_state(torch.tensor([cls_idx]))
    s['E_L'] = torch.tensor([E_L], dtype=torch.float64)
    s['E_L_mod'] = s['E_L'].clone()
    I0 = torch.zeros(1, dtype=torch.float64)
    n_skip, n_all, cnt = (round(skip * 1000 / P.dt_ms), round(seconds * 1000 / P.dt_ms), 0)
    for i in range(n_all):
        R.step_node(s, I0)
        if i >= n_skip:
            cnt += int(s['spikes'].item())
    return cnt / (seconds - skip)

def test_pacemakers_cross_threshold_at_E_L_max():
    assert isolated_rate(0, P.CLASS['E_L_max_mV'][0]) > 0.0

def xfail_dan_tonic_3hz_at_E_L_base():
    r = isolated_rate(5, P.CLASS['E_L_base_mV'][5])
    assert abs(r - 3.0) < 0.5, f'class VI at -49.2 mV fires {r:.2f} Hz, spec says 3.0'

def xfail_class_I_reaches_target_within_E_L_box():
    r = isolated_rate(0, P.CLASS['E_L_max_mV'][0])
    assert r >= P.CLASS['r_target_Hz'][0] - 1.0, f'class I at E_L,max fires {r:.2f} Hz < 12.5'
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
        elif name.startswith('xfail_'):
            try:
                fn()
                print('XPASS', name, '- spec claim now holds; update SPEC_FINDINGS F7')
            except AssertionError as e:
                print('XFAIL', name, '-', e, '(SPEC_FINDINGS F7)')
    sys.exit(1 if fails else 0)
