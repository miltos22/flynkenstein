import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flybrain import connectome, cxmap
CONN = Path(__file__).resolve().parents[1] / 'cache' / 'connectome_v1.npz'

def test_16_wedges_alternate_left_right_and_tile_the_ring():
    w = cxmap.epg_wedges(connectome.load(str(CONN)))
    sides = [g[0] for g in w['wedge_glom']]
    assert len(sides) == 16
    assert all((sides[k] != sides[(k + 1) % 16] for k in range(16))), f"not alternating: {w['wedge_glom']}"
    gaps = np.diff(np.concatenate([w['wedge_angle_deg'], [w['wedge_angle_deg'][0] + 360]]))
    assert gaps.min() > 10 and gaps.max() < 40, f'wedge spacing {gaps.round(1)} (ideal 22.5)'
    for side in 'LR':
        a = [w['wedge_angle_deg'][list(w['wedge_glom']).index(f'{side}{n}')] for n in range(1, 9)]
        steps = [(a[n + 1] - a[n] + 180) % 360 - 180 for n in range(7)]
        assert all((25 < abs(s) < 65 for s in steps)) and len({np.sign(s) for s in steps}) == 1, (side, np.round(steps, 1))

def test_heading_decoder_recovers_a_bump():
    angles = np.arange(16) * 22.5
    act = np.exp(np.cos(np.radians(angles - 100.0)) * 3)
    assert abs(cxmap.heading_deg(act, angles) - 100.0) < 1e-06
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
