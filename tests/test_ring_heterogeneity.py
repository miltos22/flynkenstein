import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flybrain import connectome, cxmap, params as P
CONN = Path(__file__).resolve().parents[1] / 'cache' / 'connectome_v1.npz'

def wedge_matrix():
    conn = connectome.load(str(CONN))
    w = cxmap.epg_wedges(conn)
    wedge_of = -np.ones(len(conn['cls']), dtype=int)
    wedge_of[w['idx']] = w['wedge']
    rp, col, W, rec = (conn['row_ptr'], conn['col_idx'], conn['W_struct'].astype(float), conn['rec'])
    M, cnt, row_tot = (np.zeros((16, 16)), np.zeros(16), [])
    for i in w['idx'][w['wedge'] >= 0]:
        a = wedge_of[i]
        cnt[a] += 1
        sl = slice(rp[i], rp[i + 1])
        ex = rec[sl] <= 1
        row_tot.append(W[sl][ex].sum())
        m = ex & (wedge_of[col[sl]] >= 0)
        np.add.at(M[a], wedge_of[col[sl][m]], W[sl][m])
    return (M / cnt[:, None], float(np.mean(row_tot)))

def test_heterogeneity_within_proof4_bound_in_its_own_normalisation():
    M, row_tot = wedge_matrix()
    G0 = 1.8 * P.CLASS['g_L0_nS'][1]
    scale = G0 / row_tot
    off = (np.arange(16)[None, :] - np.arange(16)[:, None]) % 16
    circ = np.array([M[off == d].mean() for d in range(16)])
    dW = (M - circ[off]) * scale
    norm = np.abs(dW).sum(1).max()
    bound = P.Delta_W_bound_factor * G0
    assert norm <= bound, f'||ΔW||_inf = {norm / G0:.3f} G0 > 0.15 G0'
    dI = norm * abs(P.V_active_mV - P.E_exc_mV)
    print(f"  ||ΔW||_inf = {norm / G0:.3f} G0 -> ΔI_asym = {dI:.2f} pA vs plateau {P.SPEC_CHECKS['dI_plateau'][0]} pA")
    assert dI < P.SPEC_CHECKS['dI_plateau'][0]
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
