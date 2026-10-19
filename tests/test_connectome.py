import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flybrain import connectome as C
from flybrain import params
D = C.load()

def _alpha_tilde():
    r_bar = np.array(params.CLASS['alpha_hat'], float) / params.tau_char_ms
    u0 = np.array(params.CLASS['U0'], float)
    tf = np.array(params.CLASS['tau_fac_ms'], float)
    tr = np.array(params.CLASS['tau_rec_ms'], float)
    u = u0 * (1 + tf * r_bar) / (1 + u0 * tf * r_bar)
    x = 1.0 / (1 + tr * u * r_bar)
    return np.maximum(np.array(params.CLASS['alpha_hat'], float) * u * x, 0.002)

def _tau_bar(rec):
    out = np.zeros(len(rec), float)
    for code, betas in params.beta_rec.items():
        m = rec == code
        out[m] = sum((b * t for b, t in zip(betas, C.TAU_SYN4))) / params.tau_char_ms
    return out

def test_1_neurons():
    bid = D['body_id']
    n = len(bid)
    assert n == 166700, n
    assert int(D['n_kc']) == 4064
    assert np.all(D['cls'][:D['n_kc']] == 3)
    assert np.all(np.diff(bid[D['n_kc']:]) > 0)

def test_2_edge_counts():
    raw = int(D['n_edges_raw_neuron'])
    assert raw == 25582938, raw
    assert int(D['n_edges_kept']) + int(D['n_edges_amine_dropped']) + int(D['n_self_loops']) == raw

def test_3_csr_valid():
    rp = D['row_ptr']
    rs = D['row_split']
    col = D['col_idx']
    M = len(col)
    assert rp[0] == 0
    assert rp[-1] == M
    assert np.all(np.diff(rp) >= 0)
    assert np.all(rs <= rp[1:]) and np.all(rs >= rp[:-1])
    assert col.min() >= 0 and col.max() < len(D['body_id'])
    row = np.repeat(np.arange(len(rp) - 1), np.diff(rp))
    g = D['is_graded'][col]
    pos = np.arange(M)
    before = pos < rs[row]
    assert np.all(g[before])
    assert not np.any(g[~before])

def test_4_normalisation():
    col = D['col_idx']
    rec = D['rec']
    W = D['W_struct'].astype(np.float64)
    a = _alpha_tilde()[D['cls'][col]]
    tb = _tau_bar(rec)
    contrib = W * a * tb
    n = len(D['cls'])
    row = np.repeat(np.arange(n), np.diff(D['row_ptr']))
    G0 = params.G0_factor * np.array(params.CLASS['g_L0_nS'], float)[D['cls']]
    exc = D['is_exc']
    s_exc = np.bincount(row[exc], weights=contrib[exc], minlength=n)
    inh = ~exc
    s_inh = np.bincount(row[inh], weights=contrib[inh], minlength=n)
    ne = np.bincount(row[exc], minlength=n)
    ni = np.bincount(row[inh], minlength=n)
    se = ne > 0
    si = ni > 0
    assert np.max(np.abs(s_exc[se] - G0[se]) / G0[se]) < 0.0001
    assert np.max(np.abs(s_inh[si] - C.RHO_EI * G0[si]) / (C.RHO_EI * G0[si])) < 0.0001

def test_5_delay_range():
    d = D['delay']
    assert d.min() >= 3 and d.max() <= 63

def test_6_sets():
    names = [k for k in D if k.startswith('set_')]
    assert names
    sizes = {k[4:]: int(len(D[k])) for k in names}
    print('set sizes:', sizes)
    empty = [k for k, v in sizes.items() if v == 0]
    assert not empty, empty
if __name__ == '__main__':
    fns = [test_1_neurons, test_2_edge_counts, test_3_csr_valid, test_4_normalisation, test_5_delay_range, test_6_sets]
    for fn in fns:
        fn()
        print('ok', fn.__name__)
    print('ALL TESTS PASSED')
