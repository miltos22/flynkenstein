import json
import os
import re
import time
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
from . import params
DATA_DIR = os.environ.get('FLYBRAIN_DATA', 'data/malecns_v1.0')
DEFAULT_PATH = os.path.join(os.environ.get('FLYBRAIN_CACHE', 'cache'), 'connectome_v1.npz')
ANN_FEATHER = os.path.join(DATA_DIR, 'body-annotations-male-cns-v1.0-minconf-0.5.feather')
NT_FEATHER = os.path.join(DATA_DIR, 'body-neurotransmitters-male-cns-v1.0.feather')
W_FEATHER = os.path.join(DATA_DIR, 'connectome-weights-male-cns-v1.0-minconf-0.5.feather')
VOXEL_UM = 0.008
GRID_NX, GRID_NY, GRID_NZ = (32, 32, 64)
GRID_MIN_VOX = (2468, 4758, 10154)
GRID_MAX_VOX = (93668, 68996, 134531)
NT_CODE = {'acetylcholine': 0, 'glutamate': 1, 'gaba': 2, 'histamine': 3, 'dopamine': 4, 'octopamine': 5, 'serotonin': 6, 'tyramine': 7}
NT_NAMES = ('ACh', 'Glu', 'GABA', 'His', 'DA', 'OA', '5-HT', 'TA')
FAST_CODES = (0, 1, 2, 3)
AMINE_CODES = (4, 5, 6, 7)
EPS = 1e-09
RHO_EI = params.rho_EI_c
G0_FACTOR = params.G0_factor
TAU_SYN4 = (params.tau_syn_excf_ms, params.tau_syn_excs_ms, params.tau_syn_inhf_ms, params.tau_syn_inhs_ms)
SPEED_UM_PER_MS = 500.0
SYN_DELAY_MS = 0.5
GRADED_RE = re.compile('^(L[1-5]|Mi1|Tm[1-4]|T4[abcd]|T5[abcd]|APL|DPM|HS|VS)$')
CXPLATEAU_RE = re.compile('^(EPGt?|PEG|PEN_a|PEN_b|Delta7)$')
CLOCK_RE = re.compile('(LNv|^LNd|^5thsLNv|^DN1|^LPN)')
CXMB_CLASSES = frozenset(('CX', 'MBON', 'Kenyon_Cell'))
TYPE_SETS = (('APL', '^APL$'), ('DPM', '^DPM$'), ('MBON', '^MBON'), ('PAM', '^PAM'), ('PPL1', '^PPL1'), ('EPG', '^EPGt?$'), ('PEG', '^PEG$'), ('PEN_a', '^PEN_a'), ('PEN_b', '^PEN_b'), ('Delta7', '^Delta7$'), ('ER', '^ER[0-9]'), ('ER5', '^ER5$'), ('ER2', '^ER2_'), ('ER4d', '^ER4d$'), ('PFL2', '^PFL2$'), ('PFL3', '^PFL3$'), ('hDeltaK', '^hDeltaK$'), ('FB', '^FB[0-9]'), ('dFB', '^FB[67]'), ('LPTC', '^(HS|VS)'), ('IPC', '^IPC$'), ('DH44', '^DH44$'), ('DNa02', '^DNa02$'), ('SMP', '^SMP'))

def _derive_alpha_tilde():
    r_bar = np.array(params.CLASS['alpha_hat'], float) / params.tau_char_ms
    u0 = np.array(params.CLASS['U0'], float)
    t_fac = np.array(params.CLASS['tau_fac_ms'], float)
    t_rec = np.array(params.CLASS['tau_rec_ms'], float)
    u_inf = u0 * (1.0 + t_fac * r_bar) / (1.0 + u0 * t_fac * r_bar)
    x_inf = 1.0 / (1.0 + t_rec * u_inf * r_bar)
    alpha_hat = np.array(params.CLASS['alpha_hat'], float)
    return np.maximum(alpha_hat * u_inf * x_inf, params.alpha_hat_floor)

def _tau_bar(rec):
    out = np.zeros(len(rec), np.float64)
    for code, betas in params.beta_rec.items():
        m = rec == code
        if m.any():
            out[m] = sum((b * t for b, t in zip(betas, TAU_SYN4))) / params.tau_char_ms
    return out

def build(out_path=DEFAULT_PATH, default_class=2, profile=None):
    from . import profiles
    profiles.apply(profile)
    t0 = time.time()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ann = feather.read_table(ANN_FEATHER, columns=['bodyId', 'type', 'instance', 'superclass', 'class', 'somaLocation']).to_pydict()
    sc = ann['superclass']
    keep = np.array([s is not None and s != '' for s in sc])
    bid = np.asarray(ann['bodyId'], np.int64)[keep]
    typ = ann['type']
    ins = ann['instance']
    names = np.empty(int(keep.sum()), object)
    j = 0
    for k in range(len(sc)):
        if not keep[k]:
            continue
        t = typ[k]
        names[j] = t if t is not None and t != '' else ins[k] or ''
        j += 1
    names = names.astype(str)
    sclass = np.array([s for s, k in zip(sc, keep) if k], object).astype(str)
    classv = np.array([c if c is not None else '' for c, k in zip(ann['class'], keep) if k], object)
    soma = [s for s, k in zip(ann['somaLocation'], keep) if k]
    insts = np.array([ins[k] or '' for k in range(len(sc)) if keep[k]], object).astype(str)
    del ann, typ, ins, sc
    n_all = len(bid)
    kc = np.array([c == 'Kenyon_Cell' for c in classv])
    idx_kc = np.where(kc)[0]
    other = np.where(~kc)[0]
    other = other[np.argsort(bid[other], kind='stable')]
    order = np.concatenate([idx_kc, other])
    n_kc = len(idx_kc)
    inv = np.empty(n_all, np.int64)
    inv[order] = np.arange(n_all)
    pos = np.full((n_all, 3), np.nan, np.float32)
    has = np.zeros(n_all, bool)
    for k, sl in enumerate(soma):
        if sl is not None and len(sl) == 3:
            pos[k] = np.asarray(sl, np.float32) * VOXEL_UM
            has[k] = True
    del soma
    tnames = np.unique(names[names != ''])
    type_id = np.searchsorted(tnames, names)
    noname = names == ''
    type_id[noname] = -1
    snames = np.unique(sclass)
    superclass_id = np.searchsorted(snames, sclass).astype(np.int16)
    by_type = {}
    by_super = {}
    for k in range(n_all):
        if not has[k]:
            continue
        by_type.setdefault(int(type_id[k]), []).append(pos[k])
        by_super.setdefault(int(superclass_id[k]), []).append(pos[k])
    tmed = {t: np.median(np.asarray(v, np.float32), axis=0) for t, v in by_type.items()}
    smed = {s: np.median(np.asarray(v, np.float32), axis=0) for s, v in by_super.items()}
    for k in range(n_all):
        if has[k]:
            continue
        m = tmed.get(int(type_id[k]))
        if m is None:
            m = smed.get(int(superclass_id[k]))
        if m is not None:
            pos[k] = m
    pos = np.nan_to_num(pos, nan=0.0)
    del by_type, by_super, tmed, smed
    mn = np.asarray(GRID_MIN_VOX, np.float32) * VOXEL_UM
    mx = np.asarray(GRID_MAX_VOX, np.float32) * VOXEL_UM
    extent = mx - mn + VOXEL_UM
    ncell = np.array([GRID_NX, GRID_NY, GRID_NZ], np.float32)
    spacing = extent / ncell
    idx = np.floor((pos - mn) / spacing).astype(np.int64)
    idx[:, 0] = np.clip(idx[:, 0], 0, GRID_NX - 1)
    idx[:, 1] = np.clip(idx[:, 1], 0, GRID_NY - 1)
    idx[:, 2] = np.clip(idx[:, 2], 0, GRID_NZ - 1)
    voxel = ((idx[:, 0] * GRID_NY + idx[:, 1]) * GRID_NZ + idx[:, 2]).astype(np.int32)
    nt_tbl = feather.read_table(NT_FEATHER, columns=['body', 'consensus_nt', 'celltype_predicted_nt', 'predicted_nt'])
    nt_tbl = nt_tbl.filter(pc.is_in(nt_tbl['body'], value_set=pa.array(bid)))
    nb = nt_tbl['body'].to_numpy()
    so = np.argsort(nb, kind='stable')
    nb = nb[so]
    cols = [np.asarray(nt_tbl[c].to_pylist(), object)[so] for c in ('consensus_nt', 'celltype_predicted_nt', 'predicted_nt')]
    del nt_tbl
    p = np.searchsorted(nb, bid)
    p = np.clip(p, 0, len(nb) - 1)
    valid = nb[p] == bid
    code = np.full(n_all, -1, np.int8)
    for col in cols:
        sel = (code < 0) & valid
        if sel.any():
            code[sel] = [NT_CODE.get(col[k], -1) for k in p[sel]]
    code[code < 0] = 0
    nt = code
    del cols
    graded = np.array([bool(GRADED_RE.match(n)) for n in names])
    cxplat = np.array([bool(CXPLATEAU_RE.match(n)) for n in names])
    clock = np.array([bool(CLOCK_RE.search(n)) for n in names])
    dan = nt == NT_CODE['dopamine']
    oa = nt == NT_CODE['octopamine']
    cls = np.full(n_all, default_class, np.int8)
    cls[cxplat] = 1
    cls[graded] = 4
    cls[dan] = 5
    cls[oa | clock] = 0
    cls[kc] = 3
    if profile == 'v33c':
        cls = profiles.assign_v33c(names, sclass, nt, kc, NT_CODE)
    is_graded = np.array(params.CLASS['Delta_T_mV'])[cls] == 0
    sets = {}
    matched = {name: np.array([bool(re.match(pat, n)) for n in names]) for name, pat in TYPE_SETS}
    matched['clock'] = clock
    sets['KC'] = np.where(kc)[0].astype(np.int32)
    for name in matched:
        sets[name] = np.where(matched[name])[0].astype(np.int32)
    sets['DAN'] = np.where(dan)[0].astype(np.int32)
    sets['OA'] = np.where(oa)[0].astype(np.int32)
    sets['SER'] = np.where(nt == NT_CODE['serotonin'])[0].astype(np.int32)
    sets['descending'] = np.where(sclass == 'descending_neuron')[0].astype(np.int32)
    motor_m = (sclass == 'cb_motor') | (sclass == 'vnc_motor')
    sets['motor'] = np.where(motor_m)[0].astype(np.int32)
    sensory_m = np.array(['sensory' in s for s in sclass])
    sets['sensory'] = np.where(sensory_m)[0].astype(np.int32)
    wt = feather.read_table(W_FEATHER, columns=['body_pre', 'body_post', 'weight'], memory_map=True)
    vs = pa.array(bid)
    m = pc.and_(pc.is_in(wt['body_pre'], value_set=vs), pc.is_in(wt['body_post'], value_set=vs))
    raw = wt.filter(m)
    n_raw = raw.num_rows
    pre_b = raw['body_pre'].to_numpy()
    post_b = raw['body_post'].to_numpy()
    w_syn = raw['weight'].to_numpy()
    del wt, raw, m, vs
    sid = np.argsort(bid, kind='stable')
    sbid = bid[sid]
    pi = sid[np.searchsorted(sbid, pre_b)]
    qi = sid[np.searchsorted(sbid, post_b)]
    del pre_b, post_b, sbid, sid
    self_loop = pi == qi
    n_self = int(self_loop.sum())
    ns = ~self_loop
    amine_pre = np.isin(nt[pi], AMINE_CODES)
    drop_amine = ns & amine_pre
    n_amine = int(drop_amine.sum())
    keep_e = ns & ~amine_pre
    n_kept = int(keep_e.sum())
    j = pi[keep_e]
    i = qi[keep_e]
    w_syn = w_syn[keep_e].astype(np.int64)
    del pi, qi, self_loop, ns, amine_pre, drop_amine, keep_e
    ntj = nt[j]
    is_exc = (ntj == 0) | (ntj == 1) & motor_m[i]
    cxmb = np.array([c in CXMB_CLASSES for c in classv])
    rec = np.empty(len(j), np.uint8)
    exc_target_cxmb = is_exc & cxmb[i]
    glu_inh = ~is_exc & (ntj == 1)
    his = ntj == 3
    rec[exc_target_cxmb] = 1
    rec[is_exc & ~cxmb[i]] = 0
    rec[glu_inh | his] = 2
    rec[~is_exc & (ntj == 2)] = 3
    del ntj, cxmb
    alpha_tilde = _derive_alpha_tilde()
    tb = _tau_bar(rec)
    g_L0 = np.array(params.CLASS['g_L0_nS'], float)
    G0c = G0_FACTOR * g_L0
    n_exc_in = np.bincount(i[is_exc], weights=w_syn[is_exc].astype(np.float64), minlength=n_all)
    n_inh_in = np.bincount(i[~is_exc], weights=w_syn[~is_exc].astype(np.float64), minlength=n_all)
    den_exc = alpha_tilde[cls[j]] * tb * n_exc_in[i] + EPS
    den_inh = alpha_tilde[cls[j]] * tb * n_inh_in[i] + EPS
    gain = np.where(is_exc, G0c[cls[i]], RHO_EI * G0c[cls[i]])
    den = np.where(is_exc, den_exc, den_inh)
    W = (gain * w_syn / den).astype(np.float32)
    del den_exc, den_inh, den, gain
    ring_report = None
    if profile == 'v33c':
        W, rec, ring_report = profiles.ring_normalise(j, i, w_syn, rec, is_exc, W, cls, matched, alpha_tilde, _tau_bar)
        W, keep_kc, kc_report = profiles.kc_normalise(j, i, w_syn, rec, is_exc, W, cls, kc, matched, alpha_tilde, _tau_bar)
        ring_report.update(kc_report)
        j, i, w_syn, rec, is_exc, W = (j[keep_kc], i[keep_kc], w_syn[keep_kc], rec[keep_kc], is_exc[keep_kc], W[keep_kc])
        n_kept = int(keep_kc.sum())
        W, rec, q_report = profiles.q678_normalise(j, i, w_syn, rec, is_exc, W, cls, kc, matched, dan, insts, alpha_tilde, _tau_bar)
        ring_report.update(q_report)
    d = pos[j] - pos[i]
    dist = np.sqrt((d * d).sum(1))
    del d
    t_ms = SYN_DELAY_MS + dist / SPEED_UM_PER_MS
    delay = np.clip(np.rint(t_ms / params.dt_ms), 3, 63).astype(np.uint8)
    del dist, t_ms
    j = inv[j]
    i = inv[i]
    fgrad = is_graded[order]
    gpre = fgrad[j]
    o = np.lexsort((j, ~gpre, i))
    del gpre
    j = j[o]
    i = i[o]
    w_syn = w_syn[o]
    rec = rec[o]
    is_exc = is_exc[o]
    W = W[o]
    delay = delay[o]
    del o
    col_idx = j.astype(np.int32)
    n_syn = w_syn.astype(np.int32)
    M = len(col_idx)
    counts = np.bincount(i, minlength=n_all)
    row_ptr = np.zeros(n_all + 1, np.int64)
    np.cumsum(counts, out=row_ptr[1:])
    grad_per_row = np.bincount(i, weights=fgrad[col_idx].astype(np.float64), minlength=n_all)
    row_split = row_ptr[:-1] + grad_per_row.astype(np.int64)
    body_id = bid[order]
    out = {'profile': np.array(profile or 'v3.2'), **_profile_extras(profile, matched, insts, inv, order, n_all), 'ring_report': np.array(json.dumps(ring_report)), 'body_id': body_id, 'cls': cls[order], 'is_graded': is_graded[order], 'nt': nt[order], 'pos_um': pos[order], 'has_soma': has[order], 'voxel': voxel[order], 'grid_shape': np.array([GRID_NX, GRID_NY, GRID_NZ], np.int64), 'grid_origin_um': mn, 'grid_spacing_um': spacing, 'n_kc': np.int64(n_kc), 'type_names': tnames, 'type_id': type_id[order].astype(np.int32), 'superclass_names': snames, 'superclass_id': superclass_id[order], 'row_ptr': row_ptr, 'row_split': row_split, 'col_idx': col_idx, 'n_syn': n_syn, 'rec': rec, 'is_exc': is_exc, 'delay': delay, 'W_struct': W, 'n_edges_raw_neuron': np.int64(n_raw), 'n_edges_amine_dropped': np.int64(n_amine), 'n_self_loops': np.int64(n_self), 'n_edges_kept': np.int64(n_kept)}
    for name, arr in sets.items():
        out['set_' + name] = inv[arr].astype(np.int32)
    np.savez(out_path, **out)
    summary = _summarize(out, time.time() - t0)
    print(summary)
    return out

def _summarize(out, dt):
    cls = out['cls']
    nt = out['nt']
    rc = out['rec']
    W = out['W_struct']
    j = out['col_idx']
    lines = []
    lines.append('=== connectome_v1 summary (%.1f s) ===' % dt)
    lines.append('N=%d  n_kc=%d  M=%d' % (len(cls), int(out['n_kc']), len(j)))
    lines.append('per-class: ' + str(np.bincount(cls, minlength=len(params.CLASS['C_m_pF'])).tolist()))
    lines.append('per-NT (ACh,Glu,GABA,His,DA,OA,5HT,TA): ' + str(np.bincount(nt, minlength=8).tolist()))
    lines.append('edges raw=%d amine_dropped=%d self_loops=%d kept=%d' % (int(out['n_edges_raw_neuron']), int(out['n_edges_amine_dropped']), int(out['n_self_loops']), int(out['n_edges_kept'])))
    lines.append('W_struct min/median/max per presynaptic class:')
    for c in range(len(params.CLASS['C_m_pF'])):
        m = cls[j] == c
        if m.any():
            lines.append('  class %s: %.4g / %.4g / %.4g  (n=%d)' % (params.CLASS_NAMES[c], W[m].min(), np.median(W[m]), W[m].max(), int(m.sum())))
    hist = np.bincount(out['delay'].astype(np.int64), minlength=64)
    lines.append('delay histogram idx:count: ' + str({int(k): int(v) for k, v in enumerate(hist) if v}))
    sizes = {k[4:]: int(len(v)) for k, v in out.items() if k.startswith('set_')}
    lines.append('set sizes: ' + str(sizes))
    return '\n'.join(lines)

def _profile_extras(profile, matched, insts, inv, order, n_all):
    if profile != 'v33c':
        return {}
    from . import profiles
    v_rel = np.full(n_all, params.V_rel_mV, np.float32)
    k_rel = np.full(n_all, params.k_rel_mV, np.float32)
    m = matched['APL'] | matched['DPM']
    v_rel[m], k_rel[m] = (profiles.APL_DPM_V_REL, profiles.APL_DPM_K_REL)
    gp, gq = profiles.epg_gap_pairs(matched, insts)
    return {'V_rel_mV': v_rel[order], 'k_rel_mV': k_rel[order], 'gap_pre': inv[gp].astype(np.int64), 'gap_post': inv[gq].astype(np.int64), 'gap_g_nS': np.full(gp.size, profiles.EPG_GAP_NS, np.float32), 'gap_alpha': np.ones(gp.size, np.float32)}

def load(path=DEFAULT_PATH):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
if __name__ == '__main__':
    import resource
    build()
    print('peak RSS MB: %.1f' % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0))
