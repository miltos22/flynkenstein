import json
import re
from pathlib import Path
import numpy as np
from . import params as P
CACHE = Path(__file__).resolve().parents[1] / 'cache'
CLASS_NAMES_V33C = ('I', 'II', 'III', 'IV', 'V', 'VI', 'IIIr', 'Vd')
IIIR, VD = (6, 7)
V33C_GLOBALS = {'alpha_Ca_uM_per_pA_ms': 0.001, 'dc_spike_uM': 0.1, 'm_inf_CaT_half_mV': 45.0, 'm_inf_CaT_k_mV': 2.5}
V33C_CLASS_OVERRIDES = {0: {'g_KCa0_nS': 0.08, 'a0_nS': 0.02, 'dg_adapt0_nS': 0.03, 'g_NaP0_nS': 0.06}, 5: {'g_KCa0_nS': 0.08, 'a0_nS': 0.02, 'dg_adapt0_nS': 0.03, 'g_NaP0_nS': 0.06}, 1: {'g_h0_nS': 0.01, 'g_CaT0_nS': 0.0, 'g_KCa0_nS': 0.04, 'a0_nS': 0.02, 'dg_adapt0_nS': 0.02, 'g_NaP0_nS': 0.08}, 2: {'r_target_Hz': 5.0}}
V33C_NEW_CLASSES = {IIIR: (2, {'g_CaT0_nS': 0.01, 'g_h0_nS': 0.05, 'g_NaP0_nS': 0.02, 'g_KCa0_nS': 0.1, 'a0_nS': 0.05, 'dg_adapt0_nS': 0.05, 'E_L_base_mV': -50.5, 'r_target_Hz': 1.5}), VD: (4, {'g_h0_nS': 0.0, 'g_CaT0_nS': 0.0, 'g_KCa0_nS': 0.05, 'a0_nS': 0.0})}
CALIBRATE = (0, 2, 5)
ACTIVE = None

def apply(name):
    global ACTIVE
    if name is None:
        return
    assert name == 'v33c', name
    if ACTIVE == name:
        return
    for k, v in V33C_GLOBALS.items():
        setattr(P, k, v)
    cls = {k: list(v) for k, v in P.CLASS.items()}
    for c, ov in V33C_CLASS_OVERRIDES.items():
        for k, v in ov.items():
            cls[k][c] = v
    for new, (base, ov) in sorted(V33C_NEW_CLASSES.items()):
        for k in cls:
            cls[k].append(ov.get(k, cls[k][base]))
    P.CLASS = {k: tuple(v) for k, v in cls.items()}
    P.CLASS_NAMES = CLASS_NAMES_V33C
    _refresh_reference()
    cal = _calibration()
    cls = {k: list(v) for k, v in P.CLASS.items()}
    for c, e in cal.items():
        cls['E_L_base_mV'][int(c)] = e
    P.CLASS = {k: tuple(v) for k, v in cls.items()}
    _refresh_reference()
    ACTIVE = name

def _refresh_reference():
    import torch
    from . import reference as R
    R.CLS = {k: torch.tensor(v, dtype=torch.float64) for k, v in P.CLASS.items()}

def isolated_rates(c, E_L, seconds=3.0, skip=1.0):
    import torch
    from . import reference as R
    E = torch.as_tensor(np.asarray(E_L, dtype=np.float64))
    s = R.init_state(torch.full((E.numel(),), c))
    s['E_L'] = E.clone()
    s['E_L_mod'] = E.clone()
    s['V'] = E.clone()
    I0 = torch.zeros(E.numel(), dtype=torch.float64)
    cnt = torch.zeros(E.numel(), dtype=torch.float64)
    for n in range(round(seconds * 1000 / P.dt_ms)):
        R.step_node(s, I0)
        if n >= round(skip * 1000 / P.dt_ms):
            cnt += s['spikes']
    return (cnt / (seconds - skip)).numpy()

def _calibration(path=CACHE / 'profile_v33c_calibration.json'):
    key = json.dumps({k: [P.CLASS[k][c] for c in CALIBRATE] for k in sorted(P.CLASS)} | {g: getattr(P, g) for g in sorted(V33C_GLOBALS)}, sort_keys=True)
    if path.exists():
        d = json.loads(path.read_text())
        if d.get('key') == key:
            return {int(c): v for c, v in d['E_L_base_mV'].items()}
    out, log = ({}, {})
    for c in CALIBRATE:
        target = P.CLASS['r_target_Hz'][c]
        g1 = np.round(np.arange(-62.0, -39.95, 0.1), 4)
        r1 = isolated_rates(c, g1)
        hit = np.nonzero(r1 >= target)[0]
        if not hit.size or hit[0] == 0:
            raise RuntimeError(f'class {c}: target {target} Hz not bracketed on [-62, -40] mV (rates {r1.min()}..{r1.max()})')
        k = hit[0]
        g2 = np.round(np.arange(g1[k - 1], g1[k] + 1e-09, 0.005), 4)
        r2 = isolated_rates(c, g2)
        best = int(np.argmin(np.abs(r2 - target)))
        out[c] = float(g2[best])
        log[c] = {'target_Hz': target, 'E_L_base_mV': out[c], 'rate_Hz': float(r2[best]), 'coarse': [[float(x), float(y)] for x, y in zip(g1, r1)]}
    CACHE.mkdir(exist_ok=True)
    path.write_text(json.dumps({'key': key, 'E_L_base_mV': out, 'log': log}, indent=1))
    return out

def assign_v33c(names, sclass, nt, kc, NT_CODE):
    n = len(names)
    m = lambda rx: np.array([bool(re.match(rx, x)) for x in names])
    cls = np.full(n, IIIR, np.int8)
    ol = (sclass == 'ol_intrinsic') | m('^(L[1-5]|Mi\\d|Tm\\d|Dm\\d|T4|T5|HS|VS)')
    cls[ol | m('^(APL|DPM)$')] = 4
    cls[m('^(ER[1-6]|ExR1|FB[67])')] = 2
    pace = m('(LNv|^LNd|^DN1|^NPF|^IPC$)') | (nt == NT_CODE['octopamine']) | (nt == NT_CODE['serotonin'])
    cls[pace] = 0
    cls[m('^(APL|DPM)$')] = 4
    cls[m('^(EPGt?|PEG|PEN_a|PEN_b)')] = 1
    cls[m('^Delta7$')] = VD
    cls[nt == NT_CODE['dopamine']] = 5
    cls[kc] = 3
    return cls

def ring_normalise(j, i, w_syn, rec, is_exc, W, cls, matched, alpha_tilde, tau_bar_fn):
    g_L0 = np.array(P.CLASS['g_L0_nS'], float)
    ring_post = matched['EPG'] | matched['PEG'] | matched['PEN_a'] | matched['PEN_b'] | matched['Delta7']
    ring_exc = matched['EPG'] | matched['PEG'] | matched['PEN_a'] | matched['PEN_b']
    W = W.astype(np.float64).copy()
    rec = rec.copy()
    groups = (('ring_exc', ring_exc[j] & ring_post[i] & is_exc, 3.2, 1), ('delta7_inh', matched['Delta7'][j] & (matched['EPG'] | matched['PEN_a'] | matched['PEN_b'] | matched['Delta7'])[i] & ~is_exc, 2.6, 2), ('er_inh', matched['ER'][j] & matched['EPG'][i] & ~is_exc, 1.2, None))
    report = {}
    for name, sel, g0f, code in groups:
        if not sel.any():
            report[name] = 0
            continue
        if code is not None:
            rec[sel] = code
        tb = tau_bar_fn(rec[sel])
        n_in = np.bincount(i[sel], weights=w_syn[sel].astype(np.float64), minlength=len(cls))
        W[sel] = g0f * g_L0[cls[i[sel]]] * w_syn[sel] / (alpha_tilde[cls[j[sel]]] * tb * n_in[i[sel]] + 1e-09)
        report[name] = int(sel.sum())
    return (W.astype(np.float32), rec, report)

def kc_normalise(j, i, w_syn, rec, is_exc, W, cls, kc, matched, alpha_tilde, tau_bar_fn):
    g_L0 = np.array(P.CLASS['g_L0_nS'], float)
    G0 = P.G0_factor * g_L0
    W = W.astype(np.float64).copy()
    keep = ~(kc[j] & kc[i])
    report = {'kc_kc_dropped': int((~keep).sum())}
    groups = (('calyx_exc', keep & kc[i] & is_exc & ~kc[j]), ('kc_to_outputs_exc', kc[j] & (matched['MBON'] | matched['APL'] | matched['DPM'])[i] & is_exc))
    for name, sel in groups:
        if not sel.any():
            report[name] = 0
            continue
        tb = tau_bar_fn(rec[sel])
        n_in = np.bincount(i[sel], weights=w_syn[sel].astype(np.float64), minlength=len(cls))
        W[sel] = G0[cls[i[sel]]] * w_syn[sel] / (alpha_tilde[cls[j[sel]]] * tb * n_in[i[sel]] + 1e-09)
        report[name] = int(sel.sum())
    return (W.astype(np.float32), keep, report)
V33C_CLASS_OVERRIDES[3] = {'V_T_mV': -41.0, 'Delta_T_mV': 1.2}
APL_DPM_V_REL, APL_DPM_K_REL = (-45.0, 2.5)
EPG_GAP_NS = 0.2

def q678_normalise(j, i, w_syn, rec, is_exc, W, cls, kc, matched, dan, insts, alpha_tilde, tau_bar_fn):
    g_L0 = np.array(P.CLASS['g_L0_nS'], float)
    W = W.astype(np.float64).copy()
    rec = rec.copy()
    apl, dpm, mbon = (matched['APL'], matched['DPM'], matched['MBON'])
    epg, peg, d7 = (matched['EPG'], matched['PEG'], matched['Delta7'])
    pen = matched['PEN_a'] | matched['PEN_b']
    side = np.array([(re.search('_([LR])\\d+$', s) or [None, ''])[1] for s in insts])
    one = np.ones_like(alpha_tilde)
    rep = {}

    def renorm(name, sel, g0f, code=None, at=alpha_tilde):
        if not sel.any():
            rep[name] = 0
            return
        if code is not None:
            rec[sel] = code
        tb = tau_bar_fn(rec[sel])
        n_in = np.bincount(i[sel], weights=w_syn[sel].astype(np.float64), minlength=len(cls))
        W[sel] = g0f * g_L0[cls[i[sel]]] * w_syn[sel] / (at[cls[j[sel]]] * tb * n_in[i[sel]] + 1e-09)
        rep[name] = int(sel.sum())
    other_kc_out = kc[j] & ~kc[i] & ~(mbon | apl | dpm | dan)[i]
    W[other_kc_out] *= alpha_tilde[3]
    rep['kc_out_unboosted'] = int(other_kc_out.sum())
    renorm('apl_to_kc_inh', apl[j] & kc[i] & ~is_exc, 6.5, code=3)
    renorm('kc_to_apl_exc', kc[j] & apl[i] & is_exc, 4.0)
    renorm('calyx_to_kc_exc', ~kc[j] & kc[i] & is_exc, 1.05)
    renorm('kc_to_dan_exc', kc[j] & dan[i] & is_exc, 0.08, at=one)
    renorm('nonkc_to_dan_exc', ~kc[j] & dan[i] & is_exc, 1.8)
    renorm('hold_to_epg', (epg | peg)[j] & epg[i] & is_exc, 2.4, code=1)
    renorm('penL_to_epg', pen[j] & (side[j] == 'L') & epg[i] & is_exc, 0.4, code=1)
    renorm('penR_to_epg', pen[j] & (side[j] == 'R') & epg[i] & is_exc, 0.4, code=1)
    renorm('delta7_to_epg_inh', d7[j] & epg[i] & ~is_exc, 3.0, code=2)
    return (W.astype(np.float32), rec, rep)

def epg_gap_pairs(matched, insts):
    epg = np.nonzero(matched['EPG'])[0]
    glom = {}
    for k in epg:
        m = re.search('_([LR]\\d)$', insts[k])
        if m:
            glom.setdefault(m.group(1), []).append(int(k))
    pre, post = ([], [])
    for g, members in glom.items():
        for a in members:
            for b in members:
                if a != b:
                    pre.append(a)
                    post.append(b)
    return (np.array(pre, np.int64), np.array(post, np.int64))
