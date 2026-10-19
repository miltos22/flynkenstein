import copy
import json
import math
import time
import numpy as np
import torch
from . import lz
from . import params as P
from . import protocols
_NET_STATE = ('ring', 'buf', 'n')

def snapshot(sim):
    clone = lambda v: v.clone() if torch.is_tensor(v) else copy.deepcopy(v)
    snap = {'s': {k: clone(v) for k, v in sim.net.s.items() if k != 'c'}, 'net': {k: clone(getattr(sim.net, k)) for k in _NET_STATE if hasattr(sim.net, k)}, 'sp_w': (sim.net.sp['w_f'].clone(), sim.net.sp['w_s'].clone()) if hasattr(sim.net, 'sp') else None, 'gen': sim.gen.get_state()}
    for name in ('slow', 'diff', 'mb', 'cx'):
        obj = getattr(sim, name, None)
        if obj is not None:
            snap[name] = {k: clone(v) for k, v in vars(obj).items() if k not in ('net',) and (torch.is_tensor(v) or isinstance(v, (int, float, dict)))}
    return snap

def restore(sim, snap):
    clone = lambda v: v.clone() if torch.is_tensor(v) else copy.deepcopy(v)
    for k, v in snap['s'].items():
        sim.net.s[k] = clone(v)
    for k, v in snap['net'].items():
        setattr(sim.net, k, clone(v))
    if snap['sp_w'] is not None:
        sim.net.sp['w_f'].copy_(snap['sp_w'][0])
        sim.net.sp['w_s'].copy_(snap['sp_w'][1])
    sim.gen.set_state(snap['gen'])
    for name in ('slow', 'diff', 'mb', 'cx'):
        if name in snap:
            obj = getattr(sim, name)
            for k, v in snap[name].items():
                setattr(obj, k, clone(v))

def mr_branching(counts, k_max=40):
    a = np.asarray(counts, dtype=np.float64)
    ks = np.arange(1, k_max + 1)
    r = np.array([np.polyfit(a[:-k], a[k:], 1)[0] for k in ks])
    best = (np.inf, float('nan'))
    for m in np.arange(0.5, 1.2, 0.0001):
        mk = m ** ks
        b = (r * mk).sum() / (mk * mk).sum()
        sse = ((r - b * mk) ** 2).sum()
        if sse < best[0]:
            best = (sse, float(m))
    return best[1]

def run(sim, conn, out_dir, deprivation_s=P.b1_deprivation_s, K=P.K_ens, window_ms=300.0, force_sleep=False, seed=0, log=print, pulse_pA=P.b1_pulse_pA):
    out = {}
    N = sim.net.N
    s = sim.net.s
    is_spk = ~s['is_graded']
    rng = np.random.default_rng(seed)
    smp = np.asarray(conn['set_SMP'])
    targets = torch.as_tensor(rng.choice(smp, P.b1_pulse_n_neurons, replace=False), device=s['V'].device)
    if force_sleep:
        sim.slow.force_sleep = True
    t0 = time.time()
    protocols.boot(sim, seed=seed)
    I0 = torch.zeros(N, device=s['V'].device, dtype=s['V'].dtype)
    steps = round(deprivation_s * 1000 / P.dt_ms)
    per_ms = round(1.0 / P.dt_ms)
    rec_from = max(steps - round(10000 / P.dt_ms), 0)
    counts_t = torch.zeros((steps - rec_from) // per_ms + 1, device=s['V'].device)
    last_sec = torch.zeros((), device=s['V'].device)
    for n in range(steps):
        sim.step(I0)
        if n >= rec_from:
            counts_t[(n - rec_from) // per_ms] += s['spikes'][is_spk].sum()
        if n >= steps - round(1000 / P.dt_ms):
            last_sec += s['spikes'][is_spk].sum()
        if n % 25000 == 0:
            log(f'deprivation t={n * P.dt_ms / 1000:.1f} s wall {time.time() - t0:.0f} s')
    out['mean_rate_Hz_t30'] = float(last_sec) / int(is_spk.sum()) / (min(steps, round(1000 / P.dt_ms)) * P.dt_ms / 1000)
    counts = counts_t[:(steps - rec_from) // per_ms].cpu().numpy()
    out['branching_ratio_MR'] = mr_branching(counts)
    np.save(f'{out_dir}/pop_counts_last10s.npy', counts)
    snap = snapshot(sim)
    bins = round(window_ms)
    sum_d = torch.zeros(bins, N, dtype=torch.float64, device=s['V'].device)
    sum_b = torch.zeros_like(sum_d)
    sum_b2 = torch.zeros_like(sum_d)
    for k in range(K):
        traces = []
        for pert in (False, True):
            restore(sim, snap)
            sim.reseed(1000 + k)
            g = torch.zeros(bins, N, dtype=torch.float32, device=s['V'].device)
            for n in range(bins * per_ms):
                I = I0
                if pert and n < round(P.b1_pulse_ms / P.dt_ms):
                    I = I0.clone()
                    I[targets] = pulse_pA
                sim.step(I)
                if n % per_ms == per_ms - 1:
                    g[n // per_ms] = s['g_exc_s']
            traces.append(g)
        base, pert_g = (traces[0].double(), traces[1].double())
        sum_d += pert_g - base
        sum_b += base
        sum_b2 += base * base
        log(f'seed pair {k + 1}/{K} done, wall {time.time() - t0:.0f} s')
    mean_d = sum_d / K
    var_b = (sum_b2 / K - (sum_b / K) ** 2).clamp(min=0) * K / max(K - 1, 1)
    sigma = torch.clamp(math.sqrt(2.0) * var_b.sqrt(), min=P.pci_sigma_floor_nS)
    S = (mean_d.abs() > P.pci_z * sigma / math.sqrt(K)).to(torch.uint8)
    col = S.sum(1).cpu().numpy()
    nz = np.nonzero(col)[0]
    out['reverberation_ms'] = float(nz[-1] + 1) if nz.size else 0.0
    out['n_significant_sources'] = int(S.any(0).sum())
    S_np = S.T.cpu().numpy()
    np.savez_compressed(f'{out_dir}/S_packed.npz', bits=np.packbits(S_np, axis=1), shape=np.array(S_np.shape))
    np.save(f'{out_dir}/S_col_sums.npy', col)
    out['wall_s_before_pci'] = round(time.time() - t0, 1)
    json.dump(out, open(f'{out_dir}/result.json', 'w'), indent=1)
    log(f"S saved ({out['n_significant_sources']} significant sources); computing PCI_LZ")
    out['PCI_LZ'] = lz.pci_lz(S_np)
    out['wall_s'] = round(time.time() - t0, 1)
    json.dump(out, open(f'{out_dir}/result.json', 'w'), indent=1)
    sim.slow.force_sleep = False
    return out
