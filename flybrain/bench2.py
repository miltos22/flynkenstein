import json
import math
import os
import re
import time
import numpy as np
import torch
from . import cxmap, params as P, protocols
from .world import World
ANNOT = os.environ.get('FLYBRAIN_ANNOT', os.path.join(os.environ.get('FLYBRAIN_DATA', 'data/malecns_v1.0'), 'body-annotations-male-cns-v1.0-minconf-0.5.feather'))

def pen_sides(conn, annot_path=ANNOT):
    import pyarrow.feather as pf
    inst = pf.read_table(annot_path, columns=['bodyId', 'instance']).to_pandas().set_index('bodyId')['instance'].reindex(conn['body_id'])
    L, R = ([], [])
    for i in np.concatenate([conn['set_PEN_a'], conn['set_PEN_b']]):
        m = re.search('_([LR])\\d+$', str(inst.iloc[int(i)]))
        (L if m and m.group(1) == 'L' else R if m else []).append(int(i))
    return (np.array(L), np.array(R))

def unwrap_deg(a):
    return np.degrees(np.unwrap(np.radians(np.asarray(a))))

def run(sim, conn, out_dir, landmark_deg=0.0, cue_s=3.0, dark_s=10.0, wrench_deg=-90.0, yaw_deg_s=200.0, goal_deg=60.0, log=print):
    dev = sim.net.device
    world = World(sim, conn)
    w = cxmap.epg_wedges(conn)
    epg = torch.as_tensor(w['idx'], dtype=torch.long, device=dev)
    ang = w['angle_deg']
    penL, penR = (torch.as_tensor(x, dtype=torch.long, device=dev) for x in pen_sides(conn))
    s = sim.net.s
    ema = torch.zeros(epg.numel(), device=dev, dtype=s['V'].dtype)
    a = math.exp(-P.dt_ms / 100.0)
    t0 = time.time()
    protocols.boot(sim, I_sens=None)
    rec, psi_body = ([], 0.0)
    steps = round((cue_s + dark_s) * 1000 / P.dt_ms)
    wrench_steps = round(abs(wrench_deg) / yaw_deg_s * 1000 / P.dt_ms)
    t_w = round(cue_s * 1000 / P.dt_ms)
    for n in range(steps):
        omega = 0.0
        if t_w <= n < t_w + wrench_steps:
            omega = math.copysign(yaw_deg_s, wrench_deg)
        psi_body = (psi_body + omega * P.dt_ms / 1000.0) % 360.0
        world.heading_deg = psi_body
        extra = torch.zeros(sim.net.N, device=dev, dtype=s['V'].dtype)
        cmd = 11.5 * max(-1.0, min(1.0, omega / 200.0))
        extra[penL] += cmd
        extra[penR] -= cmd
        I = world.inputs(landmark_deg=landmark_deg if n < t_w else None, extra=extra)
        sim.step(I)
        ema = a * ema + (1 - a) * s['spikes'][epg]
        if n % 50 == 49:
            act = ema.cpu().numpy()
            rec.append((n * P.dt_ms / 1000.0, psi_body, cxmap.heading_deg(act, ang) if act.sum() > 0 else float('nan'), float(act.sum()), float(act.max())))
        if n % 5000 == 0:
            log(f't={n * P.dt_ms / 1000:.1f} s wall {time.time() - t0:.0f} s')
    arr = np.array(rec)
    np.save(f'{out_dir}/trace.npy', arr)
    t, body, bump = (arr[:, 0], arr[:, 1], arr[:, 2])
    ok = ~np.isnan(bump)
    out = {'bump_present_fraction': float(ok.mean())}
    if ok.sum() > 10:
        cue = ok & (t < cue_s)
        off = math.degrees(math.atan2(np.mean(np.sin(np.radians(bump[cue] - body[cue]))), np.mean(np.cos(np.radians(bump[cue] - body[cue]))))) if cue.any() else 0.0
        dark = ok & (t >= cue_s)
        err = (bump - (body + off) + 180) % 360 - 180
        tt = t[dark]
        e = unwrap_deg(err[dark])
        out['drift_deg_per_s'] = float(np.polyfit(tt, e, 1)[0]) if dark.sum() > 2 else float('nan')
        out['blackout_error_deg'] = float(abs(err[ok][-1]))
        goal_bump = (goal_deg + off) % 360
        out['recovery_torque_sin'] = float(math.sin(math.radians(goal_bump - bump[ok][-1])))
        out['bump_offset_deg'] = off
    out['wall_s'] = round(time.time() - t0, 1)
    json.dump(out, open(f'{out_dir}/result.json', 'w'), indent=1)
    return out
