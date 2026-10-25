import math
import os
import re
from pathlib import Path
import numpy as np
CACHE = Path(__file__).resolve().parents[1] / 'cache'
ANNOT = os.environ.get('FLYBRAIN_ANNOT', os.path.join(os.environ.get('FLYBRAIN_DATA', 'data/malecns_v1.0'), 'body-annotations-male-cns-v1.0-minconf-0.5.feather'))

def circmean_deg(a):
    r = np.radians(np.asarray(a, dtype=np.float64))
    return math.degrees(math.atan2(np.sin(r).mean(), np.cos(r).mean())) % 360

def epg_wedges(conn, annot_path=ANNOT):
    import pandas as pd
    import pyarrow.feather as pf
    X = np.load(CACHE / 'eb_sample.npy').astype(np.float64)
    c0 = X.mean(0)
    _, _, Vt = np.linalg.svd(X - c0, full_matrices=False)
    e1, e2 = (Vt[0], Vt[1])
    cen = pd.read_parquet(CACHE / 'eb_centroids.parquet')
    cen = cen[(cen.role == 'in') & (cen.roi == 'EB')].set_index('neuron_index')
    inst = pf.read_table(annot_path, columns=['bodyId', 'instance']).to_pandas().set_index('bodyId')['instance']
    idx, glom, ang = ([], [], [])
    for i in conn['set_EPG']:
        i = int(i)
        if i not in cen.index:
            continue
        m = re.search('_([LR])(\\d)$', str(inst.get(conn['body_id'][i], '')))
        if not m:
            continue
        v = cen.loc[i, ['x', 'y', 'z']].to_numpy(dtype=np.float64) - c0
        idx.append(i)
        glom.append(m.group(1) + m.group(2))
        ang.append(math.degrees(math.atan2(v @ e2, v @ e1)) % 360)
    glom, ang = (np.array(glom), np.array(ang))
    g16 = [f'{s}{n}' for s in 'LR' for n in range(1, 9)]
    centres = {g: circmean_deg(ang[glom == g]) for g in g16 if (glom == g).any()}
    order = sorted(centres, key=centres.get)
    wedge_of = {g: k for k, g in enumerate(order)}
    return {'idx': np.array(idx), 'glom': glom, 'angle_deg': ang, 'wedge': np.array([wedge_of.get(g, -1) for g in glom]), 'wedge_angle_deg': np.array([centres[g] for g in order]), 'wedge_glom': np.array(order)}

def heading_deg(activity, angle_deg):
    w = np.asarray(activity, dtype=np.float64)
    r = np.radians(angle_deg)
    return math.degrees(math.atan2((w * np.sin(r)).sum(), (w * np.cos(r)).sum())) % 360

def build_plastic(conn, net, roi_path=CACHE / 'roi_profile.parquet', ppm3_fb_frac=0.5):
    import pandas as pd
    N = net.N
    tn, tid = (conn['type_names'], conn['type_id'])
    types = np.array([tn[t] if t >= 0 else '' for t in tid])
    roi = pd.read_parquet(roi_path)
    d = roi[roi.neuron_index.isin(conn['set_DAN']) & (roi.role == 'out')]
    tot = d.groupby('neuron_index').n_syn.sum()
    fb = d[d.roi == 'FB'].groupby('neuron_index').n_syn.sum().reindex(tot.index, fill_value=0)
    ppm3 = np.array(sorted((int(i) for i, f in (fb / tot).items() if f > ppm3_fb_frac)))
    mask = lambda ix: np.isin(np.arange(N), ix)
    is_er, is_epg = (mask(conn['set_ER']), mask(conn['set_EPG']))
    is_fb = mask(np.setdiff1d(conn['set_FB'], ppm3))
    is_fc2 = np.char.startswith(types.astype(str), 'FC2')
    is_lptc = mask(conn['set_LPTC'])
    sc = conn['superclass_names'][conn['superclass_id']].astype(str)
    is_ol = np.char.startswith(sc, 'ol_') | (sc == 'visual_projection')
    pre = net.sp['pre'].cpu().numpy()
    post = (net.sp['idx_f'] % N).cpu().numpy()
    exc = (net.sp['idx_f'] // N).cpu().numpy() == 0
    pick = lambda m: np.nonzero(m)[0]
    er, fbq, eff = (pick(is_er[pre] & is_epg[post]), pick(is_fb[pre] & is_fc2[post]), pick(~is_ol[pre] & is_lptc[post]))
    from . import params as P
    return {'er_edge': er, 'er_pre': pre[er], 'er_post': post[er], 'fb_edge': fbq, 'fb_pre': pre[fbq], 'fb_post': post[fbq], 'eff_edge': eff, 'eff_pre': pre[eff], 'eff_post': post[eff], 'eff_E_rev': np.where(exc[eff], P.E_exc_mV, P.E_Cl_mV), 'ppm3_idx': ppm3}
