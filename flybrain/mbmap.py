import re
LOBES = {'y': ('g', 5), "a'": ("a'", 3), "B'": ("b'", 2), "b'": ("b'", 2), 'a': ('a', 3), 'B': ('b', 2), 'b': ('b', 2)}
COMPARTMENTS = [f'g{i}' for i in range(1, 6)] + [f"a'{i}" for i in range(1, 4)] + [f"b'{i}" for i in range(1, 3)] + [f'a{i}' for i in range(1, 4)] + [f'b{i}' for i in range(1, 3)]
INDEX = {c: i for i, c in enumerate(COMPARTMENTS)}
_TOKEN = re.compile("(y|a'|a|B'|b'|B|b)(\\d)")
_SAME_LOBE = re.compile('^[pmsd]+(\\d)')

def compartments(instance):
    m = re.search('\\(([^)]*)\\)', instance or '')
    if not m:
        return []
    body = m.group(1)
    body = re.split('[<>]', body)[0]
    out = []
    for t in _TOKEN.finditer(body):
        lobe, n = (LOBES[t.group(1)], int(t.group(2)))
        if 1 <= n <= lobe[1]:
            out.append(INDEX[f'{lobe[0]}{n}'])
        rest = body[t.end():]
        while True:
            c = _SAME_LOBE.match(rest)
            if not c:
                break
            n2 = int(c.group(1))
            if 1 <= n2 <= lobe[1]:
                out.append(INDEX[f'{lobe[0]}{n2}'])
            rest = rest[c.end():]
    return sorted(set(out))
OVERRIDE = {'PPL106': [INDEX['a3']]}

def compartments_of(type_name, instance):
    return OVERRIDE.get(type_name) or compartments(instance)
LOBE_OF_ROI = {'gL': 'g', 'aL': 'a', 'bL': 'b', "a'L": "a'", "b'L": "b'"}
LOBE_OF_COMP = [c[:-1] for c in COMPARTMENTS]
ANNOT = '/run/media/miltos/Ironwolf/Downloads/malecns_v1.0/body-annotations-male-cns-v1.0-minconf-0.5.feather'
KC_MBON_ROI = '/run/media/miltos/Linuxaddon/AI_Workspace/Flybrain/cache/kc_mbon_roi.parquet'
LTM_AV_TYPES, LTM_APP_TYPES = (('PPL105', 'PPL106'), ('PAM11',))
VETO_AV_TYPE, VETO_APP_TYPE = ('MBON01', 'MBON07')

def build(conn, net, annot_path=ANNOT, roi_path=KC_MBON_ROI):
    import numpy as np
    import pandas as pd
    import pyarrow.feather as pf
    import torch
    ann = pf.read_table(annot_path, columns=['bodyId', 'type', 'instance']).to_pandas().set_index('bodyId')
    body = conn['body_id']
    inst = ann['instance'].reindex(body).to_numpy()
    typ = ann['type'].reindex(body).to_numpy()
    comps_of = lambda i: compartments_of(typ[i] if isinstance(typ[i], str) else '', inst[i] if isinstance(inst[i], str) else '')
    dan_idx, dan_comp = ([], [])
    for i in np.concatenate([conn['set_PAM'], conn['set_PPL1']]):
        for k in comps_of(int(i)):
            dan_idx.append(int(i))
            dan_comp.append(k)
    aversive = [False] * len(COMPARTMENTS)
    for i in conn['set_PPL1']:
        for k in comps_of(int(i)):
            aversive[k] = True
    mbon_comps = {int(i): comps_of(int(i)) for i in conn['set_MBON']}
    roi = pd.read_parquet(roi_path)
    roi['lobe'] = roi['roi'].str.replace('\\((L|R)\\)$', '', regex=True).map(LOBE_OF_ROI)
    lobe_n = roi.dropna(subset=['lobe']).groupby(['kc_index', 'mbon_index', 'lobe']).n_syn.sum()
    N, n_kc = (net.N, int(conn['n_kc']))
    pre = net.sp['pre'].cpu().numpy()
    post = (net.sp['idx_f'] % N).cpu().numpy()
    is_mbon = np.zeros(N, dtype=bool)
    is_mbon[list(mbon_comps)] = True
    edges = np.nonzero((pre < n_kc) & is_mbon[post])[0]
    syn_kc, syn_comp, syn_edge, skipped = ([], [], [], 0)
    for e in edges:
        j, i = (int(pre[e]), int(post[e]))
        cs = mbon_comps[i]
        if not cs:
            skipped += 1
            continue
        if len(cs) > 1:

            def n_in(k):
                try:
                    return int(lobe_n.loc[j, i, LOBE_OF_COMP[k]])
                except KeyError:
                    return 0
            best = max(cs, key=lambda k: (n_in(k), -k))
            k = best
        else:
            k = cs[0]
        syn_kc.append(j)
        syn_comp.append(k)
        syn_edge.append(int(e))
    pick = lambda types: [int(i) for i in range(N) if isinstance(typ[i], str) and typ[i] in types]
    mb = {'dan_idx': dan_idx, 'dan_comp': dan_comp, 'comp_aversive': aversive, 'syn_kc': syn_kc, 'syn_comp': syn_comp, 'syn_edge': syn_edge, 'dpm_idx': [int(i) for i in conn['set_DPM']], 'n_kc': n_kc, 'ltm_av': pick(LTM_AV_TYPES), 'ltm_app': pick(LTM_APP_TYPES), 'veto_av': pick((VETO_AV_TYPE,)), 'veto_app': pick((VETO_APP_TYPE,)), 'n_kc_mbon_skipped': skipped}
    return mb
