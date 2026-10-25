import math
import os
import zlib
import numpy as np
import torch
from . import params as P
ANNOT = os.environ.get('FLYBRAIN_ANNOT', os.path.join(os.environ.get('FLYBRAIN_DATA', 'data/malecns_v1.0'), 'body-annotations-male-cns-v1.0-minconf-0.5.feather'))

class World:

    def __init__(self, sim, conn, seed=0, annot_path=ANNOT):
        import pyarrow.feather as pf
        self.sim = sim
        dev, dt = (sim.net.device, sim.net.dtype)
        a = pf.read_table(annot_path, columns=['bodyId', 'type', 'somaSide', 'class']).to_pandas().set_index('bodyId').reindex(conn['body_id'])
        typ = a['type'].fillna('').to_numpy().astype(str)
        side = a['somaSide'].fillna('').to_numpy().astype(str)
        cls = a['class'].fillna('').to_numpy().astype(str)
        ix = lambda m: torch.as_tensor(np.nonzero(m)[0], dtype=torch.long, device=dev)
        self.dna02_L, self.dna02_R = (ix((typ == 'DNa02') & (side == 'L')), ix((typ == 'DNa02') & (side == 'R')))
        self.hs_L, self.hs_R = (ix(np.char.startswith(typ, 'HS') & (side == 'L')), ix(np.char.startswith(typ, 'HS') & (side == 'R')))
        er = np.nonzero(np.char.startswith(typ, 'ER2_') | (typ == 'ER4d'))[0]
        rng = np.random.default_rng(seed)
        self.er_idx = torch.as_tensor(er, dtype=torch.long, device=dev)
        self.er_pref = torch.as_tensor(rng.uniform(0, 2 * math.pi, er.size), dtype=dt, device=dev)
        self.alpn = np.nonzero(cls == 'ALPN')[0]
        self.ppl1 = torch.as_tensor(conn['set_PPL1'].astype(np.int64), device=dev)
        self.pam = torch.as_tensor(conn['set_PAM'].astype(np.int64), device=dev)
        self.heading_deg, self.yaw_deg_s, self.G_yaw = (0.0, 0.0, 10.0)
        self.rL = torch.zeros((), dtype=dt, device=dev)
        self.rR = torch.zeros((), dtype=dt, device=dev)
        self.N, self.dev, self.dt = (sim.net.N, dev, dt)
        self._odours = {}

    def odour(self, name):
        if name not in self._odours:
            rng = np.random.default_rng(zlib.crc32(name.encode()))
            pick = rng.choice(self.alpn, max(1, round(0.1 * self.alpn.size)), replace=False)
            self._odours[name] = torch.as_tensor(pick, dtype=torch.long, device=self.dev)
        return self._odours[name]

    def inputs(self, landmark_deg=None, odour=None, shock=False, sugar=False, extra=None):
        I = torch.zeros(self.N, dtype=self.dt, device=self.dev)
        if landmark_deg is not None:
            th = math.radians(landmark_deg - self.heading_deg)
            I[self.er_idx] += 15.0 * torch.exp(2.0 * (torch.cos(th - self.er_pref) - 1.0))
        flow = 0.05 * self.yaw_deg_s
        I[self.hs_L] += flow
        I[self.hs_R] -= flow
        if odour is not None:
            I[self.odour(odour)] += 12.0
        us = self.sim.slow.I_US_ext
        us.zero_()
        if shock:
            us[self.ppl1] = 25.0
        if sugar:
            us[self.pam] = 25.0
        if extra is not None:
            I = I + extra
        return I

    def after_step(self):
        s = self.sim.net.s
        a = math.exp(-P.dt_ms / 50.0)
        rate = lambda ix: s['spikes'][ix].mean() * 1000.0 / P.dt_ms if ix.numel() else torch.zeros((), dtype=self.dt, device=self.dev)
        self.rL = a * self.rL + (1 - a) * rate(self.dna02_L)
        self.rR = a * self.rR + (1 - a) * rate(self.dna02_R)
        self.yaw_deg_s = self.G_yaw * float(self.rR - self.rL)
        self.heading_deg = (self.heading_deg + self.yaw_deg_s * P.dt_ms / 1000.0) % 360.0
