import math
import torch
from . import params as P
CLS = {k: torch.tensor(v, dtype=torch.float64) for k, v in P.CLASS.items()}
TAU_SYN = torch.tensor([P.tau_syn_excf_ms, P.tau_syn_excs_ms, P.tau_syn_inhf_ms, P.tau_syn_inhs_ms], dtype=torch.float64)
BETA_REC = torch.tensor([P.beta_rec[k] for k in range(4)], dtype=torch.float64)

def m_inf_NaP(V):
    return 1.0 / (1.0 + torch.exp(-(V - P.V_half_NaP_mV) / P.k_NaP_mV))

def m_inf_CaT(V):
    return 1.0 / (1.0 + torch.exp(-(V + P.m_inf_CaT_half_mV) / P.m_inf_CaT_k_mV))

def r_inf(V):
    return 1.0 / (1.0 + torch.exp((V + P.r_inf_half_mV) / P.r_inf_k_mV))

def tau_r(V):
    return P.tau_r_base_ms + P.tau_r_amp_ms / (1.0 + torch.exp((V + P.tau_r_half_mV) / P.tau_r_k_mV))

def B_Mg(V):
    return 1.0 / (1.0 + P.Mg_o_mM / P.Mg_half_mM * torch.exp(-P.Mg_slope_per_mV * V))

def phi_release(V, V_rel=None, k_rel=None):
    V_rel = P.V_rel_mV if V_rel is None else V_rel
    k_rel = P.k_rel_mV if k_rel is None else k_rel
    return P.phi_max_per_ms / (1.0 + torch.exp(-(V - V_rel) / k_rel))

def kappa_zoh(dt, every=5):
    return (1 - torch.exp(-every * dt / TAU_SYN)) / (1 - torch.exp(-dt / TAU_SYN))

def init_state(cls, dtype=torch.float64, device='cpu', V0=None):
    cls = cls.to(device)
    c = {k: v.to(device=device, dtype=dtype)[cls] for k, v in CLS.items()}
    n = cls.numel()
    z = lambda: torch.zeros(n, dtype=dtype, device=device)
    s = {'cls': cls, 'is_graded': c['Delta_T_mV'] == 0, 'V': c['E_L_base_mV'].clone() if V0 is None else V0.to(device=device, dtype=dtype), 'g_adapt': z(), 'c_ca': z(), 's_bar': z(), 'r_bar': z(), 'x_stp': torch.ones(n, dtype=dtype, device=device), 'spikes': z(), 'refrac': torch.zeros(n, dtype=torch.int32, device=device), 'g_exc_f': z(), 'g_exc_s': z(), 'g_inh_f': z(), 'g_inh_s': z(), 'G_gap_sum': z(), 'I_gap_num': z(), 'E_L': c['E_L_base_mV'].clone(), 'E_L_mod': c['E_L_base_mV'].clone(), 'g_L': c['g_L0_nS'].clone(), 'g_NaP': c['g_NaP0_nS'].clone(), 'g_h': c['g_h0_nS'].clone(), 'g_KCa': c['g_KCa0_nS'].clone(), 'U0': c['U0'].clone(), 'a_adapt': c['a0_nS'].clone(), 'dg_adapt': c['dg_adapt0_nS'].clone()}
    s['u_stp'] = s['U0'].clone()
    s['V_rel'] = torch.full((n,), P.V_rel_mV, dtype=dtype, device=device)
    s['k_rel'] = torch.full((n,), P.k_rel_mV, dtype=dtype, device=device)
    s['r_ih'] = r_inf(s['V'])
    s['c'] = c
    return s

def step_node(s, I_ext, dt=P.dt_ms, xi=None):
    c, g = (s['c'], s['is_graded'].to(s['V'].dtype))
    V = s['V']
    phi = phi_release(V, s['V_rel'], s['k_rel']) * g
    x, u, U0 = (s['x_stp'], s['u_stp'], s['U0'])
    x_m = 1.0 - (1.0 - x) * torch.exp(-dt / c['tau_rec_ms']) - g * u * x * phi * dt
    u_m = U0 + (u - U0) * torch.exp(-dt / c['tau_fac_ms']) + g * U0 * (1.0 - u) * phi * dt
    spk_prev = s['spikes'] * (1 - g)
    q = torch.ones_like(V) if xi is None else xi
    tilde_s = torch.where(s['is_graded'], u_m * x_m * phi * dt, q * u_m * x_m * spk_prev)
    s['x_stp'] = torch.where(s['is_graded'], x_m, x_m * (1.0 - u_m * spk_prev))
    s['u_stp'] = torch.where(s['is_graded'], u_m, u_m + U0 * (1.0 - u_m) * spk_prev)
    s['s_bar'] = math.exp(-dt / P.tau_s_ms) * s['s_bar'] + tilde_s / P.tau_s_ms
    a_r = math.exp(-dt / P.tau_r_bar_ms)
    s['r_bar'] = a_r * s['r_bar'] + 1000.0 / P.tau_r_bar_ms * (spk_prev + phi * dt)
    V_T, dT = (c['V_T_mV'], c['Delta_T_mV'])
    V_t = torch.minimum(V, V_T + P.upstroke_factor * dT)
    I_exp = (1 - g) * s['g_L'] * dT * torch.exp((V_t - V_T) / torch.clamp(dT, min=P.delta_T_floor_mV))
    B = B_Mg(V)
    mN, mC, r = (m_inf_NaP(V), m_inf_CaT(V), s['r_ih'])
    kca = s['c_ca'] ** 2 / (s['c_ca'] ** 2 + P.K_Ca_uM ** 2)
    gCaT = c['g_CaT0_nS']
    G_eff = s['g_L'] + s['g_NaP'] * mN + s['g_h'] * r + gCaT * mC * r + s['g_KCa'] * kca + s['g_adapt'] + s['g_exc_f'] + s['g_exc_s'] * B + s['g_inh_f'] + s['g_inh_s'] + s['G_gap_sum']
    num = s['g_L'] * s['E_L_mod'] + (s['g_exc_f'] + s['g_exc_s'] * B) * P.E_exc_mV + (s['g_NaP'] * mN + gCaT * mC * r) * P.E_Na_mV + s['g_h'] * r * P.E_h_mV + (s['g_KCa'] * kca + s['g_adapt'] + s['g_inh_s']) * P.E_K_mV + s['g_inh_f'] * P.E_Cl_mV + s['I_gap_num'] + I_exp + I_ext
    a_V = torch.exp(-dt * G_eff / c['C_m_pF'])
    V_new = (1.0 - a_V) * (num / G_eff) + a_V * V
    refr = (s['refrac'] > 0) & ~s['is_graded']
    fire = ~refr & (V_new >= P.V_peak_mV) & ~s['is_graded']
    V_reset = torch.nan_to_num(c['V_reset_mV'], nan=0.0)
    s['V'] = torch.where(refr | fire, V_reset, V_new)
    s['spikes'] = fire.to(V.dtype)
    s['refrac'] = torch.where(refr, s['refrac'] - 1, torch.where(fire, torch.full_like(s['refrac'], P.refrac_steps), s['refrac']))
    Vn = s['V']
    a_rr = torch.exp(-dt / tau_r(Vn))
    s['r_ih'] = a_rr * r + (1.0 - a_rr) * r_inf(Vn)
    a_ad = torch.exp(-dt / c['tau_adapt_ms'])
    g_inf = s['a_adapt'] * torch.relu(torch.minimum(Vn, torch.tensor(P.V_peak_mV, dtype=Vn.dtype)) - s['E_L_mod']) / (P.V_peak_mV - P.E_K_mV)
    s['g_adapt'] = a_ad * s['g_adapt'] + (1.0 - a_ad) * g_inf + s['dg_adapt'] * s['spikes']
    I_CaT_in = gCaT * m_inf_CaT(Vn) * r * torch.relu(P.E_Ca_mV - Vn)
    s['c_ca'] = torch.exp(-dt / c['tau_Ca_ms']) * s['c_ca'] + dt * (P.alpha_Ca_uM_per_pA_ms * I_CaT_in + P.alpha_Ca_grad_uM * phi) + (1 - g) * P.dc_spike_uM * s['spikes']
    return tilde_s
RING = 64

class Network:

    def __init__(self, conn, device='cuda', dtype=torch.float32, event=True, compile=True, fast=False):
        self.device, self.dtype, self.event = (device, dtype, event)
        self.node = torch.compile(step_node, dynamic=False) if compile and str(device).startswith('cuda') else step_node
        cls = torch.as_tensor(conn['cls'], dtype=torch.long)
        self.N = cls.numel()
        self.s = init_state(cls, dtype=dtype, device=device)
        rp = torch.as_tensor(conn['row_ptr'])
        post = torch.repeat_interleave(torch.arange(self.N), rp[1:] - rp[:-1]).to(device)
        rec = torch.as_tensor(conn['rec'].astype('int64')).to(device)
        W = torch.as_tensor(conn['W_struct']).to(device=device, dtype=dtype)
        beta = BETA_REC.to(device=device, dtype=dtype)[rec]
        exc = rec <= 1
        ch_f = torch.where(exc, 0, 2)
        self.idx_f = post + self.N * ch_f
        self.idx_s = post + self.N * (ch_f + 1)
        self.w_f = W * beta.gather(1, ch_f[:, None]).squeeze(1)
        self.w_s = W * beta.gather(1, (ch_f + 1)[:, None]).squeeze(1)
        self.is_exc_edge = exc
        self.pre = torch.as_tensor(conn['col_idx'].astype('int64')).to(device)
        self.delay = torch.as_tensor(conn['delay'].astype('int64')).to(device)
        self.ring = torch.zeros(RING, self.N, dtype=dtype, device=device)
        self.n = 0
        self.decay = torch.exp(-P.dt_ms / TAU_SYN).to(device=device, dtype=dtype)
        if 'V_rel_mV' in conn:
            self.s['V_rel'] = torch.as_tensor(conn['V_rel_mV']).to(device=device, dtype=dtype)
            self.s['k_rel'] = torch.as_tensor(conn['k_rel_mV']).to(device=device, dtype=dtype)
        self.gap = None
        if 'gap_pre' in conn and len(conn['gap_pre']):
            t = lambda k, d: torch.as_tensor(conn[k]).to(device=device, dtype=d)
            self.gap = {'pre': t('gap_pre', torch.long), 'post': t('gap_post', torch.long), 'g': t('gap_g_nS', dtype), 'alpha': t('gap_alpha', dtype)}
        if event:
            self._build_event_path()
        self.fast = None
        if fast:
            assert event and dtype == torch.float32 and str(device).startswith('cuda')
            from .kernels import FastTransmission
            self.fast = FastTransmission(self)
            self.is_graded_u8 = self.s['is_graded'].to(torch.uint8)

    def _gap_update(self):
        if self.gap is None:
            return
        s, gp = (self.s, self.gap)
        Vi, Vj = (s['V'][gp['post']], s['V'][gp['pre']])
        g = gp['g'] * (gp['alpha'] + (1.0 - gp['alpha']) / (1.0 + torch.exp(-P.beta_rect_per_mV * (Vi - Vj))))
        G = torch.zeros_like(s['V'])
        G.index_add_(0, gp['post'], g)
        In = torch.zeros_like(s['V'])
        In.index_add_(0, gp['post'], g * Vj)
        s['G_gap_sum'], s['I_gap_num'] = (G, In)

    def _build_event_path(self):
        g = self.s['is_graded'][self.pre]
        keys = ('pre', 'delay', 'idx_f', 'idx_s', 'w_f', 'w_s')
        self.gd = {k: getattr(self, k)[g] for k in keys}
        sp = ~g
        order = torch.argsort(self.pre[sp], stable=True)
        self.sp = {k: getattr(self, k)[sp][order] for k in keys}
        cnt = torch.bincount(self.sp['pre'], minlength=self.N)
        self.out_ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=self.device), torch.cumsum(cnt, 0)])
        for k in keys + ('is_exc_edge',):
            delattr(self, k)
        self.buf = torch.zeros(RING, 4 * self.N, dtype=self.dtype, device=self.device)

    def _step_event(self, I_ext, gain_exc, gain_inh, xi):
        s, N, slot = (self.s, self.N, self.n % RING)
        g_in = self.buf[slot].clone()
        self.buf[slot].zero_()
        gd = self.gd
        if self.fast is not None:
            self.fast.graded(g_in, self.ring, self.n)
        else:
            src = self.ring[(self.n - gd['delay']) % RING, gd['pre']]
            g_in.index_add_(0, gd['idx_f'], gd['w_f'] * src)
            g_in.index_add_(0, gd['idx_s'], gd['w_s'] * src)
        g_in = g_in.view(4, N)
        if gain_exc != 1.0 or gain_inh != 1.0:
            g_in = g_in * torch.tensor([gain_exc, gain_exc, gain_inh, gain_inh], dtype=self.dtype, device=self.device)[:, None]
        for k, name in enumerate(('g_exc_f', 'g_exc_s', 'g_inh_f', 'g_inh_s')):
            s[name] = self.decay[k] * s[name] + g_in[k]
        self._gap_update()
        tilde = self.node(s, I_ext, xi=xi)
        self.ring[slot] = tilde
        if self.fast is not None:
            self.fast.scatter(tilde, self.is_graded_u8, self.n)
            self.n += 1
            return tilde
        act = torch.nonzero((tilde != 0) & ~s['is_graded']).squeeze(1)
        if act.numel():
            st = self.out_ptr[act]
            cnt = self.out_ptr[act + 1] - st
            rep = torch.repeat_interleave(torch.arange(act.numel(), device=self.device), cnt)
            e = st[rep] + torch.arange(rep.numel(), device=self.device) - (torch.cumsum(cnt, 0) - cnt)[rep]
            r = tilde[act][rep]
            sp, flat = (self.sp, self.buf.view(-1))
            base = (self.n + sp['delay'][e]) % RING * (4 * N)
            flat.index_add_(0, base + sp['idx_f'][e], sp['w_f'][e] * r)
            flat.index_add_(0, base + sp['idx_s'][e], sp['w_s'][e] * r)
        self.n += 1
        return tilde

    def step(self, I_ext, gain_exc=1.0, gain_inh=1.0, xi=None):
        if self.event:
            return self._step_event(I_ext, gain_exc, gain_inh, xi)
        s = self.s
        src = self.ring[(self.n - self.delay) % RING, self.pre]
        g_in = torch.zeros(4 * self.N, dtype=self.dtype, device=self.device)
        if gain_exc != 1.0 or gain_inh != 1.0:
            src = src * torch.where(self.is_exc_edge, gain_exc, gain_inh).to(self.dtype)
        g_in.index_add_(0, self.idx_f, self.w_f * src)
        g_in.index_add_(0, self.idx_s, self.w_s * src)
        g_in = g_in.view(4, self.N)
        for k, name in enumerate(('g_exc_f', 'g_exc_s', 'g_inh_f', 'g_inh_s')):
            s[name] = self.decay[k] * s[name] + g_in[k]
        self._gap_update()
        tilde = self.node(s, I_ext, xi=xi)
        self.ring[self.n % RING] = tilde
        self.n += 1
        return tilde

def dopamine_step(D, tilde_s_dans, n_dan, dt=P.dt_ms):
    return math.exp(-dt / P.tau_DA_ms) * D + 1000.0 * P.alpha_DA_s / (P.tau_DA_ms * n_dan) * tilde_s_dans

def tyson_step(M, Pc, dt_s, L=0.0):
    dM = (P.v_m / (1.0 + (Pc / P.P_0) ** 4) - M) / P.tau_circ_s
    dP = (P.v_p * M - P.k_p1 * Pc / (P.K_p1 + Pc + P.k_p2 * Pc ** 2) - (P.k_p3 + P.eta_light * L) * Pc) / P.tau_circ_s
    return (M + dt_s * dM, Pc + dt_s * dP)

class SlowPass:

    def __init__(self, net, conn):
        dev, dt = (net.device, net.dtype)
        self.net = net
        ix = lambda name: torch.as_tensor(conn['set_' + name].astype('int64'), device=dev)
        self.dFB, self.ER5, self.motor, self.clock = (ix('dFB'), ix('ER5'), ix('motor'), ix('clock'))
        self.PAM, self.PPL1, self.DAN, self.IPC = (ix('PAM'), ix('PPL1'), ix('DAN'), ix('IPC'))
        self.n_kc = int(conn['n_kc'])
        N = net.N
        c = net.s['c']
        self.E_L_max = c['E_L_max_mV']
        self.E_L_min = P.E_L_box_min_mV
        self.span = self.E_L_max - self.E_L_min
        self.is_spk = ~net.s['is_graded']
        self.is_dFB = torch.zeros(N, dtype=torch.bool, device=dev)
        self.is_dFB[self.dFB] = True
        self.is_clock = torch.zeros(N, dtype=torch.bool, device=dev)
        self.is_clock[self.clock] = True
        self.is_PAM = torch.zeros(N, dtype=torch.bool, device=dev)
        self.is_PAM[self.PAM] = True
        self.is_PPL1 = torch.zeros(N, dtype=torch.bool, device=dev)
        self.is_PPL1[self.PPL1] = True
        self.is_DAN = torch.zeros(N, dtype=torch.bool, device=dev)
        self.is_DAN[self.DAN] = True
        z = lambda: torch.zeros(N, dtype=dt, device=dev)
        self.s_cAMP, self.s_Gq, self.H_redox = (z(), z(), z())
        self.T_tag_av, self.T_tag_app = (z(), z())
        self.I_endo, self.I_US_ext, self.I_replay = (z(), z(), z())
        self.h = torch.full((5,), P.h_star, dtype=torch.float64)
        self.M_circ, self.P_circ = (0.5, 0.5)
        self.S_sleep, self.A_sws, self.A_prev, self.n_up, self.p_up = (0.0, 0.0, 0.0, 0, 0)
        self.light = 0.0

    def run(self, dt_mod_ms=P.dt_mod_ms):
        s, dts = (self.net.s, 0.001 * dt_mod_ms)
        r_bar, s_bar = (s['r_bar'], s['s_bar'])
        intake = torch.zeros(5, dtype=torch.float64)
        intake[4] = P.gamma_rest_rec_per_s * self.S_sleep
        r_motor = float(r_bar[self.motor].mean()) if self.motor.numel() else 0.0
        dh = -P.gamma_met_per_s * (1.0 + P.beta_loco_per_Hz * r_motor) * self.h + intake * (1.0 - self.h)
        self.h = self.h + dts * dh
        D_need = torch.relu(P.h_star - self.h)
        V_hed = float((P.w_hed * D_need * dh).sum())
        S_sat = max(float(self.h[0]) - P.h_star, 0.0)
        I_US = self.I_US_ext + (self.is_PAM * P.G_hed_pA_s * max(V_hed, 0.0) + self.is_PPL1 * P.G_hed_pA_s * max(-V_hed, 0.0)).to(s['V'].dtype)
        I_endo = torch.zeros_like(self.I_endo)
        I_endo[self.IPC] = P.G_sat_pA * S_sat
        self.I_endo = torch.where(self.is_DAN, I_US + self.I_replay, I_endo)
        r_dFB = float(r_bar[self.dFB].mean())
        if getattr(self, 'force_sleep', False):
            self.S_sleep = 1.0
        elif r_dFB >= P.theta_sleep_on_Hz:
            self.S_sleep = 1.0
        elif r_dFB <= P.theta_sleep_off_Hz:
            self.S_sleep = 0.0
        self.A_sws = self.S_sleep * min(max(float(s_bar[self.ER5].mean()) / P.s_SWS_max_per_ms, 0.0), 1.0)
        if self.S_sleep == 1.0 and self.A_prev >= P.A_SWS_threshold and (self.A_sws < P.A_SWS_threshold):
            self.n_up += 1
        self.p_up = self.n_up % 2
        self.A_prev = self.A_sws
        S = self.S_sleep
        c = s['c']
        rt = c['r_target_Hz']
        dE = (P.eta_homeo_mV_per_Hz * (1.0 - S) * (torch.relu(rt - r_bar) * (self.E_L_max - s['E_L']) / self.span - torch.relu(r_bar - rt) * (s['E_L'] - self.E_L_min) / self.span) - P.gamma_sleep * S * (s['E_L'] - c['E_L_base_mV'])) / P.tau_homeo_s
        mask = getattr(self, 'homeo_mask', None)
        s['E_L'] = s['E_L'] + dts * getattr(self, 'homeo_speed', 1.0) * (dE if mask is None else dE * mask)
        self.M_circ, self.P_circ = tyson_step(self.M_circ, self.P_circ, dts, self.light)
        r_glob = float(r_bar[self.is_spk].mean())
        dH = (P.alpha_wake_per_Hz * (1.0 - S) * (r_glob + P.gamma_shunt_Hz * self.s_Gq) * (1.0 - self.H_redox) - P.beta_clear * S * self.H_redox) / P.tau_redox_s
        self.H_redox = torch.where(self.is_dFB, self.H_redox + dts * dH, self.H_redox)
        circ = P.delta_V_circ_mV * (self.P_circ / (self.P_circ + P.P_0) - 0.5)
        bias = self.is_clock * circ
        kc = torch.arange(self.net.N, device=s['V'].device) < self.n_kc
        tag = self.A_sws * (P.delta_V_up_KC_mV + P.delta_V_tag_mV * ((1 - self.p_up) * self.T_tag_av + self.p_up * self.T_tag_app))
        bias = bias + kc * tag
        s['E_L_mod'] = torch.clamp(s['E_L'] + P.delta_E_max_mV * torch.tanh(P.gamma_1 * self.s_cAMP + P.gamma_2 * self.s_Gq) + bias, self.E_L_min, P.E_L_mod_max_mV)
        s['g_h'] = c['g_h0_nS'] * torch.relu(1.0 + P.cAMP_gain_h * self.s_cAMP)
        s['g_NaP'] = c['g_NaP0_nS'] * torch.relu(1.0 + P.cAMP_gain_NaP * self.s_cAMP)
        s['g_KCa'] = c['g_KCa0_nS'] * torch.exp(-P.lambda_KCa * self.s_Gq)
        s['U0'] = torch.clamp(c['U0'] * (1.0 + P.kappa_rel * self.s_cAMP), P.U0_clip_min, P.U0_clip_max)
        s['g_L'] = c['g_L0_nS'] * (1.0 + self.is_dFB * P.gamma_Sandman * self.s_Gq)
        hk = torch.exp(-self.is_dFB.to(self.H_redox.dtype) * P.kappa_Hk * self.H_redox)
        s['a_adapt'] = c['a0_nS'] * hk
        s['dg_adapt'] = c['dg_adapt0_nS'] * hk

class Sim:

    def __init__(self, conn, device='cuda', dtype=torch.float32, seed=0, quantal=True, mb=None, cx=None, fast=None):
        prof = str(conn['profile']) if 'profile' in conn else 'v3.2'
        if prof != 'v3.2':
            from . import profiles
            if profiles.ACTIVE != prof:
                raise RuntimeError(f'connectome built with profile {prof}; call flybrain.profiles.apply({prof!r}) before importing data / building Sim')
        if fast is None:
            fast = str(device).startswith('cuda') and dtype == torch.float32
        self.net = Network(conn, device=device, dtype=dtype, fast=fast)
        self.slow = SlowPass(self.net, conn)
        self.diff = Diffusion(conn, device=device, dtype=dtype)
        if mb == 'build':
            from . import mbmap
            mb = mbmap.build(conn, self.net)
        self.mb = MBPlasticity(self.net, mb) if mb is not None else None
        if cx == 'build':
            from . import cxmap
            cx = cxmap.build_plastic(conn, self.net)
        self.cx = CXPlasticity(self.net, cx) if cx is not None else None
        comp = (lambda f: torch.compile(f, dynamic=False)) if fast else lambda f: f
        self._mb_clock1 = comp(self.mb.clock1) if self.mb is not None else None
        self._mb_replay = comp(self.mb.replay) if self.mb is not None else None
        self._cx_clock1 = comp(self.cx.clock1) if self.cx is not None else None
        self.quantal = quantal
        self.reseed(seed)

    def reseed(self, seed):
        self.gen = torch.Generator(device=self.net.device).manual_seed(int(seed))

    def step(self, I_sens, gain_exc=1.0, gain_inh=1.0, xi=None):
        if xi is None and self.quantal:
            z = torch.randn(self.net.N, generator=self.gen, device=self.net.device, dtype=self.net.dtype)
            xi = torch.clamp(1.0 + P.sigma_quantal * z, P.xi_min, P.xi_max)
        I = I_sens + self.slow.I_endo
        if self.mb is not None:
            I = I + self._mb_replay(self.slow.S_sleep, self.slow.p_up)
        tilde = self.net.step(I, gain_exc, gain_inh, xi)
        sub = (self.net.n - 1) % P.cudagraph_steps
        if self.mb is not None:
            self._mb_clock1(tilde)
            if sub % 50 == 49:
                self.mb.clock2(self.slow.S_sleep, self.slow.A_sws)
        if self.cx is not None:
            self._cx_clock1(tilde)
            if sub % 50 == 49:
                self.cx.clock2(self.slow.S_sleep)
        if sub in (P.pingpong_substep_1, P.pingpong_substep_2):
            self.diff.step(self.net.s['s_bar'])
        if sub == P.pingpong_substep_2:
            if self.mb is not None:
                self.slow.T_tag_av[:self.mb.n_kc] = self.mb.T_av
                self.slow.T_tag_app[:self.mb.n_kc] = self.mb.T_app
            self.slow.run()
        return tilde
AMINE_OF_NT = {4: 0, 5: 1, 7: 2, 6: 3}

class Diffusion:
    N_CH = 12

    def __init__(self, conn, device='cuda', dtype=torch.float32):
        self.shape = tuple((int(x) for x in conn['grid_shape']))
        self.h = torch.as_tensor(conn['grid_spacing_um'], dtype=torch.float64).tolist()
        self.device, self.dtype = (device, dtype)
        self.C = torch.zeros((self.N_CH,) + self.shape, dtype=dtype, device=device)
        D = [P.D_amine_um2_per_s] * 4 + [P.D_pep_um2_per_s] * 8
        Vm = [P.V_max_amine_uM_per_s] * 4 + [P.V_max_pep_uM_per_s] * 8
        self.D = torch.tensor(D, dtype=dtype, device=device)[:, None, None, None]
        self.Vmax = torch.tensor(Vm, dtype=dtype, device=device)[:, None, None, None]
        nt = torch.as_tensor(conn['nt'].astype('int64'))
        sec = [(i, AMINE_OF_NT[int(c)]) for i, c in enumerate(nt.tolist()) if int(c) in AMINE_OF_NT]
        self.sec_neuron = torch.tensor([a for a, _ in sec], dtype=torch.long, device=device)
        self.sec_flat = torch.tensor([b for _, b in sec], dtype=torch.long, device=device) * int(torch.tensor(self.shape).prod()) + torch.as_tensor(conn['voxel'].astype('int64'), device=device)[self.sec_neuron]
        self.voxel = torch.as_tensor(conn['voxel'].astype('int64'), device=device)

    def laplacian(self, C):
        out = torch.zeros_like(C)
        for ax, h in zip((1, 2, 3), self.h):
            Cp = torch.cat([C.narrow(ax, 1, C.shape[ax] - 1), C.narrow(ax, C.shape[ax] - 1, 1)], ax)
            Cm = torch.cat([C.narrow(ax, 0, 1), C.narrow(ax, 0, C.shape[ax] - 1)], ax)
            out += (Cp - 2 * C + Cm) / (h * h)
        return out

    def step(self, s_bar, dt_ms=P.dt_diff_ms):
        dts = 0.001 * dt_ms
        src = torch.zeros(self.C.numel(), dtype=self.dtype, device=self.device)
        src.index_add_(0, self.sec_flat, 1000.0 * P.alpha_amine_uM * s_bar[self.sec_neuron])
        src = src.view_as(self.C)
        C = self.C
        num = C + dts * (self.D * self.laplacian(C) + src)
        self.C = torch.clamp(num / (1.0 + dts * self.Vmax / (P.K_M_uM + C)), min=0.0)
        return self.C

    def at_neurons(self):
        return self.C.view(self.N_CH, -1)[:, self.voxel]

class MBPlasticity:

    def __init__(self, net, mb):
        dev, dt = (net.device, net.dtype)
        self.net = net
        t = lambda x, d=torch.long: torch.as_tensor(x, dtype=d, device=dev)
        self.dan_idx, self.dan_comp = (t(mb['dan_idx']), t(mb['dan_comp']))
        self.K = int(len(mb['comp_aversive']))
        self.n_dan = torch.bincount(self.dan_comp, minlength=self.K).clamp(min=1).to(dt)
        self.aversive = t(mb['comp_aversive'], torch.bool)
        av = [k for k, v in enumerate(mb['comp_aversive']) if v]
        self.av_idx, self.app_idx = (t(av), t([k for k in range(self.K) if k not in av]))
        self.syn_kc, self.syn_comp, self.syn_edge = (t(mb['syn_kc']), t(mb['syn_comp']), t(mb['syn_edge']))
        self.dpm = t(mb['dpm_idx'])
        self.n_kc = int(mb['n_kc'])
        z = lambda n: torch.zeros(n, dtype=dt, device=dev)
        self.D, self.e_bwd = (z(self.K), z(self.K))
        self.e_fwd, self.T_av, self.T_app = (z(self.n_kc), z(self.n_kc), z(self.n_kc))
        n = self.syn_kc.numel()
        self.w_fast, self.w_interm, self.w_slow = (z(n), z(n), z(n))
        sp = net.sp
        self.base_f, self.base_s = (sp['w_f'][self.syn_edge].clone(), sp['w_s'][self.syn_edge].clone())
        self.ltm_av, self.ltm_app = (t(mb.get('ltm_av', [])), t(mb.get('ltm_app', [])))
        self.veto_av, self.veto_app = (t(mb.get('veto_av', [])), t(mb.get('veto_app', [])))
        self.I_replay = torch.zeros(net.N, dtype=dt, device=dev)

    def replay(self, S_sleep, p_up):
        sb = self.net.s['s_bar']
        kc_sum = sb[:self.n_kc].sum()
        mean = lambda ix: sb[ix].mean() if ix.numel() else torch.zeros((), dtype=sb.dtype, device=sb.device)
        I_av = S_sleep * (1 - p_up) * torch.relu(P.G_up_pA_ms * kc_sum - P.W_veto_pA_ms * mean(self.veto_av))
        I_app = S_sleep * p_up * torch.relu(P.G_up_pA_ms * kc_sum - P.W_veto_pA_ms * mean(self.veto_app))
        self.I_replay.zero_()
        self.I_replay[self.ltm_av] = I_av
        self.I_replay[self.ltm_app] = I_app
        return self.I_replay

    @property
    def D_hat(self):
        return torch.relu(self.D - P.theta_dDAT)

    def clock1(self, tilde, dt=P.dt_ms):
        s, dts = (self.net.s, 0.001 * dt)
        rel = torch.zeros(self.K, dtype=self.D.dtype, device=self.D.device)
        rel.index_add_(0, self.dan_comp, tilde[self.dan_idx])
        self.D = math.exp(-dt / P.tau_DA_ms) * self.D + 1000.0 * P.alpha_DA_s / (P.tau_DA_ms * self.n_dan) * rel
        Dh = self.D_hat
        c_kc = s['c_ca'][:self.n_kc]
        self.e_fwd = self.e_fwd + dts * (-self.e_fwd / P.tau_fwd_s + P.alpha_fwd_per_uM_per_s * c_kc * (1.0 - self.e_fwd))
        self.e_bwd = self.e_bwd + dts * (-self.e_bwd / P.tau_bwd_s + P.alpha_bwd_per_s * Dh * (1.0 - self.e_bwd))
        zero = torch.zeros((), dtype=Dh.dtype, device=Dh.device)
        max_av = Dh[self.av_idx].max() if self.av_idx.numel() else zero
        max_app = Dh[self.app_idx].max() if self.app_idx.numel() else zero
        self.T_av = self.T_av + dts * (-self.T_av / P.tau_tag_s + P.alpha_tag_per_s * self.e_fwd * max_av * (1.0 - self.T_av))
        self.T_app = self.T_app + dts * (-self.T_app / P.tau_tag_s + P.alpha_tag_per_s * self.e_fwd * max_app * (1.0 - self.T_app))

    def clock2(self, S_sleep, A_sws, dt_ms=P.dt_plast_ms):
        s, dts = (self.net.s, 0.001 * dt_ms)
        j, k = (self.syn_kc, self.syn_comp)
        Dh, ef, c = (self.D_hat[k], self.e_fwd[j], s['c_ca'][j])
        wf, wi, ws = (self.w_fast, self.w_interm, self.w_slow)
        Phi = -P.eta_LTD_per_s * ef * Dh + P.eta_LTP_bwd_per_s * c * self.e_bwd[k] + P.eta_LTP_solo_per_s * Dh * torch.exp(-ef / P.epsilon_0) * torch.relu(-(wf + wi))
        g_dpm = min(max(float(s['r_bar'][self.dpm].mean()) / P.r_DPM_max_Hz, 0.0), 1.0) if self.dpm.numel() else 0.0
        J = P.kappa_1to2_per_s * g_dpm * (torch.relu(wf) * (1.0 - wi) - torch.relu(-wf) * (1.0 + wi))
        tag = torch.where(self.aversive[k], self.T_av[j], self.T_app[j])
        dwf = (1.0 - S_sleep) * Phi - J - wf / P.tau_fast_s - P.gamma_SHY_fast_per_s * S_sleep * A_sws * wf
        dwi = J - wi / P.tau_interm_s - P.gamma_SHY_interm_per_s * S_sleep * A_sws * wi * (1.0 - ws.abs())
        dws = (1.0 - S_sleep) * (-P.kappa_2to3_dep_per_s * torch.relu(-wi - P.theta_spaced) * (1.0 + ws) + P.kappa_2to3_ext_per_s * torch.relu(wi - P.theta_tag) * -ws) - P.kappa_consol_per_s * S_sleep * A_sws * tag * ef * Dh * (1.0 + ws) - P.lambda_consol_per_s * ws * (ws + 0.25) * (ws + 1.0)
        self.w_fast = wf + dts * dwf
        self.w_interm = torch.clamp(wi + dts * dwi, P.w_interm_min, P.w_interm_max)
        self.w_slow = torch.clamp(ws + dts * dws, P.w_slow_min, P.w_slow_max)
        omega = self.omega()
        self.net.sp['w_f'][self.syn_edge] = self.base_f * omega
        self.net.sp['w_s'][self.syn_edge] = self.base_s * omega

    def omega(self):
        return torch.clamp(1.0 + self.w_fast + self.w_interm + self.w_slow, P.omega_min, P.omega_max)

class CXPlasticity:
    W_MIN, W_MAX = (-0.95, 1.0)

    def __init__(self, net, cx):
        dev, dt = (net.device, net.dtype)
        self.net = net
        t = lambda k: torch.as_tensor(cx.get(k, []), dtype=torch.long, device=dev)
        self.er = {k: t('er_' + k) for k in ('edge', 'pre', 'post')}
        self.fb = {k: t('fb_' + k) for k in ('edge', 'pre', 'post')}
        self.eff = {k: t('eff_' + k) for k in ('edge', 'pre', 'post')}
        self.eff_E_rev = torch.as_tensor(cx.get('eff_E_rev', []), dtype=dt, device=dev)
        self.ppm3 = t('ppm3_idx')
        N = net.N
        self.e_trace = torch.zeros(N, dtype=dt, device=dev)
        self.D_fb = torch.zeros((), dtype=dt, device=dev)
        z = lambda d: torch.zeros(d['edge'].numel(), dtype=dt, device=dev)
        self.w = {k: {'fast': z(d), 'slow': z(d)} for k, d in (('er', self.er), ('fb', self.fb))}
        self.w['eff'] = {'fast': z(self.eff)}
        sp = net.sp
        self.base = {k: (sp['w_f'][d['edge']].clone(), sp['w_s'][d['edge']].clone()) for k, d in (('er', self.er), ('fb', self.fb), ('eff', self.eff))}

    @property
    def D_hat_fb(self):
        return torch.relu(self.D_fb - P.theta_dDAT)

    def clock1(self, tilde, dt=P.dt_ms):
        self.e_trace = math.exp(-dt / P.tau_CX_ms) * self.e_trace + 1000.0 / P.tau_CX_ms * tilde
        if self.ppm3.numel():
            self.D_fb = math.exp(-dt / P.tau_DA_ms) * self.D_fb + 1000.0 * P.alpha_DA_s / (P.tau_DA_ms * self.ppm3.numel()) * tilde[self.ppm3].sum()

    def _cascade(self, key, Phi, S, dts):
        w = self.w[key]
        J = w['fast'] / P.tau_CX_slow_s
        w['fast'] = w['fast'] + dts * ((1.0 - S) * Phi - J - P.gamma_SHY_CX_per_s * S * w['fast'])
        w['slow'] = w['slow'] + dts * J
        return torch.clamp(1.0 + w['fast'] + w['slow'], P.omega_min, P.omega_max)

    def _write(self, key, omega):
        e = getattr(self, key)['edge']
        if e.numel():
            self.net.sp['w_f'][e] = self.base[key][0] * omega
            self.net.sp['w_s'][e] = self.base[key][1] * omega

    def clock2(self, S_sleep, dt_ms=P.dt_plast_ms):
        s, dts = (self.net.s, 0.001 * dt_ms)
        c = s['c_ca']
        if self.er['edge'].numel():
            w = self.w['er']
            wt = w['fast'] + w['slow']
            ck = c[self.er['post']]
            Phi = P.eta_ring_per_s * self.e_trace[self.er['pre']] * (-P.alpha_LTD_per_uM * ck * (wt - self.W_MIN) + P.alpha_LTP_per_uM * torch.relu(P.c_thresh_uM - ck) * (self.W_MAX - wt))
            self._write('er', self._cascade('er', Phi, S_sleep, dts))
        if self.fb['edge'].numel():
            w = self.w['fb']
            wt = w['fast'] + w['slow']
            cq = c[self.fb['post']]
            Phi = P.eta_goal_per_s * self.D_hat_fb * self.e_trace[self.fb['pre']] * (cq * (self.W_MAX - wt) - P.beta_LTD * (wt - self.W_MIN))
            self._write('fb', self._cascade('fb', Phi, S_sleep, dts))
        if self.eff['edge'].numel():
            V = s['V'][self.eff['post']]
            dw = dts * 1000.0 * P.eta_eff_per_mV * torch.sign(V - self.eff_E_rev) * (V - s['E_L_mod'][self.eff['post']]) * s['s_bar'][self.eff['pre']]
            w = self.w['eff']
            w['fast'] = torch.clamp(w['fast'] + dw, P.w_eff_min, P.w_eff_max)
            self._write('eff', torch.clamp(1.0 + w['fast'], P.omega_min, P.omega_max))
