import torch
from . import params as P

def ramp(t, t0, t1):
    return min(max((t - t0) / (t1 - t0), 0.0), 1.0)

def boot_gains(t_ms):
    return (ramp(t_ms, P.boot_exc_ramp_start_ms, P.boot_exc_ramp_end_ms), ramp(t_ms, P.boot_inh_ramp_start_ms, P.boot_inh_ramp_end_ms))

def boot(sim, until_ms=P.boot_phase3_end_ms, I_sens=None, seed=0, on_step=None):
    s = sim.net.s
    g = torch.Generator(device=s['V'].device).manual_seed(seed)
    s['V'] = s['c']['E_L_base_mV'] + P.boot_V0_sigma_mV * torch.randn(s['V'].shape, generator=g, device=s['V'].device, dtype=s['V'].dtype)
    I = torch.zeros_like(s['V']) if I_sens is None else I_sens
    for n in range(round(until_ms / P.dt_ms)):
        t = n * P.dt_ms
        ge, gi = boot_gains(t)
        sim.step(I, ge, gi)
        if on_step is not None:
            on_step(n, t)

def equilibrate(sim, seconds=40.0, speed=10.0, I_sens=None, on_window=None, window_s=2.0):
    s = sim.net.s
    I = torch.zeros_like(s['V']) if I_sens is None else I_sens
    sim.slow.homeo_speed = speed
    per = round(window_s * 1000 / P.dt_ms)
    cnt = torch.zeros_like(s['V'])
    try:
        for n in range(round(seconds * 1000 / P.dt_ms)):
            sim.step(I)
            cnt += s['spikes']
            if on_window is not None and n % per == per - 1:
                on_window((n + 1) * P.dt_ms / 1000.0, cnt / window_s)
                cnt.zero_()
    finally:
        sim.slow.homeo_speed = 1.0
    return s['E_L'].clone()
