import math
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flybrain import params as P
from flybrain import reference as R
CHK = P.SPEC_CHECKS
T = lambda v: torch.tensor([v], dtype=torch.float64)

def close(got, name, atol):
    want = CHK[name][0]
    assert abs(got - want) <= atol, f'{name}: got {got:.5f}, spec {want} (line {CHK[name][2]})'

def test_gating_values():
    close(float(R.B_Mg(T(-45.5))), 'B_at_-45.5mV', 0.001)
    close(float(R.B_Mg(T(-54.0))), 'B_at_-54mV', 0.001)
    close(float(R.m_inf_NaP(T(-45.5))), 'm_inf_NaP_at_-45.5mV', 0.001)
    close(float(R.m_inf_NaP(T(-54.0))), 'm_inf_NaP_at_-54mV', 0.001)
    close(math.exp(P.upstroke_factor), 'e_pow_7_5', 0.005)

def test_nap_inward_current_class_I():
    g = P.CLASS['g_NaP0_nS'][0]
    I = g * float(R.m_inf_NaP(T(-48.5))) * (-48.5 - P.E_Na_mV)
    close(-I, 'I_NaP_at_-48.5mV', 0.005)

def _clamped_graded_calcium(V_clamp, ms=3000.0):
    s = R.init_state(torch.tensor([4]))
    for _ in range(int(ms / P.dt_ms)):
        s['V'] = T(V_clamp)
        R.step_node(s, I_ext=T(0.0))
    return float(s['c_ca'])

def test_graded_calcium_rest_and_max():
    close(_clamped_graded_calcium(-52.0), 'c_rest_grad', 0.0005)
    close(_clamped_graded_calcium(0.0), 'c_max_grad', 0.005)

def _dan_train(period_steps, n_periods, s=None, D=0.0):
    if s is None:
        s = R.init_state(torch.tensor([5]))
    U0 = float(s['U0'])
    Ds, ux = ([], None)
    for n in range(period_steps * n_periods):
        s['V'] = s['c']['E_L_base_mV'].clone()
        spk = n % period_steps == 0
        s['spikes'] = T(1.0 if spk else 0.0)
        tilde = float(R.step_node(s, I_ext=T(0.0)))
        if spk:
            u_m = (float(s['u_stp']) - U0) / (1 - U0)
            ux = (u_m, tilde / u_m)
        D = float(R.dopamine_step(T(D), T(tilde), n_dan=1))
        Ds.append(D)
    return (Ds, ux, s, D)

def test_dan_tonic_3hz():
    per = round(P.T_tonic_ms / P.dt_ms)
    Ds, (u_m, x_m), _, _ = _dan_train(per, 10)
    close(u_m, 'u_minus_333.3ms', 0.0006)
    close(x_m, 'x_minus_333.3ms', 0.0002)
    tail = Ds[-per:]
    close(sum(tail) / len(tail), 'D_tonic_mean', 0.0006)
    close(max(tail), 'D_tonic_peak', 0.0006)
    close(max(0.0, max(tail) - P.theta_dDAT), 'D_hat_tonic', 1e-09)

def test_dan_burst_40hz():
    per = round(P.T_burst_40Hz_ms / P.dt_ms)
    Ds, (u_m, x_m), _, _ = _dan_train(per, 120)
    close(u_m, 'u_minus_25ms', 0.0006)
    close(x_m, 'x_minus_25ms', 0.0006)
    close(40.0 * u_m * x_m, 'release_rate_burst', 0.01)
    tail = Ds[-20 * per:]
    mean = sum(tail) / len(tail)
    close(mean, 'D_burst', 0.01)
    close(mean - P.theta_dDAT, 'D_hat_burst', 0.01)

def test_dan_short_bursts_from_tonic():
    tonic = round(P.T_tonic_ms / P.dt_ms)
    per = round(P.T_burst_40Hz_ms / P.dt_ms)
    for ms, name in ((250.0, 'D_hat_burst_250ms'), (100.0, 'D_hat_burst_100ms')):
        _, _, s, D = _dan_train(tonic, 10)
        Ds, _, _, _ = _dan_train(per, round(ms / P.T_burst_40Hz_ms) + 1, s=s, D=D)
        close(max(0.0, max(Ds) - P.theta_dDAT), name, 0.01)

def test_kappa_zoh_is_exact_formula():
    k = R.kappa_zoh(P.dt_ms)
    for i, tau in enumerate((5.0, 85.0, 6.0, 120.0)):
        exact = (1 - math.exp(-5 * P.dt_ms / tau)) / (1 - math.exp(-P.dt_ms / tau))
        assert abs(float(k[i]) - exact) < 1e-12
if __name__ == '__main__':
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            try:
                fn()
                print('PASS', name)
            except AssertionError as e:
                fails += 1
                print('FAIL', name, '-', e)
    print('kappa_ZOH per channel (exc,f exc,s inh,f inh,s):', [round(float(x), 3) for x in R.kappa_zoh(P.dt_ms)])
    sys.exit(1 if fails else 0)
