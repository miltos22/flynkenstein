import sys
import time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flybrain import connectome, reference as R
CONN = Path(__file__).resolve().parents[1] / 'cache' / 'connectome_v1.npz'

def test_fast_path_matches_event_path_and_is_faster():
    conn = connectome.load(str(CONN))
    nets = [R.Network(conn, device='cuda', fast=f) for f in (False, True)]
    g = torch.Generator(device='cuda').manual_seed(0)
    V0 = nets[0].s['c']['E_L_base_mV'] + 2.5 * torch.randn(nets[0].N, generator=g, device='cuda')
    for net in nets:
        net.s['V'] = V0.clone()
    I = torch.full((nets[0].N,), 6.0, device='cuda')
    mism, spikes = (0, 0)
    for n in range(150):
        for net in nets:
            net.step(I)
        mism += int((nets[0].s['spikes'] != nets[1].s['spikes']).sum())
        spikes += int(nets[0].s['spikes'].sum())
        if n == 10:
            for k in ('g_exc_f', 'g_exc_s', 'g_inh_f', 'g_inh_s'):
                a, b = (nets[0].s[k], nets[1].s[k])
                assert float((a - b).abs().max()) <= 0.0001 * max(float(a.abs().max()), 1.0), k
    assert spikes > 1000, spikes
    assert mism <= 0.001 * spikes, f'{mism} spike mismatches of {spikes}'
    t = {}
    for net, name in zip(nets, ('event', 'fast')):
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(300):
            net.step(I)
        torch.cuda.synchronize()
        t[name] = (time.time() - t0) / 300 * 1000.0
    print(f"  ms/step: event {t['event']:.3f}, fast {t['fast']:.3f}")
    assert t['fast'] < t['event']
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
    sys.exit(1 if fails else 0)
