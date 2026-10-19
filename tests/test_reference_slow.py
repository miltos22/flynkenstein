import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flybrain import params as P
from flybrain import reference as R

def test_tyson_period_is_24h():
    M, Pc, dt = (0.5, 0.5, 1.0)
    trace = []
    for _ in range(6 * 86400):
        M, Pc = R.tyson_step(M, Pc, dt)
        trace.append(Pc)
    tail = trace[2 * 86400:]
    mean = sum(tail) / len(tail)
    ups = [i for i in range(1, len(tail)) if tail[i - 1] < mean <= tail[i]]
    assert len(ups) >= 3, f'no sustained oscillation (crossings {len(ups)})'
    period = (ups[-1] - ups[0]) / (len(ups) - 1) * dt
    want = P.SPEC_CHECKS['circ_period_product'][0]
    assert abs(period - want) / want < 0.01, f'period {period:.0f} s vs spec {want:.0f} s'
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
